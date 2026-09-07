"""知识库管理 API：上传文档、删除文档、重建索引。

接口:
  POST   /api/knowledge/upload        - 上传文档并构建索引（支持 async_=true 后台执行）
  DELETE /api/knowledge/{doc_id}      - 删除文档（含关联 chunks）
  POST   /api/knowledge/rebuild       - 重建指定策略的索引（默认后台执行）
  GET    /api/knowledge/status        - 查看索引状态
  GET    /api/knowledge/documents     - 列出所有文档
  GET    /api/knowledge/tasks/{id}    - 查询后台任务状态
  GET    /api/knowledge/tasks         - 列出后台任务（可按 type 过滤）

存储策略:
  - 所有写入通过 IndexWriter 统一分发到 FAISS + metadata.json + MySQL + ES
  - config.storage.backends.mysql.enabled=true: chunks 写入 MySQL（按 strategy 隔离）
  - config.storage.backends.es.enabled=true: chunks 写入 ES（按 strategy 隔离索引）
  - MySQL/ES 软失败：写入异常不影响索引构建
"""
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.audit.logger import record
from app.auth.dependencies import AuthUser, get_current_user, require_permission
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# 可复用决策逻辑抽到独立模块（P2-5：减负路由文件），以原名回导出保持外部/测试兼容
from app.acl.service import (  # noqa: E402
    can_delete as _can_delete,
    can_manage_acl as _can_manage_acl,
    readable_document_ids as _readable_document_ids,
)
from app.ingestion.loader.registry import LOADER_MAP as _LOADER_MAP  # noqa: E402


def _current_tenant(user: AuthUser) -> str:
    """当前请求的租户（auth 关闭时回落 default）。"""
    return user.tenant_id if user else "default"

# 上传大小限制（50MB），防止超大文件打爆内存
_MAX_UPLOAD_SIZE = 50 * 1024 * 1024


async def _stream_upload_to_file(file: UploadFile, save_path: Path, max_size: int) -> int:
    """流式写盘上传内容，超过 max_size 立即 413 拒绝（P2-1）。

    原实现把整个上传聚合为一个内存 bytes（b"".join 单次持有最高 max_size 字节）——
    改为边读边写，内存占用恒定。返回实际写入字节数。
    """
    total = 0
    try:
        with open(save_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail="文件大小超过限制: {} bytes".format(max_size),
                    )
                out.write(chunk)
    except HTTPException:
        # 超限：清理部分写入文件，旧源回滚由调用方处理
        try:
            save_path.unlink()
        except OSError:
            pass
        raise
    return total


class UploadResponse(BaseModel):
    """上传响应。"""
    strategy: str
    document_count: int
    chunk_count: int
    dimension: int
    index_path: str
    metadata_path: str
    task_id: str = None
    async_: bool = False


class RebuildRequest(BaseModel):
    """重建索引请求。"""
    strategy: str = Field("recursive", description="分块策略: fixed/recursive")


class RebuildResponse(BaseModel):
    """重建索引响应。"""
    strategy: str
    document_count: int
    chunk_count: int
    dimension: int
    task_id: str = None
    async_: bool = False


class DeleteResponse(BaseModel):
    """删除文档响应。

    rebuilt_indexes: 已提交后台重建的索引策略列表（异步执行，完成后索引才真正更新）。
    task_id: 后台重建任务 ID（可到 /api/knowledge/tasks 查询状态）。
    """
    document_id: str
    deleted_from_mysql: bool
    deleted_chunks: int
    deleted_file: bool
    rebuilt_indexes: List[str]
    task_id: Optional[str] = None


class DocumentItem(BaseModel):
    document_id: str
    file_name: str
    content_length: int = 0
    source: str = ""
    owner_user_id: str = ""
    chunk_count: int = 0
    created_at: str = ""


class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]
    total: int


class StatusResponse(BaseModel):
    """索引状态响应。"""
    indexes: dict


class TaskStatusResponse(BaseModel):
    task_id: str
    type: str
    status: str
    progress: float | None = None
    result: dict | None = None
    error: str | None = None
    elapsed: float = 0.0


@router.post(
    "/upload",
    response_model=UploadResponse,
    dependencies=[Depends(require_permission("knowledge:upload"))],
)
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = Form("recursive"),
    async_: bool = False,
    user: AuthUser = Depends(get_current_user),
):
    """上传文档并构建索引。

    接收文件 → 保存到 data/raw/{tenant}/ → IndexWriter.write() → 返回结果。
    租户隔离：文件保存到当前用户租户目录，索引构建按租户隔离。

    参数:
      - file: 上传的文件（支持 .txt/.html/.pdf/.docx）
      - strategy: 分块策略 fixed/recursive（表单字段，前端 FormData 传递）
      - async_: 是否后台异步执行（大文件推荐 true，query 参数 ?async_=true）
    """
    if strategy not in ("fixed", "recursive"):
        raise HTTPException(status_code=400, detail="strategy 必须为 fixed 或 recursive")

    tenant_id = _current_tenant(user)
    owner_user_id = user.user_id if user else ""

    # 检查文件类型
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _LOADER_MAP:
        raise HTTPException(
            status_code=400,
            detail="不支持的文件类型: {}，支持: {}".format(suffix, list(_LOADER_MAP.keys())),
        )

    # 保存到 data/raw/{tenant}/
    from app.core.config import Config
    raw_dir = Config().raw_dir_for(tenant_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_path = raw_dir / Path(file.filename or "").name

    # P1-3: 覆盖更新前先备份旧源文件，索引失败时回滚，避免丢失回滚源。
    #   backup 后缀 '.bak' 不在 loader 支持列表，不会被 rebuild/扫描误读。
    backup_path = None
    if save_path.exists():
        backup_path = raw_dir / (save_path.name + ".bak")
        try:
            import shutil as _shutil
            _shutil.copy2(save_path, backup_path)
            logger.info("已备份旧源文件: %s → %s", save_path, backup_path)
        except Exception as be:
            logger.error("备份旧源文件失败，中止上传: %s", be, exc_info=True)
            raise HTTPException(status_code=500, detail="备份旧源文件失败: {}".format(be))

    # P2-1: 流式写盘，不把上传整包聚合进内存；超限(413)时回滚旧源
    try:
        total = await _stream_upload_to_file(file, save_path, _MAX_UPLOAD_SIZE)
    except HTTPException:
        if backup_path is not None and backup_path.exists():
            try:
                backup_path.replace(save_path)
            except Exception:
                pass
        raise
    logger.info(
        "文件已保存: %s (%d bytes, tenant=%s)", save_path, total, tenant_id
    )

    # 后台异步执行
    if async_:
        from app.core.task_queue import task_manager
        task_id = task_manager.submit(
            "upload", _do_upload, save_path, strategy, tenant_id, owner_user_id,
            backup_path,
        )
        record(
            action="document.upload", tenant_id=tenant_id,
            actor_user_id=user.user_id if user else "",
            actor_username=user.username if user else "",
            resource=save_path.name, detail="strategy={}, async=true".format(strategy),
        )
        return UploadResponse(
            strategy=strategy,
            document_count=0,
            chunk_count=0,
            dimension=0,
            index_path=str(Path("data/index") / strategy / "faiss.index"),
            metadata_path=str(Path("data/index") / strategy / "metadata.json"),
            task_id=task_id,
            async_=True,
        )

    # 同步执行（用线程池避免阻塞事件循环）
    from starlette.concurrency import run_in_threadpool
    try:
        result = await run_in_threadpool(
            _do_upload, save_path, strategy, tenant_id, owner_user_id, backup_path
        )
    except Exception as e:
        logger.error("索引构建失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="索引构建失败: {}".format(e))

    record(
        action="document.upload", tenant_id=tenant_id,
        actor_user_id=user.user_id if user else "",
        actor_username=user.username if user else "",
        resource=save_path.name,
        detail="strategy={}, chunks={}".format(strategy, result["chunk_count"]),
    )
    return UploadResponse(
        strategy=strategy,
        document_count=result["document_count"],
        chunk_count=result["chunk_count"],
        dimension=result["dimension"],
        index_path=result["index_path"],
        metadata_path=result["metadata_path"],
    )


def _do_upload(save_path: Path, strategy: str, tenant_id: str = "default",
               owner_user_id: str = "", backup_path: Optional[Path] = None) -> dict:
    """上传单个文档并构建索引（通过 IndexWriter 统一写入）。

    返回 dict 形式的 UploadResponse 数据，供 task_manager 查询时返回。

    backup_path 非空时，成功则删除备份、失败则把旧源文件回滚到 save_path
    （P1-3：覆盖更新不丢回滚源）。回滚放在 finally 前的包边界处理，保证
    无论成功/失败旧源都可恢复。
    """
    import time as _time
    t = _time.time()
    logger.info(
        "_do_upload 开始: file=%s, strategy=%s, tenant=%s, owner=%s, backup=%s",
        save_path.name, strategy, tenant_id, owner_user_id or "-",
        backup_path if backup_path is not None else "-",
    )

    try:
        from app.ingestion.writer import IndexWriter

        writer = IndexWriter()
        document = writer._load_single_document(save_path)
        result = writer.write(
            documents=[document],
            strategy=strategy,
            tenant_id=tenant_id,
            owner_user_id=owner_user_id,
        )

        # 索引已变更，无需清空缓存：get_service 按索引版本号自动刷新 service，
        # embedding/reranker 模型继续复用
        logger.info("索引已更新（upload 完成，service 缓存将按版本号自动刷新）")

        # 索引构建成功：删除旧源备份
        if backup_path is not None and backup_path.exists():
            try:
                backup_path.unlink()
                logger.info("上传成功，已删除旧源备份: %s", backup_path)
            except Exception as be:
                logger.warning("删除旧源备份失败（可手动清理）: %s", be)
    except Exception as e:
        # 索引构建失败：回滚旧源文件，保证不丢回滚源
        if backup_path is not None and backup_path.exists():
            try:
                backup_path.replace(save_path)
                logger.warning(
                    "索引构建失败，已回滚旧源文件: %s", save_path,
                )
            except Exception as be:
                logger.error("回滚旧源文件失败: %s（旧源保留于 %s）", be, backup_path)
        logger.error(
            "_do_upload 失败: %.3fs, file=%s, strategy=%s, tenant=%s, error=%s: %s",
            _time.time() - t, save_path.name, strategy, tenant_id,
            type(e).__name__, e, exc_info=True,
        )
        raise

    logger.info(
        "_do_upload 完成: %.3fs, file=%s, strategy=%s, tenant=%s, docs=%d, chunks=%d, "
        "dim=%d, mysql=%s, es=%s",
        _time.time() - t, save_path.name, strategy, tenant_id,
        result["document_count"], result["chunk_count"],
        result["dimension"], result["mysql_persisted"],
        result["es_persisted"],
    )
    return {
        "strategy": strategy,
        "document_count": result["document_count"],
        "chunk_count": result["chunk_count"],
        "dimension": result["dimension"],
        "index_path": result["index_path"],
        "metadata_path": result["metadata_path"],
    }


@router.post(
    "/rebuild",
    response_model=RebuildResponse,
    dependencies=[Depends(require_permission("knowledge:rebuild"))],
)
async def rebuild_index(req: RebuildRequest, async_: bool = True,
                        user: AuthUser = Depends(get_current_user)):
    """重建指定策略的索引（使用 data/raw/{tenant}/ 下已有文档）。

    参数:
      - strategy: 分块策略 fixed/recursive
      - async_: 是否后台异步执行，默认 True（rebuild 耗时较长）
    租户隔离：仅重建当前用户租户的索引。
    """
    if req.strategy not in ("fixed", "recursive"):
        raise HTTPException(status_code=400, detail="strategy 必须为 fixed 或 recursive")

    tenant_id = _current_tenant(user)

    from app.core.config import Config
    raw_dir = Config().raw_dir_for(tenant_id)
    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail="data/raw 目录不存在 (tenant={})".format(tenant_id))

    # 收集所有支持的文件
    files = [f for f in raw_dir.iterdir() if f.suffix.lower() in _LOADER_MAP]
    if not files:
        raise HTTPException(status_code=404, detail="data/raw 下无可处理的文件 (tenant={})".format(tenant_id))

    logger.info(
        "重建索引: strategy=%s, tenant=%s, files=%d, async=%s",
        req.strategy, tenant_id, len(files), async_,
    )

    # 后台异步执行
    if async_:
        from app.core.task_queue import task_manager
        task_id = task_manager.submit("rebuild", _do_rebuild, req.strategy, tenant_id)
        record(
            action="document.rebuild", tenant_id=tenant_id,
            actor_user_id=user.user_id if user else "",
            actor_username=user.username if user else "",
            resource=req.strategy, detail="async=true",
        )
        return RebuildResponse(
            strategy=req.strategy,
            document_count=0,
            chunk_count=0,
            dimension=0,
            task_id=task_id,
            async_=True,
        )

    # 同步执行（用线程池避免阻塞事件循环）
    from starlette.concurrency import run_in_threadpool
    try:
        result = await run_in_threadpool(_do_rebuild, req.strategy, tenant_id)
    except Exception as e:
        logger.error("重建索引失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="重建索引失败: {}".format(e))

    record(
        action="document.rebuild", tenant_id=tenant_id,
        actor_user_id=user.user_id if user else "",
        actor_username=user.username if user else "",
        resource=req.strategy,
        detail="chunks={}".format(result["chunk_count"]),
    )
    return RebuildResponse(
        strategy=req.strategy,
        document_count=result["document_count"],
        chunk_count=result["chunk_count"],
        dimension=result["dimension"],
    )


def _do_rebuild(strategy: str, tenant_id: str = "default") -> dict:
    """幂等重建索引（通过 IndexWriter.rebuild 统一清理 + 写入）。

    返回 dict 形式的 RebuildResponse 数据。
    """
    import time as _time
    t = _time.time()
    logger.info("_do_rebuild 开始: strategy=%s, tenant=%s", strategy, tenant_id)

    try:
        from app.ingestion.writer import IndexWriter

        writer = IndexWriter()
        result = writer.rebuild(strategy=strategy, tenant_id=tenant_id)

        # 索引已变更，无需清空缓存：get_service 按索引版本号自动刷新 service
        logger.info("索引已更新（rebuild 完成，service 缓存将按版本号自动刷新）")

        logger.info(
            "_do_rebuild 完成: %.3fs, strategy=%s, tenant=%s, docs=%d, chunks=%d, "
            "dim=%d, mysql_deleted=%d, es_dropped=%s, mysql=%s, es=%s",
            _time.time() - t, strategy, tenant_id,
            result["document_count"], result["chunk_count"],
            result["dimension"], result.get("mysql_deleted", 0),
            result.get("es_dropped", False),
            result["mysql_persisted"], result["es_persisted"],
        )
        return {
            "strategy": strategy,
            "document_count": result["document_count"],
            "chunk_count": result["chunk_count"],
            "dimension": result["dimension"],
        }
    except Exception as e:
        logger.error(
            "_do_rebuild 失败: %.3fs, strategy=%s, tenant=%s, error=%s: %s",
            _time.time() - t, strategy, tenant_id, type(e).__name__, e, exc_info=True,
        )
        raise


@router.get(
    "/status",
    response_model=StatusResponse,
    dependencies=[Depends(require_permission("knowledge:read"))],
)
async def index_status(user: AuthUser = Depends(get_current_user)):
    """查看当前租户各策略的索引状态。"""
    from app.core.config import Config
    config = Config()
    tenant_id = _current_tenant(user)
    indexes = {}
    for strategy in ("fixed", "recursive"):
        index_dir = config.index_dir_for(strategy, tenant_id)
        faiss_path = index_dir / "faiss.index"
        meta_path = index_dir / "metadata.json"
        chunk_count = 0
        if meta_path.exists():
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                chunk_count = len(json.load(f))
        indexes[strategy] = {
            "faiss_exists": faiss_path.exists(),
            "metadata_exists": meta_path.exists(),
            "chunk_count": chunk_count,
        }
    return StatusResponse(indexes=indexes)


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    dependencies=[Depends(require_permission("knowledge:read"))],
)
async def get_task(task_id: str):
    """查询后台任务状态。

    status: pending / running / done / failed
    result: 任务完成后返回的数据（done 时）
    error: 失败原因（failed 时）
    """
    from app.core.task_queue import task_manager
    info = task_manager.get(task_id)
    if info is None:
        raise HTTPException(status_code=404, detail="任务不存在: {}".format(task_id))
    return TaskStatusResponse(
        task_id=info["task_id"],
        type=info["type"],
        status=info["status"],
        progress=info.get("progress"),
        result=info.get("result"),
        error=info.get("error"),
        elapsed=info.get("elapsed", 0.0),
    )


@router.get(
    "/tasks",
    dependencies=[Depends(require_permission("knowledge:read"))],
)
async def list_tasks(task_type: str = None, limit: int = 50):
    """列出后台任务（可按 type 过滤）。"""
    from app.core.task_queue import task_manager
    if task_type:
        tasks = task_manager.list_by_type(task_type, limit=limit)
    else:
        # 无 type 过滤：合并所有类型
        all_tasks = []
        for t in ("upload", "rebuild"):
            all_tasks.extend(task_manager.list_by_type(t, limit=limit))
        all_tasks.sort(key=lambda x: x["started_at"], reverse=True)
        tasks = all_tasks[:limit]
    return {"tasks": tasks, "total": len(tasks)}


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    dependencies=[Depends(require_permission("knowledge:delete"))],
)
async def delete_document(doc_id: str, user: AuthUser = Depends(get_current_user)):
    """删除文档：MySQL 记录 + 原始文件 + 重建受影响的索引。

    流程:
      1. 从 MySQL 删除文档记录（CASCADE 自动删 chunks，按租户过滤）
      2. 从 data/raw/{tenant}/ 删除原始文件
      3. 重建 fixed/recursive 两个索引（基于剩余文件）

    若 storage.enabled=false，则仅做文件删除 + 索引重建。
    租户隔离：所有操作限定在当前用户租户，防止跨租户误删。
    所有同步阻塞操作通过线程池执行，避免阻塞事件循环。
    """
    from starlette.concurrency import run_in_threadpool

    tenant_id = _current_tenant(user)

    class _NotFoundError(Exception):
        pass

    class _MysqlError(Exception):
        pass

    def _do_delete():
        from app.core.config import Config
        config = Config()
        deleted_from_mysql = False
        deleted_chunks = 0
        deleted_file = False
        resolved_doc_id = doc_id
        deleted_doc_ids_for_rebuild: List[str] = []
        vector_ids: List[int] = []

        # 0. 先定位原始文件（doc_id 可能是 document_id / 文件名 / 带后缀），拿到文件 stem
        raw_dir = config.raw_dir_for(tenant_id)
        target = None
        file_stem = Path(doc_id).stem if Path(doc_id).suffix else doc_id
        if raw_dir.exists():
            if (raw_dir / doc_id).exists():
                target = raw_dir / doc_id
            else:
                for f in raw_dir.iterdir():
                    if f.stem == doc_id or f.name == doc_id or f.stem == file_stem:
                        target = f
                        break
            if target is not None and target.exists():
                file_stem = target.stem

        # 1. 删除 MySQL 记录（按 document_id + tenant；用候选 id + 文件名变体回查）
        if config.storage_mysql_enabled:
            try:
                from app.storage import DocumentRepository, ChunkRepository
                doc_repo = DocumentRepository()
                chunk_repo = ChunkRepository(tenant_id=tenant_id)

                existing = None
                # 候选 document_id：原始 doc_id、文件 stem、去后缀的 doc_id
                candidates = [doc_id, file_stem, Path(doc_id).stem]
                for c in candidates:
                    if not c:
                        continue
                    existing = doc_repo.get(c, tenant_id=tenant_id)
                    if existing:
                        break
                # 兜底：file_name 变体匹配（完整路径 / 纯文件名）
                if existing is None:
                    for row in doc_repo.list_all(limit=10000, tenant_id=tenant_id):
                        fn = row.get("file_name") or ""
                        if (fn == doc_id or fn == file_stem
                                or Path(fn).name in (doc_id, file_stem)
                                or fn.endswith("/" + doc_id) or fn.endswith("\\" + doc_id)):
                            existing = row
                            break
                if existing:
                    resolved_doc_id = existing["document_id"]
                    # 文档级 ACL：删除前校验权限（superadmin / owner / delete 授权）
                    if not _can_delete(user, tenant_id, resolved_doc_id):
                        raise HTTPException(
                            status_code=403,
                            detail="无权删除文档: {}（非归属人或未授权）".format(resolved_doc_id),
                        )
                    # 稳定 ID 索引：先取该文档全部 chunk 的 vector_id（删除前）
                    vector_ids = chunk_repo.get_vector_ids_by_document(resolved_doc_id)
                    deleted_chunks = chunk_repo.delete_by_document(resolved_doc_id)
                    doc_repo.delete(resolved_doc_id, tenant_id=tenant_id)
                    # 文档级 ACL：级联清理该文档的全部授权记录，
                    # 防止孤儿 ACL 在 document_id 复用（同名文件重新上传）时静默挂到新文档
                    try:
                        from app.acl.repository import ACLRepository
                        acl_removed = ACLRepository().delete_by_document(resolved_doc_id)
                        if acl_removed:
                            logger.info(
                                "已清理文档 ACL 记录: doc_id=%s, 行数=%d",
                                resolved_doc_id, acl_removed,
                            )
                    except Exception as acl_err:
                        logger.warning(
                            "清理文档 ACL 记录失败: doc_id=%s, %s",
                            resolved_doc_id, acl_err, exc_info=True,
                        )
                    deleted_from_mysql = True
                    deleted_doc_ids_for_rebuild.append(resolved_doc_id)
                    logger.info(
                        "已从 MySQL 删除文档: doc_id=%s (原始=%s), tenant=%s, chunks=%d, vector_ids=%d",
                        resolved_doc_id, doc_id, tenant_id, deleted_chunks, len(vector_ids),
                    )
                else:
                    logger.warning("MySQL 中未找到文档: %s (tenant=%s)", doc_id, tenant_id)
            except HTTPException:
                # ACL 403 等由 FastAPI 直接处理，不包装为 _MysqlError
                raise
            except Exception as e:
                logger.error("MySQL 删除失败: %s", e, exc_info=True)
                raise _MysqlError(str(e))

        # 2. 删除原始文件
        if target is not None and target.exists():
            if target.stem not in deleted_doc_ids_for_rebuild:
                deleted_doc_ids_for_rebuild.append(target.stem)
            target.unlink()
            deleted_file = True
            logger.info("已删除原始文件: %s (tenant=%s)", target, tenant_id)

        if not deleted_from_mysql and not deleted_file:
            raise _NotFoundError(doc_id)

        # 2b. 文档级 ACL：file-only 删除路径（无 MySQL 记录）也校验权限
        #     （MySQL 路径已在步骤 1 删除前校验过）
        if not deleted_from_mysql and not _can_delete(user, tenant_id, resolved_doc_id):
            raise HTTPException(
                status_code=403,
                detail="无权删除文档: {}（非归属人或未授权）".format(resolved_doc_id),
            )

        # P1-2: MySQL 未启用时，删除原始文件后向量/metadata/ES 会残留孤儿。
        #   从 metadata.json 按 document_id 派生该文档的 vector_ids，
        #   供步骤 3 的 remove_document 稳定 ID 清理（无需重建）。
        if not deleted_from_mysql and not vector_ids:
            import json as _json
            # 上传时 document_id 取文件名 stem（见 loaders）
            resolved_doc_id = file_stem or resolved_doc_id
            for strategy in ("fixed", "recursive"):
                meta_path = config.index_dir_for(strategy, tenant_id) / "metadata.json"
                if not meta_path.exists():
                    continue
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        entries = _json.load(f)
                except Exception as e:
                    logger.warning(
                        "读取 metadata.json 失败（跳过策略 %s）: %s", strategy, e,
                    )
                    continue
                for vid, e in entries.items():
                    if e.get("document_id") == resolved_doc_id:
                        raw = e.get("vector_id", vid)
                        try:
                            vector_ids.append(int(raw))
                        except (TypeError, ValueError):
                            pass
            vector_ids = sorted(set(vector_ids))
            logger.info(
                "MySQL 未启用，从 metadata.json 派生 vector_ids: doc=%s, count=%d",
                resolved_doc_id, len(vector_ids),
            )

        return {
            "deleted_from_mysql": deleted_from_mysql,
            "deleted_chunks": deleted_chunks,
            "deleted_file": deleted_file,
            "deleted_doc_ids": deleted_doc_ids_for_rebuild,
            "vector_ids": vector_ids,
            "document_id": resolved_doc_id,
        }

    try:
        r = await run_in_threadpool(_do_delete)
    except _NotFoundError:
        raise HTTPException(status_code=404, detail="文档不存在: {}".format(doc_id))
    except _MysqlError as e:
        raise HTTPException(status_code=500, detail="MySQL 删除失败: {}".format(e))

    # 3. 稳定 ID 索引：从向量后端 + metadata.json + ES 移除该文档数据（无需重建）
    #    FAISS/Milvus 按 vector_id 删除、metadata 按 vector_id 摘除、ES 按文档删除
    if r["vector_ids"]:
        from app.ingestion.writer import IndexWriter
        writer = IndexWriter()
        orphan_issues = []
        for strategy in ("fixed", "recursive"):
            try:
                issues = writer.remove_document(
                    strategy=strategy, tenant_id=tenant_id,
                    document_id=r["document_id"], vector_ids=r["vector_ids"],
                    raise_on_orphan=False,
                )
                if issues:
                    orphan_issues.extend("[{}] {}".format(strategy, i) for i in issues)
            except Exception as e:
                orphan_issues.append("[{}] {}".format(strategy, e))
        # P1-1: 孤儿清理失败不再静默——聚合后以 5xx 暴露，提示对账/重建修复
        if orphan_issues:
            logger.error(
                "文档 %s 删除后向量/元数据清理失败（MySQL/文件已删除，存在孤儿）: %s",
                r["document_id"], "; ".join(orphan_issues),
            )
            raise HTTPException(
                status_code=500,
                detail="文档已删除，但向量/元数据清理失败（存在孤儿），请运行对账/重建修复",
            )

    record(
        action="document.delete", tenant_id=tenant_id,
        actor_user_id=user.user_id if user else "",
        actor_username=user.username if user else "",
        resource=doc_id,
        detail="deleted_chunks={}, vectors_removed={}".format(
            r["deleted_chunks"], len(r["vector_ids"]),
        ),
    )
    return DeleteResponse(
        document_id=doc_id,
        deleted_from_mysql=r["deleted_from_mysql"],
        deleted_chunks=r["deleted_chunks"],
        deleted_file=r["deleted_file"],
        rebuilt_indexes=[],
        task_id=None,
    )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    dependencies=[Depends(require_permission("knowledge:read"))],
)
async def list_documents(user: AuthUser = Depends(get_current_user)):
    """列出当前租户内「用户可读」的文档（优先从 MySQL 读取，否则扫描 data/raw/{tenant}/）。

    文档级 ACL：非 superadmin 仅能看到归属自己 / 被授权 / 存量共享的文档。
    """
    from starlette.concurrency import run_in_threadpool

    tenant_id = _current_tenant(user)

    def _do_list():
        from app.core.config import Config
        config = Config()
        documents: List[DocumentItem] = []

        if config.storage_mysql_enabled:
            try:
                from app.storage import DocumentRepository, ChunkRepository
                doc_repo = DocumentRepository()
                chunk_repo = ChunkRepository(tenant_id=tenant_id)
                # 文档级 ACL：仅 MySQL 为数据源时按可读文档过滤
                readable = _readable_document_ids(user, tenant_id)
                rows = [
                    r for r in doc_repo.list_all(limit=1000, tenant_id=tenant_id)
                    if readable is None or r["document_id"] in readable
                ]
                # P2-2：批量统计 chunk 数，避免逐文档 COUNT（N+1）
                counts = chunk_repo.counts_by_documents(
                    [r["document_id"] for r in rows]
                )
                for row in rows:
                    doc_id = row["document_id"]
                    documents.append(DocumentItem(
                        document_id=doc_id,
                        file_name=row.get("file_name", ""),
                        content_length=row.get("content_length", 0),
                        source=row.get("source") or "",
                        owner_user_id=row.get("owner_user_id") or "",
                        chunk_count=counts.get(doc_id, 0),
                        created_at=str(row.get("created_at", "")),
                    ))
                return DocumentListResponse(documents=documents, total=len(documents))
            except Exception as e:
                logger.error("MySQL 查询文档失败，回退到文件扫描: %s", e)

        # 回退：扫描 data/raw/{tenant}/（无 owner 元数据，不做文档级 ACL 过滤）
        raw_dir = config.raw_dir_for(tenant_id)
        if raw_dir.exists():
            for f in raw_dir.iterdir():
                if f.is_file() and f.suffix.lower() in _LOADER_MAP:
                    documents.append(DocumentItem(
                        document_id=f.stem,
                        file_name=f.name,
                        content_length=f.stat().st_size,
                    ))
        return DocumentListResponse(documents=documents, total=len(documents))

    return await run_in_threadpool(_do_list)


# ---------------- 文档级 ACL 管理 ----------------

class GrantRequest(BaseModel):
    """文档授权请求。"""
    principal_type: str = Field(..., description="授权主体类型: user / role")
    principal_id: str = Field(..., description="user_id 或 role_code")
    permission: str = Field(..., description="权限: read / write / delete")


@router.get(
    "/{doc_id}/acl",
    dependencies=[Depends(require_permission("knowledge:grant"))],
)
async def list_document_acl(doc_id: str, user: AuthUser = Depends(get_current_user)):
    """列出某文档的授权（需 knowledge:grant + 文档归属人/superadmin）。"""
    from starlette.concurrency import run_in_threadpool

    tenant_id = _current_tenant(user)

    def _do():
        if not _can_manage_acl(user, tenant_id, doc_id):
            raise HTTPException(status_code=403, detail="无权管理该文档授权")
        from app.acl.repository import ACLRepository
        return ACLRepository().list_grants(doc_id)

    grants = await run_in_threadpool(_do)
    return {"document_id": doc_id, "grants": grants}


@router.post(
    "/{doc_id}/acl",
    dependencies=[Depends(require_permission("knowledge:grant"))],
)
async def grant_document_acl(doc_id: str, req: GrantRequest,
                             user: AuthUser = Depends(get_current_user)):
    """授权用户/角色访问某文档（需 knowledge:grant + 文档归属人/superadmin）。"""
    from starlette.concurrency import run_in_threadpool

    tenant_id = _current_tenant(user)

    def _do():
        if not _can_manage_acl(user, tenant_id, doc_id):
            raise HTTPException(status_code=403, detail="无权管理该文档授权")
        from app.acl.repository import ACLRepository
        ACLRepository().grant(doc_id, req.principal_type, req.principal_id, req.permission)

    await run_in_threadpool(_do)
    record(
        action="document.grant", tenant_id=tenant_id,
        actor_user_id=user.user_id if user else "",
        actor_username=user.username if user else "",
        resource=doc_id,
        detail="{}({}) +{}".format(req.principal_id, req.principal_type, req.permission),
    )
    return {"document_id": doc_id, "granted": req.model_dump()}


@router.delete(
    "/{doc_id}/acl",
    dependencies=[Depends(require_permission("knowledge:grant"))],
)
async def revoke_document_acl(doc_id: str, principal_type: str = None,
                              principal_id: str = None, permission: str = None,
                              user: AuthUser = Depends(get_current_user)):
    """撤销某文档的授权（需 knowledge:grant + 文档归属人/superadmin）。

    参数均可选，留空则撤销匹配范围内的全部授权。
    """
    from starlette.concurrency import run_in_threadpool

    tenant_id = _current_tenant(user)

    def _do():
        if not _can_manage_acl(user, tenant_id, doc_id):
            raise HTTPException(status_code=403, detail="无权管理该文档授权")
        from app.acl.repository import ACLRepository
        return ACLRepository().revoke(
            doc_id, principal_type=principal_type,
            principal_id=principal_id, permission=permission,
        )

    removed = await run_in_threadpool(_do)
    record(
        action="document.revoke", tenant_id=tenant_id,
        actor_user_id=user.user_id if user else "",
        actor_username=user.username if user else "",
        resource=doc_id,
        detail="removed={}".format(removed),
    )
    return {"document_id": doc_id, "removed": removed}
