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

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.audit.logger import record
from app.auth.dependencies import AuthUser, get_current_user, require_permission
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


def _current_tenant(user: AuthUser) -> str:
    """当前请求的租户（auth 关闭时回落 default）。"""
    return user.tenant_id if user else "default"


def _readable_document_ids(user: AuthUser, tenant_id: str):
    """计算当前用户可读文档集合；None 表示不设文档级过滤（鉴权关闭/superadmin）。

    软失败：ACL 查询异常时回退为不设过滤，避免阻断主流程。
    """
    if user is None or user.is_superadmin:
        return None
    try:
        from app.acl.repository import ACLRepository
        return ACLRepository().get_readable_document_ids(user, tenant_id)
    except Exception as e:
        logger.warning("ACL 可读文档计算失败，回退为不设文档级过滤: %s", e)
        return None


def _can_delete(user: AuthUser, tenant_id: str, document_id: str) -> bool:
    """判断用户是否有权删除文档（superadmin / owner / delete 授权）。

    软失败：ACL 查询异常时放行（与项目软失败约定一致）。
    """
    if user is None or user.is_superadmin:
        return True
    try:
        from app.acl.repository import ACLRepository
        return ACLRepository().has_permission(user, document_id, "delete", tenant_id)
    except Exception as e:
        logger.warning("ACL 删除权限判定失败，放行: %s", e)
        return True


def _can_manage_acl(user: AuthUser, tenant_id: str, document_id: str) -> bool:
    """判断用户是否有权管理文档授权（superadmin / 文档归属人）。"""
    if user is None or user.is_superadmin:
        return True
    try:
        from app.storage.document_repository import DocumentRepository
        doc = DocumentRepository().get(document_id, tenant_id=tenant_id)
        return doc is not None and doc.get("owner_user_id") == user.user_id
    except Exception as e:
        logger.warning("ACL 管理权限判定失败: %s", e)
        return False

# 支持的文件类型 → 对应 loader
_LOADER_MAP = {
    ".txt": "app.ingestion.loader.txt_loader.TxtLoader",
    ".html": "app.ingestion.loader.html_loader.HtmlLoader",
    ".pdf": "app.ingestion.loader.pdf_loader.PdfLoader",
    ".docx": "app.ingestion.loader.word_loader.WordLoader",
}

# 上传大小限制（50MB），防止超大文件打爆内存
_MAX_UPLOAD_SIZE = 50 * 1024 * 1024


async def _read_upload_limited(file: UploadFile, max_size: int) -> bytes:
    """分块读取上传文件，超过 max_size 直接拒绝，避免一次性读入内存。"""
    chunks = []
    total = 0
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
        chunks.append(chunk)
    return b"".join(chunks)


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
    strategy: str = "recursive",
    async_: bool = False,
    user: AuthUser = Depends(get_current_user),
):
    """上传文档并构建索引。

    接收文件 → 保存到 data/raw/{tenant}/ → IndexWriter.write() → 返回结果。
    租户隔离：文件保存到当前用户租户目录，索引构建按租户隔离。

    参数:
      - file: 上传的文件（支持 .txt/.html/.pdf/.docx）
      - strategy: 分块策略 fixed/recursive
      - async_: 是否后台异步执行（大文件推荐 true）
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

    content = await _read_upload_limited(file, _MAX_UPLOAD_SIZE)
    save_path.write_bytes(content)
    logger.info(
        "文件已保存: %s (%d bytes, tenant=%s)", save_path, len(content), tenant_id
    )

    # 后台异步执行
    if async_:
        from app.core.task_queue import task_manager
        task_id = task_manager.submit(
            "upload", _do_upload, save_path, strategy, tenant_id, owner_user_id,
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
            _do_upload, save_path, strategy, tenant_id, owner_user_id
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
               owner_user_id: str = "") -> dict:
    """上传单个文档并构建索引（通过 IndexWriter 统一写入）。

    返回 dict 形式的 UploadResponse 数据，供 task_manager 查询时返回。
    """
    import time as _time
    t = _time.time()
    logger.info(
        "_do_upload 开始: file=%s, strategy=%s, tenant=%s, owner=%s",
        save_path.name, strategy, tenant_id, owner_user_id or "-",
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
    except Exception as e:
        logger.error(
            "_do_upload 失败: %.3fs, file=%s, strategy=%s, tenant=%s, error=%s: %s",
            _time.time() - t, save_path.name, strategy, tenant_id,
            type(e).__name__, e, exc_info=True,
        )
        raise


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
        for strategy in ("fixed", "recursive"):
            writer.remove_document(
                strategy=strategy, tenant_id=tenant_id,
                document_id=r["document_id"], vector_ids=r["vector_ids"],
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
                for row in doc_repo.list_all(limit=1000, tenant_id=tenant_id):
                    doc_id = row["document_id"]
                    if readable is not None and doc_id not in readable:
                        continue
                    documents.append(DocumentItem(
                        document_id=doc_id,
                        file_name=row.get("file_name", ""),
                        content_length=row.get("content_length", 0),
                        source=row.get("source") or "",
                        owner_user_id=row.get("owner_user_id") or "",
                        chunk_count=len(chunk_repo.get_by_document(doc_id)),
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
