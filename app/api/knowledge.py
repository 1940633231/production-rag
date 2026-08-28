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
from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

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
    """删除文档响应。"""
    document_id: str
    deleted_from_mysql: bool
    deleted_chunks: int
    deleted_file: bool
    rebuilt_indexes: List[str]


class DocumentItem(BaseModel):
    document_id: str
    file_name: str
    content_length: int = 0
    source: str = ""
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


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    strategy: str = "recursive",
    async_: bool = False,
):
    """上传文档并构建索引。

    接收文件 → 保存到 data/raw/ → IndexWriter.write() → 返回结果。

    参数:
      - file: 上传的文件（支持 .txt/.html/.pdf/.docx）
      - strategy: 分块策略 fixed/recursive
      - async_: 是否后台异步执行（大文件推荐 true）
    """
    if strategy not in ("fixed", "recursive"):
        raise HTTPException(status_code=400, detail="strategy 必须为 fixed 或 recursive")

    # 检查文件类型
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _LOADER_MAP:
        raise HTTPException(
            status_code=400,
            detail="不支持的文件类型: {}，支持: {}".format(suffix, list(_LOADER_MAP.keys())),
        )

    # 保存到 data/raw/
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_path = raw_dir / Path(file.filename or "").name

    content = await _read_upload_limited(file, _MAX_UPLOAD_SIZE)
    save_path.write_bytes(content)
    logger.info("文件已保存: %s (%d bytes)", save_path, len(content))

    # 后台异步执行
    if async_:
        from app.core.task_queue import task_manager
        task_id = task_manager.submit(
            "upload", _do_upload, save_path, strategy,
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
        result = await run_in_threadpool(_do_upload, save_path, strategy)
    except Exception as e:
        logger.error("索引构建失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="索引构建失败: {}".format(e))

    return UploadResponse(
        strategy=strategy,
        document_count=result["document_count"],
        chunk_count=result["chunk_count"],
        dimension=result["dimension"],
        index_path=result["index_path"],
        metadata_path=result["metadata_path"],
    )


def _do_upload(save_path: Path, strategy: str) -> dict:
    """上传单个文档并构建索引（通过 IndexWriter 统一写入）。

    返回 dict 形式的 UploadResponse 数据，供 task_manager 查询时返回。
    """
    import time as _time
    t = _time.time()
    logger.info(
        "_do_upload 开始: file=%s, strategy=%s", save_path.name, strategy
    )

    try:
        from app.ingestion.writer import IndexWriter

        writer = IndexWriter()
        document = writer._load_single_document(save_path)
        result = writer.write(
            documents=[document],
            strategy=strategy,
        )

        # 索引已变更，无需清空缓存：get_service 按索引版本号自动刷新 service，
        # embedding/reranker 模型继续复用
        logger.info("索引已更新（upload 完成，service 缓存将按版本号自动刷新）")

        logger.info(
            "_do_upload 完成: %.3fs, file=%s, strategy=%s, docs=%d, chunks=%d, "
            "dim=%d, mysql=%s, es=%s",
            _time.time() - t, save_path.name, strategy,
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
            "_do_upload 失败: %.3fs, file=%s, strategy=%s, error=%s: %s",
            _time.time() - t, save_path.name, strategy,
            type(e).__name__, e, exc_info=True,
        )
        raise


@router.post("/rebuild", response_model=RebuildResponse)
async def rebuild_index(req: RebuildRequest, async_: bool = True):
    """重建指定策略的索引（使用 data/raw/ 下已有文档）。

    参数:
      - strategy: 分块策略 fixed/recursive
      - async_: 是否后台异步执行，默认 True（rebuild 耗时较长）
    """
    if req.strategy not in ("fixed", "recursive"):
        raise HTTPException(status_code=400, detail="strategy 必须为 fixed 或 recursive")

    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        raise HTTPException(status_code=404, detail="data/raw/ 目录不存在")

    # 收集所有支持的文件
    files = [f for f in raw_dir.iterdir() if f.suffix.lower() in _LOADER_MAP]
    if not files:
        raise HTTPException(status_code=404, detail="data/raw/ 下无可处理的文件")

    logger.info("重建索引: strategy=%s, files=%d, async=%s", req.strategy, len(files), async_)

    # 后台异步执行
    if async_:
        from app.core.task_queue import task_manager
        task_id = task_manager.submit("rebuild", _do_rebuild, req.strategy)
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
        result = await run_in_threadpool(_do_rebuild, req.strategy)
    except Exception as e:
        logger.error("重建索引失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="重建索引失败: {}".format(e))

    return RebuildResponse(
        strategy=req.strategy,
        document_count=result["document_count"],
        chunk_count=result["chunk_count"],
        dimension=result["dimension"],
    )


def _do_rebuild(strategy: str) -> dict:
    """幂等重建索引（通过 IndexWriter.rebuild 统一清理 + 写入）。

    返回 dict 形式的 RebuildResponse 数据。
    """
    import time as _time
    t = _time.time()
    logger.info("_do_rebuild 开始: strategy=%s", strategy)

    try:
        from app.ingestion.writer import IndexWriter

        writer = IndexWriter()
        result = writer.rebuild(strategy=strategy)

        # 索引已变更，无需清空缓存：get_service 按索引版本号自动刷新 service
        logger.info("索引已更新（rebuild 完成，service 缓存将按版本号自动刷新）")

        logger.info(
            "_do_rebuild 完成: %.3fs, strategy=%s, docs=%d, chunks=%d, "
            "dim=%d, mysql_deleted=%d, es_dropped=%s, mysql=%s, es=%s",
            _time.time() - t, strategy,
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
            "_do_rebuild 失败: %.3fs, strategy=%s, error=%s: %s",
            _time.time() - t, strategy, type(e).__name__, e, exc_info=True,
        )
        raise


@router.get("/status", response_model=StatusResponse)
async def index_status():
    """查看各策略的索引状态。"""
    indexes = {}
    for strategy in ("fixed", "recursive"):
        index_dir = Path("data/index") / strategy
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


@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
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


@router.get("/tasks")
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


@router.delete("/{doc_id}", response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """删除文档：MySQL 记录 + 原始文件 + 重建受影响的索引。

    流程:
      1. 从 MySQL 删除文档记录（CASCADE 自动删 chunks）
      2. 从 data/raw/ 删除原始文件
      3. 重建 fixed/recursive 两个索引（基于剩余文件）

    若 storage.enabled=false，则仅做文件删除 + 索引重建。
    所有同步阻塞操作通过线程池执行，避免阻塞事件循环。
    """
    from starlette.concurrency import run_in_threadpool

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
        rebuilt_indexes: List[str] = []
        # 记录原始文件名（用于外层判断 doc_id 映射：MySQL 中的 doc_id 可能与 stem 相同）
        resolved_doc_id = doc_id

        # 1. 删除 MySQL 记录（按 document_id 删除 chunks + 文档）
        if config.storage_mysql_enabled:
            try:
                from app.storage import DocumentRepository, ChunkRepository
                doc_repo = DocumentRepository()
                chunk_repo = ChunkRepository()
                existing = doc_repo.get(doc_id)
                # 如果 doc_id 没命中，尝试用 file_name 回查（用户传文件名而非 document_id 时）
                if existing is None:
                    for row in doc_repo.list_all(limit=10000):
                        if row.get("file_name") == doc_id:
                            existing = row
                            resolved_doc_id = row["document_id"]
                            break
                if existing:
                    deleted_chunks = chunk_repo.delete_by_document(resolved_doc_id)
                    doc_repo.delete(resolved_doc_id)
                    deleted_from_mysql = True
                    logger.info(
                        "已从 MySQL 删除文档: doc_id=%s (原始=%s), chunks=%d",
                        resolved_doc_id, doc_id, deleted_chunks,
                    )
                else:
                    logger.warning("MySQL 中未找到文档: %s", doc_id)
            except Exception as e:
                logger.error("MySQL 删除失败: %s", e, exc_info=True)
                raise _MysqlError(str(e))

        # 2. 删除原始文件（可能是 doc_id.file_name 形式，按 stem/name 多方式匹配）
        raw_dir = Path("data/raw")
        deleted_doc_ids_for_rebuild: List[str] = []
        if raw_dir.exists():
            target = None
            if (raw_dir / doc_id).exists():
                target = raw_dir / doc_id
            else:
                for f in raw_dir.iterdir():
                    if f.stem == doc_id or f.name == doc_id:
                        target = f
                        break
                    # 若 MySQL 中查到了 resolved_doc_id，同时用 resolved 匹配
                    if deleted_from_mysql and (
                        f.stem == resolved_doc_id or f.name == resolved_doc_id
                    ):
                        target = f
                        break
            if target and target.exists():
                # 删除前记录 stem（ES/MySQL 使用 document_id = 原始 stem）
                deleted_doc_ids_for_rebuild.append(target.stem)
                # 同时把 resolved_doc_id 补上（避免二选一漏）
                if deleted_from_mysql and resolved_doc_id not in deleted_doc_ids_for_rebuild:
                    deleted_doc_ids_for_rebuild.append(resolved_doc_id)
                target.unlink()
                deleted_file = True
                logger.info("已删除原始文件: %s", target)
            else:
                # 文件没找到但 MySQL 删成功了：把 MySQL 的 doc_id 给重建流程
                if deleted_from_mysql and resolved_doc_id not in deleted_doc_ids_for_rebuild:
                    deleted_doc_ids_for_rebuild.append(resolved_doc_id)

        if not deleted_from_mysql and not deleted_file:
            raise _NotFoundError(doc_id)

        # 3. 增量重建受影响的索引（fixed + recursive）
        #    - ES 增量：只删 deleted_doc_ids 的 chunks
        #    - Milvus 清理 + FAISS + metadata：重写（保证 vector_id 顺序）
        #    - MySQL：外层已按文档级清理完，**跳过** delete_by_strategy 避免全表删插
        try:
            from app.ingestion.writer import IndexWriter
            writer = IndexWriter(config=config)
            for strategy in ("fixed", "recursive"):
                writer.incremental_rebuild_after_delete(
                    strategy=strategy,
                    deleted_doc_ids=deleted_doc_ids_for_rebuild,
                )
                rebuilt_indexes.append(strategy)
        except Exception as e:
            logger.error("增量重建索引失败（文档已删除）: %s", e, exc_info=True)

        # 索引已变更，无需清空缓存：get_service 按索引版本号自动刷新 service
        logger.info("索引已更新（删除文档后，service 缓存将按版本号自动刷新）")

        return DeleteResponse(
            document_id=doc_id,
            deleted_from_mysql=deleted_from_mysql,
            deleted_chunks=deleted_chunks,
            deleted_file=deleted_file,
            rebuilt_indexes=rebuilt_indexes,
        )

    try:
        return await run_in_threadpool(_do_delete)
    except _NotFoundError:
        raise HTTPException(status_code=404, detail="文档不存在: {}".format(doc_id))
    except _MysqlError as e:
        raise HTTPException(status_code=500, detail="MySQL 删除失败: {}".format(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """列出所有文档（优先从 MySQL 读取，否则扫描 data/raw/）。"""
    from starlette.concurrency import run_in_threadpool

    def _do_list():
        from app.core.config import Config
        config = Config()
        documents: List[DocumentItem] = []

        if config.storage_mysql_enabled:
            try:
                from app.storage import DocumentRepository, ChunkRepository
                doc_repo = DocumentRepository()
                chunk_repo = ChunkRepository()
                for row in doc_repo.list_all(limit=1000):
                    doc_id = row["document_id"]
                    documents.append(DocumentItem(
                        document_id=doc_id,
                        file_name=row.get("file_name", ""),
                        content_length=row.get("content_length", 0),
                        source=row.get("source") or "",
                        chunk_count=len(chunk_repo.get_by_document(doc_id)),
                        created_at=str(row.get("created_at", "")),
                    ))
                return DocumentListResponse(documents=documents, total=len(documents))
            except Exception as e:
                logger.error("MySQL 查询文档失败，回退到文件扫描: %s", e)

        # 回退：扫描 data/raw/
        raw_dir = Path("data/raw")
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
