"""健康检查 API：GET /api/health。
深度检查各组件状态：
  - 索引文件存在性 + chunk 数
  - MySQL 连通性（如启用）
  - Elasticsearch 连通性（如启用）
  - Milvus 连通性（如启用）
  - DashScope API key 配置
  - embedding / reranker 模型配置
"""
import os
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel
from app.core.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["health"])


class ComponentStatus(BaseModel):
    """单个组件状态。"""
    status: str  # "ok" / "degraded" / "error" / "disabled"
    detail: str = ""


class HealthResponse(BaseModel):
    """健康检查响应。"""
    status: str  # "ok" / "degraded"
    indexes: dict
    llm_backend: str
    llm_model: str
    components: dict  # 各组件状态


def _check_mysql() -> ComponentStatus:
    """检查 MySQL 连通性。"""
    from app.core.config import Config
    config = Config()
    if not config.storage_mysql_enabled:
        return ComponentStatus(status="disabled", detail="storage.backends.mysql.enabled=false")
    try:
        from app.storage.mysql import MySQLManager, _MYSQL_AVAILABLE
        if not _MYSQL_AVAILABLE:
            return ComponentStatus(status="error", detail="pymysql/dbutils 未安装")
        mgr = MySQLManager()
        if mgr.ping():
            return ComponentStatus(status="ok", detail="mysql 连接正常")
        return ComponentStatus(status="error", detail="ping 返回 False")
    except Exception as e:
        return ComponentStatus(status="error", detail=str(e))


def _check_elasticsearch() -> ComponentStatus:
    """检查 Elasticsearch 连通性。"""
    from app.core.config import Config
    config = Config()
    if not config.storage_es_enabled:
        return ComponentStatus(status="disabled", detail="storage.backends.es.enabled=false")
    try:
        from app.storage.es_client import ESClient, _ES_AVAILABLE
        if not _ES_AVAILABLE:
            return ComponentStatus(status="error", detail="elasticsearch 客户端未安装")
        mgr = ESClient()
        if mgr.ping():
            return ComponentStatus(status="ok", detail="es ping 正常")
        return ComponentStatus(status="error", detail="es ping 返回 False")
    except Exception as e:
        return ComponentStatus(status="error", detail=str(e))


def _check_milvus() -> ComponentStatus:
    """检查 Milvus 连通性。"""
    from app.core.config import Config
    config = Config()
    if not config.storage_milvus_enabled:
        return ComponentStatus(status="disabled", detail="storage.backends.milvus.enabled=false")
    try:
        from app.vector.milvus_store import _PYMILVUS_AVAILABLE
        if not _PYMILVUS_AVAILABLE:
            return ComponentStatus(status="error", detail="pymilvus 未安装")

        milvus_uri = f"http://{config.milvus_host}:{config.milvus_port}"
        from pymilvus import MilvusClient
        try:
            try:
                client = MilvusClient(uri=milvus_uri, timeout=5)
            except TypeError:
                client = MilvusClient(uri=milvus_uri)
            ver: str = client.get_server_version()
            return ComponentStatus(status="ok", detail=f"milvus[{milvus_uri}] server_version:{ver}")
        except Exception as inner_exc:
            msg = str(inner_exc)
            return ComponentStatus(status="error", detail=f"milvus client error: {msg}")
    except Exception as outer_exc:
        safe_detail = str(outer_exc)
        return ComponentStatus(status="error", detail=safe_detail)



def _check_dashscope() -> ComponentStatus:
    """检查 DashScope API key 是否配置。"""
    from app.core.config import Config
    config = Config()
    if config.generation_backend != "qwen":
        return ComponentStatus(
            status="disabled",
            detail="generation.backend != qwen (当前: {})".format(config.generation_backend),
        )
    api_key = os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        return ComponentStatus(status="error", detail="DASHSCOPE_API_KEY 未配置")
    if not api_key.startswith("sk-"):
        return ComponentStatus(status="degraded", detail="API key 格式异常")
    return ComponentStatus(status="ok", detail="API key 已配置")


def _check_embedding_model() -> ComponentStatus:
    """检查 embedding 模型配置（不加载，仅验证配置）。"""
    from app.core.config import Config
    config = Config()
    model_name = config.embedding_model
    if not model_name:
        return ComponentStatus(status="error", detail="embedding_model 未配置")
    return ComponentStatus(status="ok", detail="模型: {}".format(model_name))


def _check_reranker() -> ComponentStatus:
    """检查 reranker 模型配置。
    rerank 段无 enabled 开关，模型是否使用由查询时 use_rerank 参数控制。
    此处仅验证模型名是否已配置。
    """
    from app.core.config import Config
    config = Config()
    model_name = config.rerank_model
    if not model_name:
        return ComponentStatus(status="error", detail="rerank.model_name 未配置")
    return ComponentStatus(status="ok", detail="模型: {}".format(model_name))


def _check_indexes() -> dict:
    """检查索引文件状态 + chunk 数。"""
    indexes = {}
    for strategy in ("fixed", "recursive"):
        index_dir = Path("data/index") / strategy
        faiss_path = index_dir / "faiss.index"
        meta_path = index_dir / "metadata.json"
        chunk_count = 0
        if meta_path.exists():
            try:
                import json
                with open(meta_path, "r", encoding="utf-8") as f:
                    chunk_count = len(json.load(f))
            except Exception:
                pass
        indexes[strategy] = {
            "faiss_exists": faiss_path.exists(),
            "metadata_exists": meta_path.exists(),
            "chunk_count": chunk_count,
        }
    return indexes


@router.get("/health", response_model=HealthResponse)
async def health():
    """深度服务健康检查：索引 + MySQL + ES + Milvus + DashScope + embedding + reranker。"""
    import asyncio

    from app.core.config import Config
    from starlette.concurrency import run_in_threadpool

    config = Config()
    # 各组件检查全部在线程池执行并并行等待。
    # MySQL/ES/Milvus 检查是同步网络 IO，若在 async 端点中直接调用，
    # 组件挂起时会阻塞事件循环，导致其他请求全部排队。
    check_fns = {
        "mysql": _check_mysql,
        "elasticsearch": _check_elasticsearch,
        "milvus": _check_milvus,
        "dashscope": _check_dashscope,
        "embedding": _check_embedding_model,
        "reranker": _check_reranker,
    }
    results = await asyncio.gather(
        *(run_in_threadpool(fn) for fn in check_fns.values())
    )
    components = dict(zip(check_fns.keys(), results))
    # 索引检查（本地文件 IO）同样放入线程池，避免任何阻塞留在事件循环
    indexes = await run_in_threadpool(_check_indexes)
    # 更新 Prometheus 指标
    from app.core.metrics import metrics
    for strategy, info in indexes.items():
        metrics.set_index_chunks(strategy, info["chunk_count"])
    # 判断整体状态
    any_index_ready = any(
        v["faiss_exists"] and v["metadata_exists"] for v in indexes.values()
    )
    has_error = any(c.status == "error" for c in components.values())
    status = "degraded" if (not any_index_ready or has_error) else "ok"
    # 将 ComponentStatus 转为 dict
    components_dict = {
        k: {"status": v.status, "detail": v.detail}
        for k, v in components.items()
    }
    return HealthResponse(
        status=status,
        indexes=indexes,
        llm_backend=config.generation_backend,
        llm_model=config.generation_model,
        components=components_dict,
    )
