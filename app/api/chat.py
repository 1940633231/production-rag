"""聊天 API：POST /api/chat 完整 RAG 问答接口。

支持两种响应模式：
  1. 普通 JSON 响应（默认）：一次性返回完整 answer + citations + stats + chunks
  2. SSE 流式响应（stream=true）：逐段推送 answer 文本，最后推送完整元数据

请求参数:
  - query (str, 必填): 用户问题
  - strategy (str): 分块策略 fixed/recursive，默认 recursive
  - mode (str): 检索模式 vector/bm25/hybrid，默认 hybrid
  - use_rerank (bool): 是否启用 rerank，默认 true
  - stream (bool): 是否流式返回，默认 false
  - history (list): 多轮对话历史 [{role: user/assistant, content}, ...]，默认空。
    客户端需自行维护：每轮把上一轮 query 作为 user、answer 作为 assistant 追加。
    服务端仅保留最近 5 轮，超出自动截断。

使用示例:
  # 普通 JSON（多轮）
  curl -X POST http://localhost:8000/api/chat \\
    -H "Content-Type: application/json" \\
    -d '{"query": "那刚才说的供需如何影响价格？",
         "history": [{"role": "user", "content": "铁矿近期供需如何？"},
                     {"role": "assistant", "content": "供给增加、需求下降。"}]}'
"""
import json
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth.dependencies import AuthUser, get_current_user, require_permission
from app.cache.query_cache import build_query_cache_key, get_query_cache
from app.core.logger import get_logger
from app.rag.service import _index_version, get_service, reset_service_cache

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


# ---- 请求/响应模型 ----

class ChatRequest(BaseModel):
    """RAG 问答请求。"""
    query: str = Field(..., description="用户问题")
    strategy: str = Field("recursive", description="分块策略: fixed/recursive")
    mode: str = Field("hybrid", description="检索模式: vector/bm25/hybrid")
    use_rerank: bool = Field(True, description="是否启用 rerank")
    stream: bool = Field(False, description="是否流式返回（SSE）")
    history: List[Dict] = Field(
        default_factory=list,
        description="多轮对话历史：[{role: user/assistant, content}, ...]，默认空",
    )


class CitationItem(BaseModel):
    number: int
    chunk_id: str
    file_name: str
    start_offset: int
    end_offset: int
    content_preview: str


class ChunkItem(BaseModel):
    chunk_id: str = ""
    score: float = 0.0
    rerank_score: Optional[float] = None
    content: str = ""
    merged: Optional[bool] = None
    compressed: Optional[bool] = None
    budget_truncated: Optional[bool] = None


class ChatResponse(BaseModel):
    """RAG 问答完整响应。"""
    query: str
    answer: str
    context: str
    chunks: List[Dict]
    citations: List[Dict]
    stats: Dict
    elapsed: float


# ---- 路由 ----

def _get_config():
    """延迟加载 Config，避免 import 时读文件。"""
    from app.core.config import Config
    return Config()


# 缓存条目数指标刷新节流（Redis 后端统计需 scan，不宜每次请求执行）
_cache_entries_last = [0.0]


def _refresh_cache_entries_metric(cache, force: bool = False) -> None:
    """刷新缓存条目数 Gauge（覆盖内存/Redis 双后端）。

    - 读取路径每 30s 节流一次（Redis 统计需 scan，不宜每次请求执行）
    - 写入路径 force=True：条目刚变化，立即刷新（写频次低，可接受）
    """
    now = time.time()
    if not force and now - _cache_entries_last[0] < 30.0:
        return
    _cache_entries_last[0] = now
    try:
        from app.core.metrics import metrics
        metrics.set_cache_entries(int(cache.stats().get("entries", 0)))
    except Exception:
        pass


async def _stream_events_from_cached(cached: Dict):
    """把权限感知缓存中的完整响应重放为 SSE 事件流（流式命中时使用）。

    事件序列与真实流式一致：meta → delta(整段答案) → citations → done。
    """
    elapsed = float(cached.get("elapsed", 0) or 0)
    yield {
        "event": "meta",
        "data": json.dumps({
            "chunks": cached.get("chunks", []),
            "stats": cached.get("stats", {}),
            "elapsed": elapsed,
        }, ensure_ascii=False),
    }
    answer = cached.get("answer") or ""
    if answer:
        yield {
            "event": "delta",
            "data": json.dumps({"content": answer}, ensure_ascii=False),
        }
    yield {
        "event": "citations",
        "data": json.dumps(cached.get("citations", []), ensure_ascii=False),
    }
    yield {
        "event": "done",
        "data": json.dumps({
            "answer_length": len(answer),
            "elapsed": elapsed,
        }, ensure_ascii=False),
    }


@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(require_permission("chat:query"))],
)
async def chat(req: ChatRequest, user: Optional[AuthUser] = Depends(get_current_user)):
    """RAG 问答接口（普通 JSON 响应）。

    流程：query → RAGService.query() → 返回 answer + citations + stats + chunks
    租户隔离：service 按当前用户 tenant_id 构建，只检索该租户的索引。
    """
    logger.info(
        "API /chat: query=%r, strategy=%s, mode=%s, rerank=%s, tenant=%s",
        req.query[:80], req.strategy, req.mode, req.use_rerank,
        user.tenant_id if user else "default",
    )

    if req.stream:
        # 流式模式走 SSE 端点
        raise HTTPException(
            status_code=400,
            detail="stream=true 请使用 /api/chat/stream 端点",
        )

    tenant_id = user.tenant_id if user else "default"
    config = _get_config()

    # ---- 文档级 ACL：计算当前用户可读文档集合（失败回退为不设过滤）----
    document_ids = None
    if user is not None:
        try:
            from app.acl.repository import ACLRepository
            document_ids = ACLRepository().get_readable_document_ids(user, tenant_id)
        except Exception as e:
            logger.warning("ACL 可读文档计算失败，回退为不设文档级过滤: %s", e)
            document_ids = None

    # ---- 权限感知查询缓存：命中直接返回，未命中再实时检索 ----
    cache_enabled = config.cache_enabled
    cache_key = None
    cache = None
    if cache_enabled:
        index_version = _index_version(req.strategy, tenant_id)
        if index_version == "unknown":
            # strict：版本权威源（DB）不可用 → 本次禁用缓存，实时检索兜底
            logger.warning(
                "索引版本未知（DB 不可用），本次禁用查询缓存: tenant=%s, query=%r",
                tenant_id, req.query[:60],
            )
        else:
            cache = get_query_cache(config)
            _refresh_cache_entries_metric(cache)
            permissions = sorted(user.permissions) if user else []
            cache_key = build_query_cache_key(
                tenant_id=tenant_id,
                user_id=user.user_id if user else None,
                permissions=permissions,
                query=req.query,
                strategy=req.strategy,
                mode=req.mode,
                use_rerank=req.use_rerank,
                index_version=index_version,
                history=req.history,
                document_ids=document_ids,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                from app.core.metrics import metrics
                metrics.record_cache_hit()
                logger.info(
                    "API /chat 缓存命中: tenant=%s, query=%r",
                    tenant_id, req.query[:60],
                )
                return ChatResponse(**cached)

    from app.core.metrics import metrics
    metrics.record_cache_miss()

    service = get_service(
        config=config,
        strategy=req.strategy,
        mode=req.mode,
        use_rerank=req.use_rerank,
        tenant_id=tenant_id,
    )

    # RAG 查询为同步阻塞（embedding + rerank + LLM），用线程池避免阻塞事件循环
    from starlette.concurrency import run_in_threadpool

    t = time.time()
    try:
        response = await run_in_threadpool(
            service.query, req.query, req.history, document_ids
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail="索引文件不存在: {}".format(e))
    except Exception as e:
        logger.error("API /chat 查询失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="查询失败: {}".format(e))

    elapsed = time.time() - t
    logger.info("API /chat 完成: %.3fs, answer_len=%d", elapsed, len(response.answer or ""))

    result = ChatResponse(
        query=response.query,
        answer=response.answer or "",
        context=response.context,
        chunks=response.chunks,
        citations=response.citations,
        stats=response.stats,
        elapsed=round(elapsed, 3),
    )

    # 写入权限感知缓存（key 已含 tenant + 权限 + 索引版本）
    if cache_enabled and cache_key is not None and cache is not None:
        cache.set(cache_key, result.model_dump())
        _refresh_cache_entries_metric(cache, force=True)
        logger.info("API /chat 写入缓存: tenant=%s, query=%r", tenant_id, req.query[:80])

    return result


@router.post(
    "/chat/stream",
    dependencies=[Depends(require_permission("chat:query"))],
)
async def chat_stream(req: ChatRequest, user: Optional[AuthUser] = Depends(get_current_user)):
    """RAG 问答接口（SSE 流式响应，真流式）。

    SSE 事件序列：
      1. event: meta      → chunks + stats（检索阶段完成）
      2. event: delta     → LLM 增量文本片段（多次，真流式）
      3. event: citations → 引用列表
      4. event: done      → 结束标记

    采用 RAGPipeline.run_stream() 真流式输出，首字延迟≈模型开始输出时间。
    租户隔离：service 按当前用户 tenant_id 构建。
    """
    from sse_starlette.sse import EventSourceResponse
    from starlette.concurrency import iterate_in_threadpool

    logger.info(
        "API /chat/stream: query=%r, strategy=%s, mode=%s, rerank=%s, tenant=%s",
        req.query[:80], req.strategy, req.mode, req.use_rerank,
        user.tenant_id if user else "default",
    )

    tenant_id = user.tenant_id if user else "default"
    config = _get_config()

    # ---- 文档级 ACL：计算当前用户可读文档集合（失败回退为不设过滤）----
    document_ids = None
    if user is not None:
        try:
            from app.acl.repository import ACLRepository
            document_ids = ACLRepository().get_readable_document_ids(user, tenant_id)
        except Exception as e:
            logger.warning("ACL 可读文档计算失败，回退为不设文档级过滤: %s", e)
            document_ids = None

    # ---- 权限感知查询缓存：流式与同步共用同一套 key（命中重放 / 未命中写入）----
    cache_enabled = config.cache_enabled
    cache_key = None
    cache = None
    if cache_enabled:
        from app.core.metrics import metrics
        index_version = _index_version(req.strategy, tenant_id)
        if index_version == "unknown":
            # strict：版本权威源（DB）不可用 → 本次禁用缓存，实时检索兜底
            logger.warning(
                "索引版本未知（DB 不可用），本次禁用查询缓存: tenant=%s, query=%r",
                tenant_id, req.query[:60],
            )
        else:
            cache = get_query_cache(config)
            _refresh_cache_entries_metric(cache)
            permissions = sorted(user.permissions) if user else []
            cache_key = build_query_cache_key(
                tenant_id=tenant_id,
                user_id=user.user_id if user else None,
                permissions=permissions,
                query=req.query,
                strategy=req.strategy,
                mode=req.mode,
                use_rerank=req.use_rerank,
                index_version=index_version,
                history=req.history,
                document_ids=document_ids,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                metrics.record_cache_hit()
                logger.info(
                    "API /chat/stream 缓存命中: tenant=%s, query=%r",
                    tenant_id, req.query[:60],
                )
                return EventSourceResponse(_stream_events_from_cached(cached))

    service = get_service(
        config=config,
        strategy=req.strategy,
        mode=req.mode,
        use_rerank=req.use_rerank,
        tenant_id=tenant_id,
    )

    async def event_generator():
        """生成 SSE 事件流，消费 service.query_stream() 的同步事件流。"""
        from app.core.metrics import metrics
        t = time.time()
        chunks: List[Dict] = []
        stats: Dict = {}
        context = ""
        answer_parts: List[str] = []
        citations: List[Dict] = []
        errored = False
        # 用线程池迭代同步生成器，避免阻塞事件循环
        async for event in iterate_in_threadpool(
            service.query_stream(req.query, req.history, document_ids)
        ):
            evt_type = event.get("type")
            if evt_type == "meta":
                chunks = event.get("chunks", [])
                stats = event.get("stats", {})
                context = event.get("context", "")
                elapsed = round(time.time() - t, 3)
                yield {
                    "event": "meta",
                    "data": json.dumps({
                        "chunks": chunks,
                        "stats": stats,
                        "elapsed": elapsed,
                    }, ensure_ascii=False),
                }
            elif evt_type == "delta":
                answer_parts.append(event.get("content", ""))
                yield {
                    "event": "delta",
                    "data": json.dumps(
                        {"content": event.get("content", "")}, ensure_ascii=False
                    ),
                }
            elif evt_type == "citations":
                citations = event.get("citations", [])
                yield {
                    "event": "citations",
                    "data": json.dumps(
                        citations, ensure_ascii=False
                    ),
                }
            elif evt_type == "done":
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "answer_length": event.get("answer_length", 0),
                        "elapsed": round(time.time() - t, 3),
                    }, ensure_ascii=False),
                }
            elif evt_type == "error":
                errored = True
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {"error": event.get("error", "")}, ensure_ascii=False
                    ),
                }
                return

        # 流式正常结束 → 写入权限感知缓存（与同步共用 key，后续同步/流式均可命中）
        if not errored and cache_enabled and cache_key is not None and cache is not None:
            try:
                cache.set(cache_key, {
                    "query": req.query,
                    "answer": "".join(answer_parts),
                    "context": context,
                    "chunks": chunks,
                    "citations": citations,
                    "stats": stats,
                    "elapsed": round(time.time() - t, 3),
                })
                _refresh_cache_entries_metric(cache, force=True)
                logger.info(
                    "API /chat/stream 写入缓存: tenant=%s, query=%r",
                    tenant_id, req.query[:60],
                )
            except Exception as e:
                logger.warning("流式写入缓存失败: %s", e)

    return EventSourceResponse(event_generator())
