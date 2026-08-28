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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.logger import get_logger
from app.rag.service import get_service, reset_service_cache

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


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """RAG 问答接口（普通 JSON 响应）。

    流程：query → RAGService.query() → 返回 answer + citations + stats + chunks
    """
    logger.info(
        "API /chat: query=%r, strategy=%s, mode=%s, rerank=%s",
        req.query[:80], req.strategy, req.mode, req.use_rerank,
    )

    if req.stream:
        # 流式模式走 SSE 端点
        raise HTTPException(
            status_code=400,
            detail="stream=true 请使用 /api/chat/stream 端点",
        )

    config = _get_config()
    service = get_service(
        config=config,
        strategy=req.strategy,
        mode=req.mode,
        use_rerank=req.use_rerank,
    )

    # RAG 查询为同步阻塞（embedding + rerank + LLM），用线程池避免阻塞事件循环
    from starlette.concurrency import run_in_threadpool

    t = time.time()
    try:
        response = await run_in_threadpool(service.query, req.query, req.history)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail="索引文件不存在: {}".format(e))
    except Exception as e:
        logger.error("API /chat 查询失败: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="查询失败: {}".format(e))

    elapsed = time.time() - t
    logger.info("API /chat 完成: %.3fs, answer_len=%d", elapsed, len(response.answer or ""))

    return ChatResponse(
        query=response.query,
        answer=response.answer or "",
        context=response.context,
        chunks=response.chunks,
        citations=response.citations,
        stats=response.stats,
        elapsed=round(elapsed, 3),
    )


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """RAG 问答接口（SSE 流式响应，真流式）。

    SSE 事件序列：
      1. event: meta      → chunks + stats（检索阶段完成）
      2. event: delta     → LLM 增量文本片段（多次，真流式）
      3. event: citations → 引用列表
      4. event: done      → 结束标记

    采用 RAGPipeline.run_stream() 真流式输出，首字延迟≈模型开始输出时间。
    """
    from sse_starlette.sse import EventSourceResponse
    from starlette.concurrency import iterate_in_threadpool

    logger.info(
        "API /chat/stream: query=%r, strategy=%s, mode=%s, rerank=%s",
        req.query[:80], req.strategy, req.mode, req.use_rerank,
    )

    config = _get_config()
    service = get_service(
        config=config,
        strategy=req.strategy,
        mode=req.mode,
        use_rerank=req.use_rerank,
    )

    async def event_generator():
        """生成 SSE 事件流，消费 service.query_stream() 的同步事件流。"""
        t = time.time()
        # 用线程池迭代同步生成器，避免阻塞事件循环
        async for event in iterate_in_threadpool(
            service.query_stream(req.query, req.history)
        ):
            evt_type = event.get("type")
            if evt_type == "meta":
                elapsed = round(time.time() - t, 3)
                yield {
                    "event": "meta",
                    "data": json.dumps({
                        "chunks": event.get("chunks", []),
                        "stats": event.get("stats", {}),
                        "elapsed": elapsed,
                    }, ensure_ascii=False),
                }
            elif evt_type == "delta":
                yield {
                    "event": "delta",
                    "data": json.dumps(
                        {"content": event.get("content", "")}, ensure_ascii=False
                    ),
                }
            elif evt_type == "citations":
                yield {
                    "event": "citations",
                    "data": json.dumps(
                        event.get("citations", []), ensure_ascii=False
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
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {"error": event.get("error", "")}, ensure_ascii=False
                    ),
                }
                return

    return EventSourceResponse(event_generator())
