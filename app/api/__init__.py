"""FastAPI 应用工厂：创建 app 并注册所有路由 + 中间件。

启动方式:
  uvicorn app.api:create_app --factory --reload --port 8000

或直接运行:
  python -m app.api  （内置 uvicorn）

中间件（按执行顺序，后注册先执行）:
  1. TracingMiddleware: trace_id + 链路日志 + Prometheus 指标采集
  2. CORSMiddleware: 跨域
"""
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app.core.env import load_env
from app.core.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。"""
    # 加载 .env（确保 DASHSCOPE_API_KEY 等环境变量可用）
    load_env()

    app = FastAPI(
        title="Production RAG API",
        description="生产级 RAG 问答服务：文档摄入 → 检索 → 重排 → 上下文管理 → 生成 → 引用溯源",
        version="0.1.0",
    )

    # CORS（允许前端跨域调用）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 请求追踪中间件：trace_id + 链路日志 + 指标采集
    from app.core.tracing import TracingMiddleware
    app.add_middleware(TracingMiddleware)

    # 注册路由
    from app.api.chat import router as chat_router
    from app.api.health import router as health_router
    from app.api.knowledge import router as knowledge_router

    app.include_router(chat_router)
    app.include_router(health_router)
    app.include_router(knowledge_router)

    # Prometheus 指标端点
    @app.get("/metrics", tags=["monitoring"])
    async def prometheus_metrics():
        """Prometheus 指标导出端点。"""
        from app.core.metrics import metrics
        content, content_type = metrics.export()
        return Response(content=content, media_type=content_type)

    # 根路由
    @app.get("/")
    async def root():
        return {
            "service": "Production RAG API",
            "version": "0.1.0",
            "docs": "/docs",
            "endpoints": {
                "chat": "POST /api/chat",
                "chat_stream": "POST /api/chat/stream",
                "health": "GET /api/health",
                "upload": "POST /api/knowledge/upload",
                "rebuild": "POST /api/knowledge/rebuild",
                "status": "GET /api/knowledge/status",
                "documents": "GET /api/knowledge/documents",
                "task_status": "GET /api/knowledge/tasks/{task_id}",
                "tasks": "GET /api/knowledge/tasks",
                "metrics": "GET /metrics",
            },
        }

    logger.info("FastAPI 应用已创建，路由 + 中间件注册完成")
    return app


# 模块级 app 实例（uvicorn app.api:app 直接引用）
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="0.0.0.0", port=8001, reload=True)
