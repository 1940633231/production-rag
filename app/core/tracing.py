"""请求追踪中间件：为每个请求生成 trace_id，记录链路日志。

特性:
  - 每个请求生成唯一 trace_id（uuid4 前 8 位）
  - 记录请求开始/结束日志（method, path, status, duration_ms, trace_id）
  - 响应头注入 X-Trace-Id，便于前端/日志关联
  - 可观测性中间件 + Prometheus 指标采集一体化

使用:
  # 在 create_app 中注册
  from app.core.tracing import TracingMiddleware
  app.add_middleware(TracingMiddleware)
"""
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logger import get_logger
from app.core.metrics import metrics

logger = get_logger(__name__)

# 不记录指标和追踪的路径（避免健康检查/指标自身产生噪音）
_EXCLUDED_PATHS = {"/metrics", "/favicon.ico"}


class TracingMiddleware(BaseHTTPMiddleware):
    """请求追踪中间件：trace_id + 链路日志 + 指标采集。"""

    async def dispatch(self, request: Request, call_next):
        # 跳过排除路径
        path = request.url.path
        if path in _EXCLUDED_PATHS:
            return await call_next(request)

        # 生成 trace_id
        trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex[:8]
        method = request.method

        # 记录开始
        start = time.time()
        logger.info(
            "[trace=%s] → %s %s",
            trace_id, method, path,
        )

        # 执行请求
        try:
            response = await call_next(request)
        except Exception as e:
            duration = time.time() - start
            logger.error(
                "[trace=%s] ✗ %s %s 异常: %.3fs, error=%s",
                trace_id, method, path, duration, e, exc_info=True,
            )
            # 记录指标
            metrics.record_http_request(method, path, 500, duration)
            raise

        # 计算耗时
        duration = time.time() - start
        status = response.status_code

        # 注入 trace_id 到响应头
        response.headers["X-Trace-Id"] = trace_id

        # 记录结束
        status_icon = "✓" if status < 400 else "✗"
        logger.info(
            "[trace=%s] %s %s %s %s %.1fms",
            trace_id, status_icon, method, path, status, duration * 1000,
        )

        # 记录指标
        metrics.record_http_request(method, path, status, duration)

        return response
