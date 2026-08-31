"""审计中间件：自动记录越权/鉴权失败事件（401 / 403）。

覆盖 require_permission 抛出的 403、get_current_user 抛出的 401，
以及任何返回 401/403 的请求。显式场景（登录失败、业务操作）由各
处理器自行 record，避免与本中间件重复。

被排除的路径：
  - /metrics、/favicon.ico（可观测噪音）
  - /api/health（健康检查，公共）
  - /api/auth/login（登录成功/失败由登录处理器显式记录，避免重复）
  - /admin、/static（管理台静态资源）
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.logger import get_logger

logger = get_logger(__name__)

_EXCLUDED_PATHS = {
    "/metrics",
    "/favicon.ico",
    "/api/health",
    "/api/auth/login",
    "/admin",
    "/static",
}


class AuditMiddleware(BaseHTTPMiddleware):
    """请求级审计中间件：捕获 401/403 响应并记录 authz.denied 事件。"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        status = response.status_code
        if status in (401, 403) and request.url.path not in _EXCLUDED_PATHS:
            self._record_denied(request, status)
        return response

    @staticmethod
    def _record_denied(request: Request, status: int) -> None:
        """尽力提取 actor 并记录一条 authz.denied 事件。"""
        from app.audit.logger import record

        tenant_id = "default"
        actor_user_id = ""
        actor_username = ""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            try:
                from app.auth.security import decode_access_token
                from app.core.config import Config
                config = Config()
                payload = decode_access_token(
                    auth_header[7:], config.auth_jwt_secret, config.auth_algorithm
                )
                if payload:
                    tenant_id = payload.get("tenant_id", "default")
                    actor_user_id = payload.get("sub", "")
                    actor_username = payload.get("username", "")
            except Exception as e:
                logger.debug("审计中间件解析 token 失败: %s", e)

        ip = request.client.host if request.client else ""
        record(
            action="authz.denied",
            result="denied",
            tenant_id=tenant_id,
            actor_user_id=actor_user_id,
            actor_username=actor_username,
            resource=request.url.path,
            ip=ip,
            detail="HTTP {} 拒绝访问: {} {}".format(
                status, request.method, request.url.path
            ),
        )
