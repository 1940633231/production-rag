"""FastAPI 鉴权依赖：get_current_user / require_permission。

设计:
  - auth.enabled=true 时，所有受保护接口要求合法 JWT
  - token 内嵌角色 + 权限，校验阶段不查库（无状态）
  - auth.enabled=false 时 get_current_user 返回 None，require_permission 放行（本地调试）
"""
from dataclasses import dataclass, field
from typing import Optional, Set

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# 模块级持有 Config 类引用（与测试中 monkeypatch 的 Config 隔离，
# 保证鉴权始终按 config.yaml 的真实 auth 段生效）
from app.core.config import Config
from app.core.logger import get_logger

logger = get_logger(__name__)

_bearer = HTTPBearer(auto_error=False)


@dataclass
class AuthUser:
    """当前登录用户（token 解析结果）。"""
    user_id: str
    username: str
    display_name: str = ""
    tenant_id: str = "default"
    roles: list = field(default_factory=list)
    permissions: Set[str] = field(default_factory=set)
    is_superadmin: bool = False


def _get_config() -> Config:
    return Config()


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    config: Config = Depends(_get_config),
) -> Optional[AuthUser]:
    """解析当前用户。

    - auth.enabled=false → 返回 None（调用方据此放行）
    - 无 token / token 无效 / 已过期 → 401
    """
    if not config.auth_enabled:
        return None

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未提供认证信息，请先登录获取 token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from app.auth.security import decode_access_token
    payload = decode_access_token(
        credentials.credentials, config.auth_jwt_secret, config.auth_algorithm
    )
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token 无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )

    roles = list(payload.get("roles") or [])
    perms = set(payload.get("permissions") or [])
    is_superadmin = "superadmin" in roles
    return AuthUser(
        user_id=payload.get("sub", ""),
        username=payload.get("username", ""),
        display_name=payload.get("display_name", ""),
        tenant_id=payload.get("tenant_id", "default"),
        roles=roles,
        permissions=perms,
        is_superadmin=is_superadmin,
    )


def require_permission(permission: str):
    """生成 FastAPI 依赖：要求当前用户拥有指定权限点。

    用法:
        @router.post("/upload", dependencies=[Depends(require_permission("knowledge:upload"))])
        async def upload(...): ...

    auth.enabled=false 时放行（返回 None 用户直接通过）。
    """
    def _checker(user: Optional[AuthUser] = Depends(get_current_user)) -> None:
        if user is None:
            return
        if permission not in user.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="权限不足: 需要权限点 [{}]".format(permission),
            )
    return _checker
