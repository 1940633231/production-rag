"""认证 API：登录签发 JWT / 查询当前用户。

接口:
  POST /api/auth/login    用户名密码登录 → {token, user, permissions}
  GET  /api/auth/me       当前用户信息（需携带 token）

审计: 登录成功/失败均记录审计事件（action=login.success / login.failure）。
"""
from fastapi import APIRouter, Depends, HTTPException
from starlette.requests import Request

from app.audit.logger import record
from app.auth.dependencies import AuthUser, get_current_user
from app.auth.models import LoginRequest, TokenResponse, UserOut
from app.auth.security import create_access_token, verify_password
from app.core.config import Config
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _get_config() -> Config:
    return Config()


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, config: Config = Depends(_get_config),
                request: Request = None):
    """登录：校验用户名密码，签发 JWT（token 内嵌角色 + 权限）。"""
    from starlette.concurrency import run_in_threadpool

    ip = request.client.host if request and request.client else ""

    def _do_login():
        from app.auth.rbac_repository import RBACRepository
        repo = RBACRepository()
        user = repo.get_user_by_username(req.username)
        if user is None or not user.get("is_active"):
            raise HTTPException(
                status_code=401, detail="用户名或密码错误"
            )
        if not verify_password(req.password, user["password_hash"]):
            raise HTTPException(
                status_code=401, detail="用户名或密码错误"
            )
        user_id = user["user_id"]
        roles = repo.get_user_roles(user_id)
        permissions = sorted(repo.get_user_permissions(user_id))
        return user, roles, permissions

    try:
        user, roles, permissions = await run_in_threadpool(_do_login)
    except HTTPException as e:
        # 记录登录失败审计（未知用户/密码错误/停用账号）
        record(
            action="login.failure", result="failure",
            tenant_id="default", resource=req.username, ip=ip,
            detail=str(e.detail),
        )
        raise
    except Exception as e:
        logger.error("登录失败: %s", e, exc_info=True)
        record(
            action="login.failure", result="failure",
            tenant_id="default", resource=req.username, ip=ip,
            detail="服务异常",
        )
        raise HTTPException(status_code=500, detail="登录失败: {}".format(e))

    # 记录登录成功审计
    record(
        action="login.success", result="success",
        tenant_id=user.get("tenant_id", "default"),
        actor_user_id=user["user_id"],
        actor_username=user["username"],
        resource=req.username, ip=ip,
    )

    token = create_access_token(
        user_id=user["user_id"],
        username=user["username"],
        display_name=user.get("display_name", ""),
        tenant_id=user.get("tenant_id", "default"),
        roles=roles,
        permissions=permissions,
        secret=config.auth_jwt_secret,
        expires_hours=config.auth_token_expire_hours,
        algorithm=config.auth_algorithm,
    )
    return TokenResponse(
        token=token,
        expires_in=int(config.auth_token_expire_hours * 3600),
        user=UserOut(
            user_id=user["user_id"],
            username=user["username"],
            display_name=user.get("display_name", ""),
            tenant_id=user.get("tenant_id", "default"),
            is_active=bool(user.get("is_active", True)),
            created_at=str(user.get("created_at", "")),
            roles=roles,
        ),
        permissions=permissions,
    )


@router.get("/me")
async def me(user: AuthUser = Depends(get_current_user)):
    """返回当前用户信息（角色 + 权限）。"""
    if user is None:
        raise HTTPException(
            status_code=401, detail="鉴权未开启或未登录"
        )
    return {
        "user_id": user.user_id,
        "username": user.username,
        "display_name": user.display_name,
        "tenant_id": user.tenant_id,
        "roles": user.roles,
        "permissions": sorted(user.permissions),
        "is_superadmin": user.is_superadmin,
    }
