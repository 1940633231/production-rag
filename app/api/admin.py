"""用户 / 角色管理 API（RBAC 管理面）。

接口:
  GET    /api/admin/permissions            列出权限点（admin:roles）
  GET    /api/admin/roles                  列出角色（admin:roles）
  POST   /api/admin/roles                  创建角色（admin:roles）
  DELETE /api/admin/roles/{role_code}      删除角色（admin:roles）
  GET    /api/admin/users                  列出用户（admin:users）
  POST   /api/admin/users                  创建用户（admin:users）
  PATCH  /api/admin/users/{user_id}        更新用户（admin:users）
  DELETE /api/admin/users/{user_id}        删除用户（admin:users）

所有 DB 操作在线程池执行，避免阻塞事件循环。
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.audit.logger import record
from app.auth.dependencies import AuthUser, get_current_user, require_permission
from app.auth.models import (
    CreateRoleRequest,
    CreateUserRequest,
    RoleOut,
    UpdateUserRequest,
    UserOut,
)
from app.auth.rbac import ALL_PERMISSIONS, SUPERADMIN_ROLE
from app.auth.security import hash_password
from app.core.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

_require_users = Depends(require_permission("admin:users"))
_require_roles = Depends(require_permission("admin:roles"))
_require_audit = Depends(require_permission("admin:audit"))
_require_metrics = Depends(require_permission("metrics:read"))


def _repo():
    from app.auth.rbac_repository import RBACRepository
    return RBACRepository()


# ---------------- 监控指标 ----------------

@router.get("/metrics", dependencies=[_require_metrics])
async def get_metrics_snapshot():
    """返回当前指标结构化快照（后台监控页使用，需 metrics:read 权限）。"""
    from app.core.metrics import metrics
    return metrics.snapshot()


# ---------------- 权限点 ----------------

@router.get("/permissions", dependencies=[_require_roles])
async def list_permissions():
    """列出全部权限点。"""
    return {
        "permissions": [
            {"code": k, "name": v} for k, v in ALL_PERMISSIONS.items()
        ]
    }


# ---------------- 角色 ----------------

@router.get("/roles", response_model=None, dependencies=[_require_roles])
async def list_roles():
    """列出全部角色（含权限）。"""
    def _do():
        roles = _repo().list_roles()
        return [RoleOut(**r).model_dump() for r in roles]
    roles = await run_in_threadpool(_do)
    return {"roles": roles, "total": len(roles)}


@router.post("/roles", dependencies=[_require_roles])
async def create_role(req: CreateRoleRequest,
                      user: AuthUser = Depends(get_current_user)):
    """创建角色并绑定权限（幂等）。"""
    def _do():
        repo = _repo()
        if repo.get_role_by_code(req.role_code) is not None:
            raise HTTPException(status_code=409, detail="角色已存在: {}".format(req.role_code))
        invalid = [p for p in req.permissions if p not in ALL_PERMISSIONS]
        if invalid:
            raise HTTPException(status_code=400, detail="未知权限点: {}".format(invalid))
        repo.create_role(req.role_code, req.name, req.permissions)
        return req
    try:
        result = await run_in_threadpool(_do)
    except HTTPException:
        raise
    record(
        action="role.create", tenant_id=user.tenant_id if user else "default",
        actor_user_id=user.user_id if user else "",
        actor_username=user.username if user else "",
        resource=req.role_code,
        detail="角色名称: {}".format(req.name),
    )
    return {"role_code": result.role_code, "name": result.name,
            "permissions": result.permissions}


@router.delete("/roles/{role_code}", dependencies=[_require_roles])
async def delete_role(role_code: str,
                      user: AuthUser = Depends(get_current_user)):
    """删除角色。"""
    def _do():
        repo = _repo()
        role = repo.get_role_by_code(role_code)
        if role is None:
            raise HTTPException(status_code=404, detail="角色不存在: {}".format(role_code))
        if role.get("builtin"):
            raise HTTPException(status_code=400, detail="内置角色不可删除: {}".format(role_code))
        repo.delete_role(role_code)
    try:
        await run_in_threadpool(_do)
    except HTTPException:
        raise
    record(
        action="role.delete", tenant_id=user.tenant_id if user else "default",
        actor_user_id=user.user_id if user else "",
        actor_username=user.username if user else "",
        resource=role_code,
    )
    return {"deleted": role_code}


# ---------------- 用户 ----------------

@router.get("/users", dependencies=[_require_users])
async def list_users():
    """列出全部用户（含角色）。"""
    def _do():
        users = _repo().list_users()
        return [UserOut(
            user_id=u["user_id"],
            username=u["username"],
            display_name=u.get("display_name", ""),
            tenant_id=u.get("tenant_id", "default"),
            is_active=bool(u.get("is_active", True)),
            created_at=str(u.get("created_at", "")),
            roles=u.get("roles", []),
        ).model_dump() for u in users]
    users = await run_in_threadpool(_do)
    return {"users": users, "total": len(users)}


@router.post("/users", dependencies=[_require_users])
async def create_user(req: CreateUserRequest,
                      user: AuthUser = Depends(get_current_user)):
    """创建用户并分配角色。"""
    def _do():
        repo = _repo()
        if repo.get_user_by_username(req.username) is not None:
            raise HTTPException(status_code=409, detail="用户名已存在: {}".format(req.username))
        user_id = "u-" + uuid.uuid4().hex[:12]
        repo.create_user(
            user_id=user_id,
            username=req.username,
            password_hash=hash_password(req.password),
            display_name=req.display_name,
            tenant_id=req.tenant_id,
        )
        if req.roles:
            repo.set_user_roles(user_id, req.roles)
        return user_id
    try:
        user_id = await run_in_threadpool(_do)
    except HTTPException:
        raise
    record(
        action="user.create", tenant_id=req.tenant_id,
        actor_user_id=user.user_id if user else "",
        actor_username=user.username if user else "",
        resource=user_id,
        detail="用户名: {}, 角色: {}".format(req.username, req.roles or []),
    )
    return {"user_id": user_id, "username": req.username, "roles": req.roles}


@router.patch("/users/{user_id}", dependencies=[_require_users])
async def update_user(user_id: str, req: UpdateUserRequest,
                      user: AuthUser = Depends(get_current_user)):
    """更新用户：改密 / 改名 / 启停 / 换角色。"""
    def _do():
        repo = _repo()
        existing = repo.get_user_by_id(user_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="用户不存在: {}".format(user_id))
        if req.password is not None:
            repo.update_user(user_id, password_hash=hash_password(req.password))
        if req.display_name is not None:
            repo.update_user(user_id, display_name=req.display_name)
        if req.is_active is not None:
            repo.update_user(user_id, is_active=req.is_active)
        if req.roles is not None:
            repo.set_user_roles(user_id, req.roles)
        return existing
    try:
        existing = await run_in_threadpool(_do)
    except HTTPException:
        raise
    record(
        action="user.update",
        tenant_id=existing.get("tenant_id", "default"),
        actor_user_id=user.user_id if user else "",
        actor_username=user.username if user else "",
        resource=user_id,
        detail="用户名: {}".format(existing.get("username", "")),
    )
    return {"updated": user_id}


@router.delete("/users/{user_id}", dependencies=[_require_users])
async def delete_user(user_id: str,
                      user: AuthUser = Depends(get_current_user)):
    """删除用户。"""
    def _do():
        repo = _repo()
        existing = repo.get_user_by_id(user_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="用户不存在: {}".format(user_id))
        repo.delete_user(user_id)
        return existing
    try:
        existing = await run_in_threadpool(_do)
    except HTTPException:
        raise
    record(
        action="user.delete",
        tenant_id=existing.get("tenant_id", "default"),
        actor_user_id=user.user_id if user else "",
        actor_username=user.username if user else "",
        resource=user_id,
        detail="用户名: {}".format(existing.get("username", "")),
    )
    return {"deleted": user_id}


# ---------------- 审计日志 ----------------

@router.get("/audit-logs", dependencies=[_require_audit])
async def list_audit_logs(
    tenant_id: str = None,
    actor: str = None,
    action: str = None,
    result: str = None,
    limit: int = 50,
    offset: int = 0,
    user: AuthUser = Depends(get_current_user),
):
    """查询审计日志（需 admin:audit 权限）。

    非 superadmin 仅能查看本租户日志；superadmin 可用 tenant_id 查看任意租户。
    参数: tenant_id / actor(用户名或 user_id) / action / result / limit / offset
    """
    from app.audit.logger import get_audit_logger

    def _do():
        audit = get_audit_logger()
        scope_tenant = None
        if user is not None and not user.is_superadmin:
            scope_tenant = user.tenant_id
        elif tenant_id:
            scope_tenant = tenant_id
        return audit.query(
            tenant_id=scope_tenant,
            actor_user_id=actor or None,
            action=action or None,
            result=result or None,
            limit=min(int(limit), 500),
            offset=int(offset),
        )

    logs = await run_in_threadpool(_do)
    return {"logs": logs, "total": len(logs)}
