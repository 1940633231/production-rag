"""认证 / RBAC 的 Pydantic 请求/响应模型。"""
from typing import List, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """登录请求。"""
    username: str = Field(..., min_length=1, max_length=128, description="用户名")
    password: str = Field(..., min_length=1, max_length=128, description="密码")


class RoleOut(BaseModel):
    """角色信息。"""
    role_code: str
    name: str
    builtin: bool = False
    permissions: List[str] = Field(default_factory=list)


class UserOut(BaseModel):
    """用户信息。"""
    user_id: str
    username: str
    display_name: str = ""
    tenant_id: str = "default"
    is_active: bool = True
    created_at: str = ""
    roles: List[str] = Field(default_factory=list)


class TokenResponse(BaseModel):
    """登录成功响应。"""
    token: str
    token_type: str = "bearer"
    expires_in: int = 0
    user: UserOut
    permissions: List[str] = Field(default_factory=list)


class CreateUserRequest(BaseModel):
    """创建用户请求。"""
    username: str = Field(..., min_length=1, max_length=128, description="用户名")
    password: str = Field(..., min_length=6, max_length=128, description="密码（至少 6 位）")
    display_name: str = Field("", max_length=128)
    tenant_id: str = Field("default", max_length=64)
    roles: List[str] = Field(default_factory=list, description="角色编码列表")


class UpdateUserRequest(BaseModel):
    """更新用户请求（全部可选）。"""
    password: Optional[str] = Field(None, min_length=6, max_length=128)
    display_name: Optional[str] = Field(None, max_length=128)
    is_active: Optional[bool] = None
    roles: Optional[List[str]] = None


class CreateRoleRequest(BaseModel):
    """创建角色请求。"""
    role_code: str = Field(..., min_length=1, max_length=64, description="角色编码")
    name: str = Field(..., min_length=1, max_length=128, description="角色名称")
    permissions: List[str] = Field(default_factory=list, description="权限点列表")
