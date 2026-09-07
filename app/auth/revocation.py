"""token 吊销版本读取层：供鉴权依赖读取用户当前的 token_version。

**为什么独立成模块**：鉴权依赖按 `user_id` 读取吊销版本需要一个可注入的接缝——
单测/测试桩无 MySQL 时可通过 monkeypatch 本模块的 `get_user_token_version`
返回固定值，避免鉴权依赖直接耦合 RBAC 仓储与真实数据库。

语义：登录时 token 内嵌 `uv`（users.token_version）；权限/角色/租户/禁号等
变更会递增 token_version，鉴权时若 `uv` != 当前版本即拒绝（401），
使旧 token 即时失效。
"""
from typing import Optional

from app.auth.rbac_repository import RBACRepository


def get_user_token_version(user_id: str) -> Optional[int]:
    """返回用户当前吊销版本；用户不存在（含已删除）返回 None。"""
    return RBACRepository().get_user_token_version(user_id)