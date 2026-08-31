"""认证 / RBAC：JWT 鉴权 + 用户/角色/权限管理。

模块组成:
  - security.py         密码哈希 + JWT 签发/校验
  - rbac.py            权限点定义 + 内置角色
  - rbac_repository.py 用户/角色/权限 MySQL 仓储
  - dependencies.py    FastAPI 鉴权依赖（get_current_user / require_permission）
  - models.py          Pydantic 模型
  - router.py          登录 / 当前用户 API
"""
