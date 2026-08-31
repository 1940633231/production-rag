"""RBAC 常量：权限点定义 + 内置角色。"""

# 超级管理员角色编码：拥有全部权限点（动态，不依赖 role_permissions 表）
SUPERADMIN_ROLE = "superadmin"

# 全部权限点（code -> 描述）。新增权限点需在此登记，并在 seed 时落库。
ALL_PERMISSIONS = {
    "chat:query": "问答（chat）",
    "knowledge:read": "查看知识库/索引状态/任务",
    "knowledge:upload": "上传文档",
    "knowledge:delete": "删除文档",
    "knowledge:rebuild": "重建索引",
    "knowledge:grant": "文档授权管理",
    "admin:users": "用户管理",
    "admin:roles": "角色管理",
    "admin:audit": "审计日志",
}

# 内置角色（code -> (名称, 权限列表)）。"*" 表示全部权限（superadmin 专用）。
BUILTIN_ROLES = {
    SUPERADMIN_ROLE: ("超级管理员", "*"),
    "admin": ("管理员", [
        "chat:query",
        "knowledge:read",
        "knowledge:upload",
        "knowledge:delete",
        "knowledge:rebuild",
        "knowledge:grant",
        "admin:users",
        "admin:roles",
        "admin:audit",
    ]),
    "editor": ("内容编辑", [
        "chat:query",
        "knowledge:read",
        "knowledge:upload",
        "knowledge:delete",
        "knowledge:rebuild",
    ]),
    "viewer": ("只读用户", [
        "chat:query",
        "knowledge:read",
    ]),
}
