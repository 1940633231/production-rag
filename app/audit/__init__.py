"""审计日志模块（Audit Log）。

安全相关事件的持久化：登录、越权拒绝（401/403）、用户/角色管理、文档操作。

模块组成:
  - logger.py: AuditLogger（MySQL 后台异步写入 + 查询）+ 模块级 record() 入口
"""
