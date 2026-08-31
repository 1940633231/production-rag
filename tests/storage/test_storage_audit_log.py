r"""审计日志 MySQL 仓储测试（真实 DB）。

验证 AuditLogger 的写入（后台队列 + flush）/ 查询 / 计数。

前置条件: MySQL 可用（同 tests/storage/test_mysql_crud.py），不可用自动 skip。

运行:
  .venv\Scripts\python.exe -m pytest tests\storage\test_audit_log.py -v
"""
import os
import sys
import uuid
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.env import load_env
load_env()

from app.storage.mysql import MySQLManager, _MYSQL_AVAILABLE
from app.audit.logger import AuditLogger


@pytest.fixture(scope="module")
def manager():
    """创建 MySQLManager 并初始化表结构；MySQL 不可用则跳过。"""
    if not _MYSQL_AVAILABLE:
        pytest.skip("pymysql/dbutils 未安装")
    mgr = MySQLManager()
    try:
        mgr.init_schema()
    except Exception as e:
        pytest.skip("MySQL 连接失败: {}。请确认 MySQL 服务运行中且环境变量已配置。".format(e))
    return mgr


@pytest.fixture(autouse=True)
def cleanup(manager):
    """测试后清理审计测试数据。"""
    yield
    try:
        with manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM audit_logs WHERE resource LIKE '_pytest_audit_%'"
                )
    except Exception:
        pass


class TestAuditLoggerCrud:
    def test_record_flush_and_query(self, manager):
        # start_worker=False：仅由 flush() 同步落库，保证确定性
        audit = AuditLogger(manager=manager, start_worker=False)
        tag = uuid.uuid4().hex[:8]
        resource = "_pytest_audit_{}".format(tag)

        audit.record(
            tenant_id="acme", actor_user_id="u-1", actor_username="alice",
            action="document.upload", resource=resource, detail="测试上传",
        )
        audit.record(
            tenant_id="acme", actor_user_id="u-1", actor_username="alice",
            action="document.delete", resource=resource, result="success",
        )
        audit.record(
            tenant_id="beta", actor_user_id="u-2", actor_username="bob",
            action="authz.denied", resource=resource, result="denied",
        )
        # 排空队列，保证已落库
        assert audit.flush(timeout=5.0) == 3

        # 全量查询
        rows = audit.query(resource=resource)
        assert len(rows) == 3
        # 按租户过滤
        acme = audit.query(tenant_id="acme", resource=resource)
        assert len(acme) == 2
        # 按 action 过滤
        denied = audit.query(action="authz.denied", resource=resource)
        assert len(denied) == 1
        assert denied[0]["result"] == "denied"
        assert denied[0]["actor_username"] == "bob"
        # 字段完整性
        row = rows[0]
        assert row["tenant_id"] in ("acme", "beta")
        assert row["actor_user_id"]
        assert row["action"]
        assert "created_at" in row

    def test_count(self, manager):
        audit = AuditLogger(manager=manager, start_worker=False)
        tag = uuid.uuid4().hex[:8]
        resource = "_pytest_audit_cnt_{}".format(tag)

        audit.record(action="login.success", resource=resource, tenant_id="default")
        audit.record(action="login.failure", resource=resource, tenant_id="default")
        assert audit.flush(timeout=5.0) == 2

        assert audit.count(resource=resource) == 2
        assert audit.count(action="login.success", resource=resource) == 1
