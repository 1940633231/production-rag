"""索引版本仓储单元测试：get / bump / init_from_disk（mock manager，不依赖真实 MySQL）。

覆盖:
  - get_version: 有记录 → 版本号；无记录 / DB 异常 → None（strict：unknown）
  - bump: 首次写入 version=1，再次 ON DUPLICATE 递增（+1）
  - bump: DB 异常 → 抛异常（strict：索引变更登记失败即整体失败）
  - init_from_disk: 扫描 data/index 存量索引，INSERT IGNORE 初始版本 0

运行:
  .venv\\Scripts\\python.exe -m pytest tests\\storage\\test_index_version.py -v
"""
from pathlib import Path

import pytest

from app.storage.index_version_repository import IndexVersionRepository


class _FakeCursor:
    """记录 execute 调用并回放 fetchone 结果的假 cursor。"""

    def __init__(self):
        self.calls = []
        self.result = None

    def execute(self, sql, params=None):
        self.calls.append(("execute", sql, params))

    def executemany(self, sql, params):
        self.calls.append(("executemany", sql, params))

    def fetchone(self):
        return self.result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeManager:
    """假 MySQLManager：返回可注入 cursor 的连接。"""

    def __init__(self, cursor=None):
        self.cursor = cursor or _FakeCursor()
        self._raise_on_connect = False

    def get_connection(self):
        if self._raise_on_connect:
            raise RuntimeError("mysql down")
        return _FakeConn(self.cursor)


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def manager():
    return _FakeManager()


def _repo(manager):
    return IndexVersionRepository(manager=manager)


class TestGetVersion:
    def test_returns_version_when_record_exists(self, manager):
        manager.cursor.result = {"version": 3}
        assert _repo(manager).get_version("acme", "recursive") == 3

    def test_returns_none_when_no_record(self, manager):
        manager.cursor.result = None
        assert _repo(manager).get_version("acme", "recursive") is None

    def test_returns_none_on_db_error(self, manager):
        """strict：DB 异常 → None（调用方标记 unknown，禁用缓存）。"""
        manager._raise_on_connect = True
        assert _repo(manager).get_version("acme", "recursive") is None


class TestBump:
    def test_bump_uses_insert_on_duplicate_update(self, manager):
        """首次写入 version=1，已存在则 +1（同一 SQL，ON DUPLICATE KEY）。"""
        _repo(manager).bump("acme", "recursive")
        sql, params = manager.cursor.calls[0][1], manager.cursor.calls[0][2]
        assert "INSERT INTO index_versions" in sql
        assert "ON DUPLICATE KEY UPDATE version = version + 1" in sql
        assert params == ("acme", "recursive")

    def test_bump_raises_on_db_error(self, manager):
        """strict：登记失败抛异常（本次索引变更整体失败）。"""
        manager._raise_on_connect = True
        with pytest.raises(RuntimeError):
            _repo(manager).bump("acme", "recursive")


class TestInitFromDisk:
    def _mk_index(self, root, tenant, strategy):
        d = root / tenant / strategy if tenant != "default" else root / strategy
        d.mkdir(parents=True, exist_ok=True)
        (d / "metadata.json").write_text("[]", encoding="utf-8")

    def test_scans_default_and_tenant_indexes(self, tmp_path, manager):
        self._mk_index(tmp_path, "default", "recursive")
        self._mk_index(tmp_path, "default", "fixed")
        self._mk_index(tmp_path, "acme", "recursive")

        n = _repo(manager).init_from_disk(tmp_path)

        assert n == 3
        exec_calls = [c for c in manager.cursor.calls if c[0] == "executemany"]
        assert len(exec_calls) == 1
        params = exec_calls[0][2]
        assert ("default", "recursive") in params
        assert ("default", "fixed") in params
        assert ("acme", "recursive") in params

    def test_empty_root_returns_zero(self, tmp_path, manager):
        assert _repo(manager).init_from_disk(tmp_path) == 0

    def test_missing_root_returns_zero(self, tmp_path, manager):
        assert _repo(manager).init_from_disk(tmp_path / "nope") == 0
