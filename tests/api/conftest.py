r"""API 层测试共享 fixture。

提供:
  - client: FastAPI TestClient（session 级共享）

注意:
  - 所有测试避免真实依赖（embedding 模型下载 / MySQL / ES / Milvus）
  - 测试产生的 data/raw 文件使用 `_pytest_` 前缀，由各测试自行清理
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient（session 级，避免重复启动开销）。"""
    from app.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def raw_dir():
    """data/raw 目录（相对项目根）。"""
    from pathlib import Path
    d = Path("data/raw")
    d.mkdir(parents=True, exist_ok=True)
    return d
