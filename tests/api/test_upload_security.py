r"""上传安全测试（#15）：类型校验、路径穿越清洗、大小限制。

覆盖:
  - 不支持的文件类型 → 400
  - 路径穿越文件名（../）→ 清洗后仅保留文件名，不逃逸 data/raw
  - 超过大小限制 → 413
  - 正常上传 → 200 + 响应结构

运行:
  .venv\Scripts\python.exe -m pytest tests\api\test_upload_security.py -v
"""
import uuid

import pytest

from app.core.logger import get_logger

logger = get_logger(__name__)

# mock _do_upload 的返回值（避免真实索引构建 / embedding 模型加载）
_FAKE_UPLOAD_RESULT = {
    "document_count": 1,
    "chunk_count": 2,
    "dimension": 768,
    "index_path": "data/index/recursive/faiss.index",
    "metadata_path": "data/index/recursive/metadata.json",
}


@pytest.fixture
def mock_upload(monkeypatch):
    """替换 _do_upload 为假实现，跳过索引构建。"""
    import app.api.knowledge as knowledge

    def fake_do_upload(save_path, strategy, tenant_id="default", owner_user_id=""):
        logger.info(
            "[test] fake _do_upload: %s (strategy=%s, tenant=%s)",
            save_path, strategy, tenant_id,
        )
        return dict(_FAKE_UPLOAD_RESULT)

    monkeypatch.setattr(knowledge, "_do_upload", fake_do_upload)
    return knowledge


def _upload(client, filename, content=None, strategy="recursive"):
    if content is None:
        content = "测试内容".encode("utf-8")
    return client.post(
        "/api/knowledge/upload",
        files={"file": (filename, content, "text/plain")},
        data={"strategy": strategy},
    )


class TestUploadSecurity:
    def test_rejects_unsupported_type(self, client, mock_upload):
        resp = _upload(client, "evil.exe", b"x")
        assert resp.status_code == 400
        assert "不支持的文件类型" in resp.json()["detail"]

    def test_sanitizes_path_traversal_filename(self, client, mock_upload, raw_dir):
        """../evil.txt 应被清洗为 evil.txt，不逃逸出 data/raw。"""
        fname = "_pytest_trav_{}.txt".format(uuid.uuid4().hex[:8])
        evil_name = "../../" + fname  # 路径穿越尝试
        resp = _upload(client, evil_name, "内容".encode("utf-8"))
        assert resp.status_code == 200
        # 文件应落在 data/raw 下（文件名已清洗）
        saved = raw_dir / fname
        assert saved.exists()
        assert saved.read_text(encoding="utf-8") == "内容"
        # 上级目录（项目根）不应出现该文件
        assert not (raw_dir.parent / fname).exists()
        saved.unlink()  # 清理

    def test_rejects_oversized_file(self, client, monkeypatch, raw_dir):
        """超过大小限制应返回 413（用缩小后的限制避免真实传 50MB）。"""
        import app.api.knowledge as knowledge

        monkeypatch.setattr(knowledge, "_MAX_UPLOAD_SIZE", 100)
        fname = "_pytest_big_{}.txt".format(uuid.uuid4().hex[:8])
        resp = _upload(client, fname, b"x" * 200)  # 200 > 100
        assert resp.status_code == 413
        assert "大小超过限制" in resp.json()["detail"]
        # 不应残留文件
        assert not (raw_dir / fname).exists()

    def test_upload_success_structure(self, client, mock_upload, raw_dir):
        fname = "_pytest_ok_{}.txt".format(uuid.uuid4().hex[:8])
        resp = _upload(client, fname, "正常内容".encode("utf-8"))
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_count"] == 1
        assert data["chunk_count"] == 2
        assert data["dimension"] == 768
        assert "index_path" in data and "metadata_path" in data
        (raw_dir / fname).unlink()  # 清理
