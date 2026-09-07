"""token 吊销版本回归测试：权限变更后旧 token 即时失效（fail-closed）。

覆盖:
  - uv 与当前版本一致 → 放行（不抛）
  - 版本不一致（权限/角色/租户/禁号已变更）→ 401
  - 用户已删除（查无版本）→ 401
  - 旧 token 无 uv 声明 → 401（要求重登）
  - token 签发时内嵌 uv = token_version / 默认 0

运行:
  .venv\\Scripts\\python.exe -m pytest tests/api/test_token_revocation.py -v
"""
import pytest
from fastapi import HTTPException

from app.auth.dependencies import _reject_stale_token
from app.auth.security import create_access_token, decode_access_token


def _version_provider(v):
    import app.auth.revocation as rev

    def _get(user_id):
        return v
    return _get


def test_version_match_no_raise(monkeypatch):
    import app.auth.revocation as rev
    monkeypatch.setattr(rev, "get_user_token_version", _version_provider(0))
    _reject_stale_token({"uv": 0}, "u1")  # 一致，不抛异常


def test_version_mismatch_rejects(monkeypatch):
    import app.auth.revocation as rev
    monkeypatch.setattr(rev, "get_user_token_version", _version_provider(1))
    with pytest.raises(HTTPException) as ei:
        _reject_stale_token({"uv": 0}, "u1")
    assert ei.value.status_code == 401


def test_user_deleted_rejects(monkeypatch):
    import app.auth.revocation as rev
    monkeypatch.setattr(rev, "get_user_token_version", _version_provider(None))
    with pytest.raises(HTTPException) as ei:
        _reject_stale_token({"uv": 0}, "u1")
    assert ei.value.status_code == 401


def test_missing_uv_rejects(monkeypatch):
    import app.auth.revocation as rev
    monkeypatch.setattr(rev, "get_user_token_version", _version_provider(0))
    with pytest.raises(HTTPException) as ei:
        _reject_stale_token({"roles": []}, "u1")  # 旧 token 无 uv
    assert ei.value.status_code == 401


def test_create_token_embeds_uv():
    token = create_access_token(
        "u1", "u", "d", "default", ["viewer"], ["chat:query"], "secret",
        1, token_version=7,
    )
    assert decode_access_token(token, "secret")["uv"] == 7


def test_create_token_default_uv_zero():
    token = create_access_token(
        "u1", "u", "d", "default", ["viewer"], ["chat:query"], "secret", 1,
    )
    assert decode_access_token(token, "secret")["uv"] == 0