"""生产安全护栏：APP_ENV=production 时拒绝默认 JWT 密钥 / 默认管理员密码（fail-fast）。"""
import pytest

from app.core.config import Config


@pytest.fixture
def clear_env(monkeypatch):
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("JWT_SECRET", raising=False)


def _prod_seed_pw_bypass(cfg):
    """把 seed_password 换成非默认强口令（避免默认值干扰预期结果）。"""
    cfg.data.setdefault("auth", {})["seed_password"] = "S3curePassw0rd-x"
    return cfg


class TestProductionSecurityGuard:
    def test_dev_mode_allows_defaults(self, clear_env, monkeypatch):
        monkeypatch.setenv("APP_ENV", "dev")
        Config().validate_production_security()   # 不应抛错

    def test_production_with_defaults_raises(self, clear_env, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        with pytest.raises(RuntimeError, match="安全校验失败"):
            Config().validate_production_security()

    def test_production_default_password_reported(self, clear_env, monkeypatch):
        """生产 + 默认 admin123 即便密钥已改也仍被拦截。"""
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("JWT_SECRET", "k" * 48)
        with pytest.raises(RuntimeError, match="admin123"):
            Config().validate_production_security()

    def test_production_weak_secret_raises(self, clear_env, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("JWT_SECRET", "tooshort")
        cfg = _prod_seed_pw_bypass(Config())
        with pytest.raises(RuntimeError, match="过短"):
            cfg.validate_production_security()

    def test_production_strong_override_passes(self, clear_env, monkeypatch):
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("JWT_SECRET", "r" * 48)
        cfg = _prod_seed_pw_bypass(Config())
        cfg.validate_production_security()   # 不应抛错

    def test_auth_disabled_skips_secret_check(self, clear_env, monkeypatch):
        """鉴权关闭时本就无凭据风险，不拦截（避免误伤仅本地方案）。"""
        monkeypatch.setenv("APP_ENV", "production")
        cfg = _prod_seed_pw_bypass(Config())
        cfg.data["auth"]["enabled"] = False
        cfg.validate_production_security()   # 不应抛错

    def test_run_mode_normalization(self, clear_env, monkeypatch):
        monkeypatch.setenv("APP_ENV", "PROD  ")
        assert Config().is_production is True
        monkeypatch.setenv("APP_ENV", "production")
        assert Config().is_production is True
        monkeypatch.setenv("APP_ENV", "staging")
        assert Config().is_production is False