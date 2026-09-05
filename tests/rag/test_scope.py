"""QueryScopeResolver 单元测试（mock 依赖，不触真实 LLM / BM25 / MySQL）。

覆盖:
  - scope 未启用 → resolve 返回 None（不调用提取/匹配）
  - 显式 scope → content 匹配 → 文档过滤集
  - auto 模式：LLM 提取实体 → 匹配
  - 提取失败 / 输出"无" → 降级为不过滤（None）
  - 匹配不到文档 → 空集（业务范围无内容）

运行:
  .venv\\Scripts\\python.exe -m pytest tests\\rag\\test_scope.py -v
"""
import pytest

from app.rag.scope import QueryScopeResolver


class _Cfg:
    """最小配置替身（scope 属性可写，规避 Config 只读 property）。"""
    scope_enabled = False
    scope_mode = "auto"
    scope_require_entity = False
    scope_match_top_k = 10


class FakeGenerator:
    def __init__(self, result):
        self.result = result

    def generate(self, messages):
        return self.result


class FakeBM25:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, top_k=10, document_ids=None):
        return self.hits


@pytest.fixture
def resolver():
    return QueryScopeResolver(config=_Cfg())


class TestResolve:
    def test_disabled_returns_none(self, resolver, monkeypatch):
        """scope 未启用：直接 None，不进入提取/匹配。"""
        resolver.config.scope_enabled = False
        called = []

        def _extract(q):
            called.append("extract")
            return "宝钢股份"

        monkeypatch.setattr(resolver, "_extract_entity", _extract)
        assert resolver.resolve("宝钢股份财报如何") is None
        assert called == []

    def test_explicit_scope_matches_docs(self, resolver, monkeypatch):
        """显式 scope + content 匹配 → 文档过滤集。"""
        resolver.config.scope_enabled = True
        resolver.config.scope_mode = "explicit"
        hits = [
            {"document_id": "doc-a", "content": "宝钢股份财报"},
            {"document_id": "doc-b", "content": "宝钢股份营收"},
            {"document_id": "doc-a", "content": "宝钢股份另一段"},
        ]
        monkeypatch.setattr(resolver, "_get_bm25", lambda s, t: FakeBM25(hits))
        doc_ids = resolver.resolve("财报如何", scope="宝钢股份")
        assert doc_ids == {"doc-a", "doc-b"}
        assert resolver.last_entity == "宝钢股份"

    def test_auto_mode_extracts_entity(self, resolver, monkeypatch):
        """auto 模式：无显式 scope → LLM 提取实体 → 匹配。"""
        resolver.config.scope_enabled = True
        resolver.config.scope_mode = "auto"
        monkeypatch.setattr(resolver, "_get_generator", lambda: FakeGenerator("宝钢股份"))
        monkeypatch.setattr(resolver, "_get_bm25", lambda s, t: FakeBM25(
            [{"document_id": "doc-a", "content": "宝钢股份"}]
        ))
        doc_ids = resolver.resolve("宝钢股份财报如何")
        assert doc_ids == {"doc-a"}
        assert resolver.last_entity == "宝钢股份"

    def test_extract_failure_falls_back_none(self, resolver, monkeypatch):
        """LLM 返回"无"（无业务主体）→ 降级为不过滤。"""
        resolver.config.scope_enabled = True
        resolver.config.scope_mode = "auto"
        monkeypatch.setattr(resolver, "_get_generator", lambda: FakeGenerator("无"))

        def _match(entity, s, t):
            raise AssertionError("不应进入匹配: entity=%s" % entity)

        monkeypatch.setattr(resolver, "_match_docs", _match)
        assert resolver.resolve("钢铁行业今年行情如何") is None
        assert resolver.last_entity is None

    def test_extract_exception_falls_back_none(self, resolver, monkeypatch):
        """LLM 异常 → 提取内部捕获 → 降级为不过滤。"""
        resolver.config.scope_enabled = True
        resolver.config.scope_mode = "auto"

        class BoomGenerator:
            def generate(self, messages):
                raise RuntimeError("llm down")

        monkeypatch.setattr(resolver, "_get_generator", lambda: BoomGenerator())
        assert resolver.resolve("宝钢股份财报") is None

    def test_no_match_returns_empty_set(self, resolver, monkeypatch):
        """实体匹配不到文档 → 空集（业务范围无内容）。"""
        resolver.config.scope_enabled = True
        resolver.config.scope_mode = "explicit"
        monkeypatch.setattr(resolver, "_get_bm25", lambda s, t: FakeBM25([]))
        assert resolver.resolve("财报", scope="不存在的公司") == set()


class TestExtractEntity:
    def test_explicit_mode_never_extracts(self, resolver, monkeypatch):
        resolver.config.scope_mode = "explicit"
        monkeypatch.setattr(resolver, "_get_generator", lambda: FakeGenerator("宝钢股份"))
        assert resolver._extract_entity("宝钢股份财报") is None

    def test_stub_generator_skips(self, resolver, monkeypatch):
        from app.generation.generator import StubGenerator

        resolver.config.scope_mode = "auto"
        monkeypatch.setattr(resolver, "_get_generator", lambda: StubGenerator())
        assert resolver._extract_entity("宝钢股份财报") is None
