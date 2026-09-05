"""Multi-Query 多路召回：LLM 把查询扩展成多个不同角度的子查询。

设计要点：
- 复用现有 generator（qwen 真实扩展；stub 直接回退单路 [query]）
- 每路子查询独立检索后合并去重，弥补单查询召回不足
- LLM 失败/解析为空时回退 [query]，保证检索不中断
- 总路数含原始 query：multi_query=3 表示原始 + 2 个 LLM 生成
- 按需启用（auto=true）：先由 LLM 判断该 query 是否需要多路召回，
  需要才扩展（多实体/对比/泛问等），简单单实体问题直接单路——
  省 LLM 调用、减少无关噪音路；判定失败/不确定回退单路
"""
import re
import time
from typing import List

from app.core.logger import get_logger

logger = get_logger(__name__)


class MultiQueryExpander:
    """把查询扩展成多个不同角度的子查询（多路召回用）。"""

    SYSTEM_PROMPT = (
        "你是一个查询扩展助手。根据用户的问题，从不同角度生成多个"
        "独立的搜索查询，以提升检索召回率。要求：\n"
        "1. 每个查询聚焦问题的不同方面（实体、时间、对比、因果等）\n"
        "2. 每个查询独立完整、可直接检索，避免指代\n"
        "3. 每行一个查询，不要编号、不要解释、不要引号\n"
        "4. 查询数保持在 {num} 个以内"
    )

    JUDGE_PROMPT = (
        "你是一个检索策略助手。判断用户问题是否需要\"多路召回\""
        "（把问题拆成多个角度子查询分别检索、合并结果，提升召回率）。\n"
        "需要多路：问题涉及多个实体/多个方面/对比/泛泛之问/信息分散/需多角度。\n"
        "不需要多路：单实体、事实明确、简单直接的问题，单路检索即可覆盖。\n"
        "只回答\"需要\"或\"不需要\"，不要任何解释。"
    )

    # 行首编号/破折号/项目符号清理
    _CLEAN_RE = re.compile(r'^[\s\d\-*•.、]+')
    _NEED_RE = re.compile(r"需要")
    _NO_NEED_RE = re.compile(r"不需要")

    def __init__(self, generator, num_queries: int = 3, auto: bool = False):
        self.generator = generator
        # num_queries <= 1 表示关闭多路召回（由 expand 检查回退单路）
        self.num_queries = num_queries
        # auto=true：按需启用——LLM 先判断是否需要多路，需要才扩展
        self.auto = auto

    def expand(self, query: str) -> List[str]:
        """返回多路子查询（含原始 query），失败/不需要时回退 [query]。"""
        if self.num_queries <= 1:
            return [query]

        # stub 生成器无真实 LLM 能力，直接回退单路
        from app.generation.generator import StubGenerator
        if isinstance(self.generator, StubGenerator):
            logger.info("Multi-Query 扩展跳过: 当前为 StubGenerator")
            return [query]

        # 按需启用：LLM 判定该 query 是否需要多路召回
        if self.auto and not self._judge(query):
            logger.info("Multi-Query 按需判定: 该 query 不需要多路召回，走单路")
            return [query]

        gen_num = self.num_queries - 1  # 原始 query 占一路
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(num=gen_num)},
            {"role": "user", "content": "用户问题：{q}\n\n生成的查询：".format(q=query)},
        ]

        t = time.time()
        try:
            raw = self.generator.generate(messages) or ""
        except Exception as e:
            logger.warning(
                "Multi-Query 扩展失败，回退单路: %s: %s", type(e).__name__, e,
            )
            return [query]

        generated = self._parse(raw)
        queries = self._dedup([query] + generated, self.num_queries)
        logger.info(
            "Multi-Query 扩展完成: %.3fs, %r → %d 路",
            time.time() - t, query[:50], len(queries),
        )
        return queries

    def _judge(self, query: str) -> bool:
        """按需启用：LLM 判断该 query 是否需要多路召回。

        返回 False 表示走单路（不需要 / 判定失败 / 输出不明确）。
        注意先匹配"不需要"再匹配"需要"（中文"不需要"含"需要"子串）。
        """
        messages = [
            {"role": "system", "content": self.JUDGE_PROMPT},
            {"role": "user", "content": "问题：{}\n\n是否需要多路召回：".format(query)},
        ]
        try:
            raw = (self.generator.generate(messages) or "").strip()
        except Exception as e:
            logger.warning("Multi-Query 按需判定失败，回退单路: %s", e)
            return False
        if self._NO_NEED_RE.search(raw):
            return False
        if self._NEED_RE.search(raw):
            return True
        return False  # 未明确"需要"→ 保守单路（省成本）

    @classmethod
    def _parse(cls, raw: str) -> List[str]:
        """解析 LLM 输出为子查询列表：按行拆分 + 清理编号/空白/引号。"""
        queries = []
        for line in raw.splitlines():
            q = cls._CLEAN_RE.sub("", line).strip().strip('"\'“”')
            if q and q not in queries:
                queries.append(q)
        return queries

    @staticmethod
    def _dedup(queries: List[str], limit: int) -> List[str]:
        """去重并截断到 limit 个；全空时回退 [原始 query]。"""
        seen, out = set(), []
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                out.append(q)
            if len(out) >= limit:
                break
        return out or queries[:1]
