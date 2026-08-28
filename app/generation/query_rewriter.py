"""Query 改写：多轮对话中指代消解，产出独立可检索的搜索查询。

设计要点：
- 仅在存在对话历史时改写（单轮查询无需改写，避免多余 LLM 调用）
- 复用现有 generator（qwen 真实改写；stub 不支持改写，直接返回原 query）
- 改写失败（LLM 异常/返回空）时回退原 query，保证检索不中断
- 改写后的 query 用于检索 + 生成，response.query 仍保留原始用户问题
"""
import time
from typing import Dict, List, Optional

from app.core.logger import get_logger

logger = get_logger(__name__)


class QueryRewriter:
    """根据对话历史改写用户查询，消解指代、补全省略上下文。"""

    SYSTEM_PROMPT = (
        "你是一个查询改写助手。根据对话历史，把用户当前的问题改写成一个"
        "独立、完整、可直接用于文档检索的搜索查询。要求：\n"
        "1. 解决指代（如\"它\"\"刚才\"\"上面\"等），补全省略的上下文\n"
        "2. 保持问题的核心意图不变，不要引入历史中不存在的信息\n"
        "3. 只输出改写后的查询，不要任何解释、引号或多余文字\n"
        "4. 如果当前问题本身已完整清晰，原样输出"
    )

    def __init__(self, generator):
        """generator: BaseGenerator 实例（stub 时自动跳过改写）。"""
        self.generator = generator

    def rewrite(self, query: str, history: Optional[List[Dict]] = None) -> str:
        """改写查询；无历史 / stub 生成器 / 改写异常时回退原 query。"""
        history = history or []
        if not history:
            return query

        # stub 生成器无真实 LLM 能力，直接返回
        from app.generation.generator import StubGenerator
        if isinstance(self.generator, StubGenerator):
            logger.info("Query 改写跳过: 当前为 StubGenerator")
            return query

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_prompt(query, history)},
        ]

        t = time.time()
        try:
            rewritten = (self.generator.generate(messages) or "").strip()
        except Exception as e:
            logger.warning(
                "Query 改写失败，回退原 query: %s: %s", type(e).__name__, e,
            )
            return query

        logger.info(
            "Query 改写完成: %.3fs, %r → %r",
            time.time() - t, query[:50], rewritten[:50],
        )
        return rewritten or query

    @staticmethod
    def _build_user_prompt(query: str, history: List[Dict]) -> str:
        """把历史对话格式化为「用户：…/助手：…」文本。"""
        lines = []
        for msg in history[-10:]:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if not content or role not in ("user", "assistant"):
                continue
            speaker = "用户" if role == "user" else "助手"
            lines.append("{}：{}".format(speaker, content))
        history_text = "\n".join(lines) if lines else "（无）"
        return "对话历史：\n{hist}\n\n当前问题：{q}\n\n改写后的查询：".format(
            hist=history_text, q=query
        )
