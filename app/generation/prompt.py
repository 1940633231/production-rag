"""Prompt 构建：组装 system + user 消息，送入 LLM。

设计要点：
- system prompt 约束 LLM 基于上下文回答，不编造，并引用 [1][2] 编号
- user prompt 注入 context（ContextManager 已带 [1][2] 编号）+ query
- 返回 OpenAI/DashScope 兼容的 messages 格式：[{role, content}, ...]
"""
from typing import Dict, List


class PromptBuilder:
    """构建 LLM messages。"""

    SYSTEM_PROMPT = (
        "你是一个严谨的中文问答助手。请基于提供的上下文回答用户问题。"
        "要求：\n"
        "1. 仅使用上下文中的信息，不要编造或引用外部知识\n"
        "2. 如果上下文没有相关信息，明确说明\"根据现有信息无法回答\"\n"
        "3. 回答时引用上下文中的 [1] [2] 等编号标注来源\n"
        "4. 回答简洁、结构清晰"
    )

    def build(self, query: str, context: str) -> List[Dict]:
        """构建 messages 列表（system + user）。

        参数:
            query: 用户查询
            context: ContextManager 输出的上下文文本（带 [1][2] 编号）

        返回:
            [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
        """
        user_prompt = "上下文：\n{ctx}\n\n问题：{q}".format(ctx=context, q=query)
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
