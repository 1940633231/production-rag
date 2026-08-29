"""Prompt 构建：组装 system + 历史对话 + user 消息，送入 LLM。

设计要点：
- system prompt 约束 LLM 基于上下文回答，不编造，并引用 [1][2] 编号
- user prompt 注入 context（ContextManager 已带 [1][2] 编号）+ query
- 多轮对话：history 以独立 user/assistant 消息插入 system 与当前 user 之间，
  仅保留最近 N 轮（MAX_HISTORY_TURNS），避免 prompt 过长
- 返回 OpenAI/DashScope 兼容的 messages 格式：[{role, content}, ...]
"""
from typing import Dict, List, Optional


class PromptBuilder:
    """构建 LLM messages。"""

    SYSTEM_PROMPT = (
        "你是一个严谨的中文问答助手。请基于提供的上下文回答用户问题。"
        "要求：\n"
        "1. 仅使用上下文中的信息，不要编造或引用外部知识\n"
        "2. 如果上下文没有相关信息，明确说明\"根据现有信息无法回答\"\n"
        "3. 引用上下文来源时使用方括号编号，格式见下方示例\n"
        "4. 回答简洁、结构清晰\n"
        "5. 可结合对话历史理解用户指代（如\"它\"\"刚才\"等），但回答以当前上下文为准\n"
        "引用格式示例：\n"
        "- 单个来源：[1]\n"
        "- 多个来源：[1][2]（相邻连写）\n"
        "- 或多个来源写成同一括号：[1,2]（用逗号/顿号分隔均可）\n"
        "例如：铁矿石供给增加而需求下降[1,2]，供需格局偏宽松。"
    )

    # 最多保留的对话轮数（每轮 = 1 条 user + 1 条 assistant）
    MAX_HISTORY_TURNS = 5

    def build(
        self,
        query: str,
        context: str,
        history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """构建 messages 列表（system + 历史对话 + 当前 user）。

        参数:
            query: 用户查询
            context: ContextManager 输出的上下文文本（带 [1][2] 编号）
            history: 多轮对话历史 [{role: user/assistant, content}, ...]，默认 None

        返回:
            [{"role": "system", ...}, {"role": "user", ...}, ...,
             {"role": "user", "content": "上下文 + 当前问题"}]
        """
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        # 仅保留最近 MAX_HISTORY_TURNS 轮（每轮 2 条消息），并过滤非法条目
        for msg in (history or [])[-self.MAX_HISTORY_TURNS * 2:]:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        user_prompt = "上下文：\n{ctx}\n\n问题：{q}".format(ctx=context, q=query)
        messages.append({"role": "user", "content": user_prompt})
        return messages
