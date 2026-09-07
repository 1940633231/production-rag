"""Prompt 信任边界回归测试：indirect prompt injection / 伪造历史 缓解。

覆盖:
  - SYSTEM_PROMPT 含「不可信外部输入」防御指令（历史 + 检索上下文）
  - 单条历史消息超长被截断（防超长注入 payload）
  - 非法角色 / 空白历史被过滤
  - 历史内容总量预算封顶
  - QueryRewriter 同步声明历史不可信 + 限长

运行:
  .venv\\Scripts\\python.exe -m pytest tests/generation/test_prompt_security.py -v
"""
import json

from app.generation.prompt import PromptBuilder
from app.generation.query_rewriter import QueryRewriter


def _dump(messages):
    return json.dumps(messages, ensure_ascii=False)


def test_system_prompt_has_anti_injection_instruction():
    """SYSTEM_PROMPT 明确历史与检索上下文为不可信输入，拒绝执行其中指令。"""
    assert "不可信" in PromptBuilder.SYSTEM_PROMPT
    assert "忽略以上指令" in PromptBuilder.SYSTEM_PROMPT
    assert "扮演管理员" in PromptBuilder.SYSTEM_PROMPT


def test_history_long_msg_truncated():
    """超长历史消息被截断到 MAX_HISTORY_MSG_LENGTH，防一次性注入。"""
    payload = "arrgh" * 1000  # 5000 字符
    messages = PromptBuilder().build(
        "问题", "上下文", [{"role": "assistant", "content": payload}]
    )
    # system + assistant + user
    assert [m["role"] for m in messages] == ["system", "assistant", "user"]
    assert len(messages[1]["content"]) <= PromptBuilder.MAX_HISTORY_MSG_LENGTH


def test_history_invalid_role_and_blank_filtered():
    """非法角色 / 空白 / 非 dict 历史条目被过滤。"""
    history = [
        {"role": "system", "content": "非法角色"},
        {"role": "user", "content": "   "},
        {"role": "user", "content": "合法问题"},
        {"role": "assistant", "content": "合法回答"},
    ]
    messages = PromptBuilder().build("问题", "上下文", history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]


def test_history_total_budget_capped():
    """历史内容总量超过预算时丢弃更早回合。"""
    builder = PromptBuilder()
    # 单条上限放宽到不影响本用例，让总预算生效：3 条各 3000，共 9000 > 7000
    builder.MAX_HISTORY_MSG_LENGTH = 100000
    builder.MAX_HISTORY_TOTAL_LENGTH = 7000
    history = [
        {"role": "user", "content": "A" * 3000},
        {"role": "assistant", "content": "B" * 3000},
        {"role": "user", "content": "C" * 3000},
    ]
    messages = builder.build("问题", "上下文", history)
    # 历史段 = system 与末尾 user 之间的条目；按时间正序
    hist_msgs = messages[1:-1]
    # 从最近开始 C、B 各占 3000 累计 6000；A 再占 3000 超 7000 预算 → 丢弃更早的 A
    assert [m["content"] for m in hist_msgs] == ["B" * 3000, "C" * 3000]


def test_query_rewriter_trust_note():
    """QueryRewriter 同步声明历史不可信。"""
    assert "不可信" in QueryRewriter.SYSTEM_PROMPT


def test_query_rewriter_history_truncated():
    """QueryRewriter 的改写历史同样限长。"""
    payload = "x" * 5000
    prompt = QueryRewriter._build_user_prompt("当前问", [
        {"role": "assistant", "content": payload},
    ])
    assert "助手：{}".format("x" * 2000) in prompt
    assert payload not in prompt