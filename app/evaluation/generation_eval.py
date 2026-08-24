"""生成质量评估：Faithfulness（忠实度）+ Relevance（相关性）。

采用 LLM-as-Judge 范式，复用项目已有的 BaseGenerator 接口（QwenGenerator）调用大模型。

Faithfulness 评估流程（两步）：
  Step 1 - 提取事实陈述：让 LLM 从 answer 中拆分出独立的事实性声明
  Step 2 - 逐条验证：让 LLM 判断每条 claim 是否可由 context 推导出
  最终 score = supported_claims / total_claims

Relevance 评估流程（单步）：
  让 LLM 对 answer 与 query 的语义相关性打分（0.0 ~ 1.0），并给出理由

使用示例:
    from app.evaluation.generation_eval import FaithfulnessEvaluator, RelevanceEvaluator
    from app.generation.generator import create_generator

    judge = create_generator("qwen", model="qwen-turbo")

    faith_eval = FaithfulnessEvaluator(judge)
    result = faith_eval.evaluate(query, answer, context)
    print(result["score"], result["claims"])

    rel_eval = RelevanceEvaluator(judge)
    result = rel_eval.evaluate(query, answer)
    print(result["score"], result["reasoning"])
"""
import json
import re
from typing import Dict, List, Optional

from app.core.logger import get_logger
from app.generation.generator import BaseGenerator

logger = get_logger(__name__)


class FaithfulnessEvaluator:
    """忠实度评估器：验证 answer 中的事实陈述是否可由 context 推导。

    两步 LLM 调用：
      1. 提取 answer 中的事实陈述（claims）
      2. 逐条判断 claim 是否被 context 支持
    """

    # Step 1: 提取事实陈述
    CLAIM_EXTRACTION_PROMPT = (
        "请从以下回答中提取所有独立的事实性陈述。\n"
        "要求：\n"
        "1. 每条陈述应是一个可独立验证的断言\n"
        "2. 去除主观评价和重复内容\n"
        "3. 保持原文语义，不要改写\n"
        "4. 以 JSON 数组格式返回，如 [\"陈述1\", \"陈述2\"]\n\n"
        "回答内容：\n{answer}\n\n"
        "请返回 JSON 数组，不要包含其他文字。"
    )

    # Step 2: 逐条验证
    CLAIM_VERIFICATION_PROMPT = (
        "请逐条验证以下事实陈述是否可由给定上下文推导出。\n"
        "判断规则：\n"
        "- supported: 上下文中有明确依据可推导出该陈述\n"
        "- unsupported: 上下文中无依据或与上下文矛盾\n"
        "- partial: 上下文有部分依据但不足以完全支持\n\n"
        "上下文：\n{context}\n\n"
        "待验证陈述：\n{claims_json}\n\n"
        "请以 JSON 数组格式返回，每个元素包含：\n"
        '  {{"claim": "陈述原文", "verdict": "supported/unsupported/partial", '
        '"evidence": "上下文中的依据或缺失原因"}}\n'
        "只返回 JSON 数组，不要包含其他文字。"
    )

    def __init__(self, judge: BaseGenerator):
        self.judge = judge

    def evaluate(
        self, query: str, answer: str, context: str
    ) -> Dict:
        """评估 answer 相对 context 的忠实度。

        参数:
            query: 用户查询（用于上下文理解，不直接参与判断）
            answer: LLM 生成的答案
            context: RAG 检索拼装的上下文文本

        返回:
            {
                "score": float,          # 0.0~1.0, supported/total
                "claims": [              # 逐条验证结果
                    {
                        "claim": str,
                        "verdict": "supported"|"unsupported"|"partial",
                        "evidence": str
                    }, ...
                ],
                "total_claims": int,
                "supported_claims": int,
            }
        """
        logger.info("Faithfulness 评估开始: answer_len=%d", len(answer))

        # 边界情况：空答案
        if not answer or not answer.strip():
            logger.warning("答案为空，faithfulness 评估返回 0")
            return self._empty_result()

        # Step 1: 提取事实陈述
        claims = self._extract_claims(answer)
        if not claims:
            logger.warning("未提取到任何事实陈述，faithfulness 返回 1.0（无可验证声明）")
            return {
                "score": 1.0,
                "claims": [],
                "total_claims": 0,
                "supported_claims": 0,
                "note": "答案中未提取到可验证的事实陈述",
            }

        logger.info("提取到 %d 条事实陈述", len(claims))

        # Step 2: 逐条验证
        verified_claims = self._verify_claims(claims, context)
        supported = sum(
            1 for c in verified_claims if c.get("verdict") == "supported"
        )
        partial = sum(
            1 for c in verified_claims if c.get("verdict") == "partial"
        )
        # partial 算 0.5 分
        score = (supported + 0.5 * partial) / len(claims)

        logger.info(
            "Faithfulness 评估完成: score=%.2f, total=%d, supported=%d, partial=%d, unsupported=%d",
            score, len(claims), supported, partial,
            len(claims) - supported - partial,
        )

        return {
            "score": round(score, 4),
            "claims": verified_claims,
            "total_claims": len(claims),
            "supported_claims": supported,
        }

    def _extract_claims(self, answer: str) -> List[str]:
        """调用 LLM 从 answer 中提取事实陈述列表。"""
        prompt = self.CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = self.judge.generate(messages)
            claims = self._parse_json_array(raw)
            # 过滤空字符串和过短的无意义片段
            claims = [c.strip() for c in claims if c and len(c.strip()) > 2]
            return claims
        except Exception as e:
            logger.error("提取事实陈述失败: %s", e, exc_info=True)
            return []

    def _verify_claims(
        self, claims: List[str], context: str
    ) -> List[Dict]:
        """调用 LLM 逐条验证 claims 是否可由 context 推导。"""
        claims_json = json.dumps(claims, ensure_ascii=False, indent=2)
        prompt = self.CLAIM_VERIFICATION_PROMPT.format(
            context=context, claims_json=claims_json
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = self.judge.generate(messages)
            results = self._parse_json_array(raw)
            # 确保每条结果有必需字段
            verified = []
            for item in results:
                if isinstance(item, dict):
                    verified.append({
                        "claim": item.get("claim", ""),
                        "verdict": item.get("verdict", "unsupported"),
                        "evidence": item.get("evidence", ""),
                    })
            # 如果解析失败，回退为全部 unsupported
            if not verified:
                logger.warning("LLM 验证结果解析失败，回退为全部 unsupported")
                verified = [
                    {"claim": c, "verdict": "unsupported", "evidence": "解析失败"}
                    for c in claims
                ]
            return verified
        except Exception as e:
            logger.error("验证事实陈述失败: %s", e, exc_info=True)
            return [
                {"claim": c, "verdict": "unsupported", "evidence": "验证异常: {}".format(e)}
                for c in claims
            ]

    def _empty_result(self) -> Dict:
        return {
            "score": 0.0,
            "claims": [],
            "total_claims": 0,
            "supported_claims": 0,
        }

    @staticmethod
    def _parse_json_array(text: str) -> list:
        """从 LLM 输出中提取 JSON 数组。

        LLM 可能在 JSON 前后加 markdown 标记或解释文字，
        用正则提取第一个 JSON 数组。
        """
        # 去除 markdown 代码块标记
        text = text.strip()
        if text.startswith("```"):
            # 去掉 ```json 或 ``` 开头和结尾的 ```
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 回退：正则提取 [ ... ] 块
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("JSON 数组解析失败, raw text: %s", text[:200])
        return []


class RelevanceEvaluator:
    """相关性评估器：评估 answer 与 query 的语义相关性。

    单步 LLM 调用：让 LLM 对答案与问题的相关性打分（0.0~1.0），
    并给出评分理由。
    """

    RELEVANCE_PROMPT = (
        "请评估以下回答与用户问题的语义相关性。\n"
        "评分标准：\n"
        "- 1.0: 回答完全切中问题，信息完整且无冗余\n"
        "- 0.7~0.9: 回答基本切题，但有少量遗漏或偏移\n"
        "- 0.4~0.6: 回答部分相关，存在明显遗漏或偏题内容\n"
        "- 0.1~0.3: 回答仅有少量相关内容，大部分偏题\n"
        "- 0.0: 回答完全无关或未回答问题\n\n"
        "用户问题：{query}\n\n"
        "回答内容：\n{answer}\n\n"
        "请以 JSON 格式返回，包含以下字段：\n"
        '  {{"score": 0.0-1.0之间的浮点数, "reasoning": "评分理由"}}\n'
        "只返回 JSON 对象，不要包含其他文字。"
    )

    def __init__(self, judge: BaseGenerator):
        self.judge = judge

    def evaluate(self, query: str, answer: str) -> Dict:
        """评估 answer 相对 query 的相关性。

        参数:
            query: 用户查询
            answer: LLM 生成的答案

        返回:
            {
                "score": float,       # 0.0~1.0
                "reasoning": str,     # LLM 给出的评分理由
            }
        """
        logger.info("Relevance 评估开始: query=%r, answer_len=%d", query[:50], len(answer))

        # 边界情况
        if not answer or not answer.strip():
            logger.warning("答案为空，relevance 返回 0")
            return {"score": 0.0, "reasoning": "答案为空"}

        prompt = self.RELEVANCE_PROMPT.format(query=query, answer=answer)
        messages = [{"role": "user", "content": prompt}]

        try:
            raw = self.judge.generate(messages)
            result = self._parse_json_object(raw)
            score = float(result.get("score", 0.0))
            # 确保分数在 [0, 1] 范围
            score = max(0.0, min(1.0, score))
            reasoning = result.get("reasoning", "")

            logger.info("Relevance 评估完成: score=%.2f", score)
            return {
                "score": round(score, 4),
                "reasoning": reasoning,
            }
        except Exception as e:
            logger.error("Relevance 评估失败: %s", e, exc_info=True)
            return {
                "score": 0.0,
                "reasoning": "评估异常: {}".format(e),
            }

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        """从 LLM 输出中提取 JSON 对象。"""
        text = text.strip()
        # 去除 markdown 代码块标记
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 回退：正则提取 { ... } 块
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("JSON 对象解析失败, raw text: %s", text[:200])
        return {}
