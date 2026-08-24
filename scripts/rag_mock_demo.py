"""RAG 完整链路本地测试（零外部依赖，纯 mock 数据）。

用 mock 数据集 + mock retriever + 真 ContextManager + stub generator，
跑通 Retriever → ContextManager → Generator 全链路，验证日志埋点和各阶段逻辑。

不依赖 numpy/faiss/sentence-transformers/jieba，只需 pyyaml（Config 用）。
"""
import sys
from pathlib import Path

# 注入项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.pipeline import RAGPipeline
from app.generation.generator import create_generator
from app.context.manager import ContextManager


# ============================================================
# Mock 数据集（铁矿行业，基于真实数据风格构造）
# 设计要点：
# - doc1 有 chunk 0+1 相邻（触发邻接合并）
# - doc2 有 chunk 0+1 相邻（触发邻接合并）
# - doc3 的 chunk 0 与 doc1 的 chunk 0 内容相同（触发去重）
# - doc4 独立内容（测试跨文档共存）
# ============================================================
MOCK_METADATA = {
    "0": {
        "chunk_id": "doc1_chunk_0",
        "content": "铁矿石供应端，近期全球铁矿发运小幅回落，主要是巴西发运减量明显，澳洲及非主流发运仍有小幅上升。",
        "start_offset": 0,
        "end_offset": 48,
        "metadata": {"document_id": "doc1", "chunk_index": 0, "file_name": "铁矿周报.txt"},
    },
    "1": {
        "chunk_id": "doc1_chunk_1",
        "content": "此外，受台风天气影响，同期国内港口铁矿到货量大幅下滑，尽管疏港也有明显减量，本周港口铁矿库存仍有小幅下降。",
        "start_offset": 48,
        "end_offset": 108,
        "metadata": {"document_id": "doc1", "chunk_index": 1, "file_name": "铁矿周报.txt"},
    },
    "2": {
        "chunk_id": "doc2_chunk_0",
        "content": "需求端，SMM最新调研显示本周铁水产量仍有下降。不过，昨日钢联高炉开工率延续小幅回升。",
        "start_offset": 0,
        "end_offset": 40,
        "metadata": {"document_id": "doc2", "chunk_index": 0, "file_name": "需求分析.txt"},
    },
    "3": {
        "chunk_id": "doc3_chunk_0",
        # 与 doc1_chunk_0 内容完全相同（触发 Jaccard 去重）
        "content": "铁矿石供应端，近期全球铁矿发运小幅回落，主要是巴西发运减量明显，澳洲及非主流发运仍有小幅上升。",
        "start_offset": 0,
        "end_offset": 48,
        "metadata": {"document_id": "doc3", "chunk_index": 0, "file_name": "重复新闻.txt"},
    },
    "4": {
        "chunk_id": "doc2_chunk_1",
        "content": "日均铁水微增0.17万吨至238.2万吨。同期钢厂盈利率小幅回升至33.77%。钢厂铁矿库存继续小幅上升。",
        "start_offset": 40,
        "end_offset": 95,
        "metadata": {"document_id": "doc2", "chunk_index": 1, "file_name": "需求分析.txt"},
    },
    "5": {
        "chunk_id": "doc4_chunk_0",
        "content": "综合而言，当前铁矿基本面供弱需增，库存变动不大，且海运费仍处高位，短期矿价延续偏强震荡运行。",
        "start_offset": 0,
        "end_offset": 45,
        "metadata": {"document_id": "doc4", "chunk_index": 0, "file_name": "综合评述.txt"},
    },
}


class MockRetriever:
    """Mock 检索器：基于字符级匹配率打分，模拟检索相关性。

    不依赖 numpy/faiss，纯 Python 实现。
    """

    def __init__(self, metadata):
        self.metadata = metadata

    def search(self, query, top_k=10):
        query_chars = set(query)
        scored = []
        for idx, doc in self.metadata.items():
            content = doc["content"]
            # 字符级匹配率：query 中有多少字符出现在 content 里
            matched = sum(1 for c in query_chars if c in content)
            score = matched / len(query_chars) if query_chars else 0
            scored.append((idx, score, doc))

        # 按分数降序
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (idx, score, doc) in enumerate(scored[:top_k]):
            results.append({
                "rank": rank,
                "score": float(score),
                "vector_id": int(idx),
                "chunk_id": doc["chunk_id"],
                "content": doc["content"],
                "start_offset": doc.get("start_offset", 0),
                "end_offset": doc.get("end_offset", 0),
                "metadata": doc["metadata"],
            })
        return results


def run_scenario(title, query, max_tokens, top_k=6):
    """跑一个完整场景并打印结果。"""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print("Query: {}".format(query))
    print("max_context_tokens: {}".format(max_tokens))
    print("top_k: {}".format(top_k))
    print()

    retriever = MockRetriever(MOCK_METADATA)
    cm = ContextManager(
        max_context_tokens=max_tokens,
        reserved_tokens=50,
    )
    generator = create_generator("stub")
    pipeline = RAGPipeline(
        retriever=retriever,
        context_manager=cm,
        generator=generator,
        top_k=top_k,
    )

    response = pipeline.run(query)

    # 打印最终结果摘要
    print()
    print("-" * 30 + " 最终结果摘要 " + "-" * 30)
    print("context 长度: {} 字符".format(len(response.context)))
    print("final chunks: {} 条".format(len(response.chunks)))
    print("answer 长度: {} 字符".format(len(response.answer or "")))
    print()
    print("-" * 30 + " Final Context " + "-" * 30)
    print(response.context)
    print()
    if response.answer:
        print("-" * 30 + " Answer (stub) " + "-" * 30)
        print(response.answer[:200])
    return response


if __name__ == "__main__":
    QUERY = "铁矿供应和需求如何？"

    # 场景1: 正常预算（4096），测试去重 + 邻接合并，不触发压缩
    run_scenario(
        "场景1: 正常预算（4096）- 测试去重 + 邻接合并",
        QUERY, max_tokens=4096, top_k=6,
    )

    # 场景2: 小预算（200），测试压缩 + 预算装填 + 截断
    run_scenario(
        "场景2: 小预算（200）- 测试压缩 + 预算装填",
        QUERY, max_tokens=200, top_k=6,
    )

    # 场景3: 极小预算（80），测试重度截断
    run_scenario(
        "场景3: 极小预算（80）- 测试重度截断",
        QUERY, max_tokens=80, top_k=6,
    )

    print()
    print("=" * 70)
    print("全部场景测试完成")
    print("=" * 70)
