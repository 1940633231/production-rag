import time

from sentence_transformers import SentenceTransformer

from app.core.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:

    def __init__(self, model_name):
        t = time.time()
        logger.info("加载 embedding 模型: %s", model_name)
        self.model = SentenceTransformer(model_name)
        # 缓存向量维度，供 FAISSStore 动态使用（避免换模型时硬编码崩溃）
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            "embedding 模型加载完成: %.3fs, dim=%d",
            time.time() - t, self.dimension,
        )

    def encode(self, texts):
        t = time.time()
        n = len(texts) if hasattr(texts, "__len__") else 1
        logger.info("向量编码开始: texts=%d", n)
        result = self.model.encode(texts, normalize_embeddings=True)
        logger.info("向量编码完成: %.3fs", time.time() - t)
        return result
