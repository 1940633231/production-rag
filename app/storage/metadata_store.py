import json

from app.ingestion.chunk import vector_id_for


class MetadataStore:
    """metadata.json 读写。

    稳定 ID 索引：key 为 str(vector_id)（由 chunk_id 派生），不再依赖连续位置。
    因此删除单个向量后其余条目的 key 不变，无需整体重写（删除时按 key 摘除即可）。
    """

    def save(self, docs, path):
        data = {}
        for doc in docs:
            vid = int(getattr(doc, "vector_id", 0)) or vector_id_for(doc.chunk_id)
            data[str(vid)] = self._entry(doc, vid)
        self.save_entries(data, path)

    @staticmethod
    def _entry(doc, vector_id: int) -> dict:
        return {
            "chunk_id": doc.chunk_id,
            "document_id": doc.document_id,
            "vector_id": int(vector_id),
            "content": doc.content,
            "start_offset": doc.start_offset,
            "end_offset": doc.end_offset,
            "metadata": doc.metadata,
        }

    def save_entries(self, data: dict, path):
        """直接写 dict（{str(vector_id): entry}），供追加合并使用。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
