import json


class MetadataStore:

    def save(self, docs, path):

        data = {}

        for index, doc in enumerate(docs):

            data[str(index)] = {
                "chunk_id": doc.chunk_id,
                "content": doc.content,
                "start_offset": doc.start_offset,
                "end_offset": doc.end_offset,
                "metadata": doc.metadata,
            }

        with open(path, "w", encoding="utf-8") as f:

            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):

        with open(path, "r", encoding="utf-8") as f:

            return json.load(f)
