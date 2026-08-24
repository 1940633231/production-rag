from app.ingestion.chunker.base import BaseChunker
from app.ingestion.document import Document
from app.ingestion.chunk import Chunk


class Chunker(BaseChunker):

    def __init__(self, chunk_size=500, overlap=100):

        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, document: Document):

        text = document.content

        chunks = []

        start = 0

        chunk_index = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunk_id = f"{document.document_id}" f"_chunk_{chunk_index}"

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        chunk_index=chunk_index,
                        content=chunk_text,
                        start_offset=start,
                        end_offset=end,
                        metadata=document.metadata.copy(),
                    )
                )

            chunk_index += 1

            start = end - self.overlap

        return chunks
