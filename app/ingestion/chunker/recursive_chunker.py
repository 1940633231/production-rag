import re
from app.ingestion.chunker.base import BaseChunker
from app.ingestion.document import Document
from app.ingestion.chunk import Chunk


class RecursiveChunker(BaseChunker):
    def __init__(self, chunk_size=800, overlap=120):
        """
        Args:
            chunk_size: 单块最大【字符数】，不是token
            overlap: 块之间重叠【字符数】
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # 优先级从高到低：大段落换行 → 换行 → 中文句末标点
        self.separators = ["\n\n", "\n", "。", "！", "？", "；", "，"]

    def split(self, document: Document):
        text = document.content
        pieces = self._split_text(text, base_offset=0)

        chunks = []
        current_pieces = []
        current_text = ""
        chunk_index = 0

        for piece_text, p_start, p_end in pieces:
            if not piece_text:
                continue
            candidate = current_text + piece_text
            if len(candidate) <= self.chunk_size:
                current_pieces.append((piece_text, p_start, p_end))
                current_text = candidate
            else:
                if current_pieces:
                    chunks.append(
                        self._create_chunk(
                            document, current_text, chunk_index, current_pieces
                        )
                    )
                    chunk_index += 1
                    overlap_pieces = self._tail_pieces(current_pieces, self.overlap)
                    overlap_text = "".join(p[0] for p in overlap_pieces)
                else:
                    overlap_pieces = []
                    overlap_text = ""

                current_pieces = overlap_pieces + [(piece_text, p_start, p_end)]
                current_text = overlap_text + piece_text

        if current_pieces:
            chunks.append(
                self._create_chunk(document, current_text, chunk_index, current_pieces)
            )
        return chunks

    @staticmethod
    def _tail_pieces(pieces, overlap):
        """从pieces尾部取累计长度接近overlap；必要时截断片段前部"""
        if overlap <= 0:
            return []
        result = []
        total = 0
        for piece in reversed(pieces):
            if total >= overlap:
                break
            result.insert(0, piece)
            total += len(piece[0])

        if total > overlap and result:
            first_text, first_start, first_end = result[0]
            drop = total - overlap
            keep_text = first_text[drop:]
            result[0] = (keep_text, first_start + drop, first_end)
        return result

    def _split_text(self, text: str, base_offset: int = 0):
        """真正递归切分；全部分隔符切不动则进入硬兜底"""
        if len(text) <= self.chunk_size:
            return [(text, base_offset, base_offset + len(text))]

        for sep in self.separators:
            parts = self._split_with_offsets(text, sep, base_offset)
            if len(parts) > 1:
                res = []
                for p_text, p_st, p_ed in parts:
                    res.extend(self._split_text(p_text, p_st))
                return res

        # 所有分隔符失效，兜底硬切
        return self._hard_split(text, base_offset)

    @staticmethod
    def _split_with_offsets(text, sep, base):
        """切分并保留分隔符到前一段末尾，维护全局offset"""
        if sep not in text:
            return [(text, base, base + len(text))]
        result = []
        last = 0
        for m in re.finditer(re.escape(sep), text):
            part = text[last : m.end()]
            result.append((part, base + last, base + m.end()))
            last = m.end()
        if last < len(text):
            result.append((text[last:], base + last, base + len(text)))
        return result

    def _hard_split(self, text: str, base_offset: int):
        """兜底：无标点超长文本直接按字符硬切"""
        pieces = []
        pos = 0
        total_len = len(text)
        while pos < total_len:
            end = pos + self.chunk_size
            seg = text[pos:end]
            pieces.append((seg, base_offset + pos, base_offset + end))
            pos = end
        return pieces

    def _create_chunk(self, document, content, index, pieces):
        metadata = document.metadata.copy()
        metadata.update({"document_id": document.document_id, "chunk_index": index})
        start_offset = pieces[0][1] if pieces else 0
        end_offset = pieces[-1][2] if pieces else len(content)
        return Chunk(
            chunk_id=f"{document.document_id}_chunk_{index}",
            document_id=document.document_id,
            chunk_index=index,
            content=content,
            start_offset=start_offset,
            end_offset=end_offset,
            metadata=metadata,
        )
