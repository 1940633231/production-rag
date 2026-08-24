import argparse
from pathlib import Path

from app.core.config import Config
from app.ingestion.loader.txt_loader import TxtLoader
from app.ingestion.writer import IndexWriter


def main():

    # =========================
    # 1. Arguments
    # =========================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy", choices=["fixed", "recursive"], default="recursive"
    )

    args = parser.parse_args()

    # =========================
    # 2. Config
    # =========================

    config = Config()

    # =========================
    # 3. Loader
    # =========================

    loader = TxtLoader()

    document = loader.load("data/raw/test.txt")

    # =========================
    # 4. IndexWriter（统一写入：FAISS + metadata.json + MySQL + ES）
    # =========================

    writer = IndexWriter(config)

    result = writer.write(
        documents=[document],
        strategy=args.strategy,
    )

    # =========================
    # 5. Result
    # =========================

    print()
    print("=" * 50)
    print("知识库构建完成")
    print("=" * 50)

    print(f"Strategy: {args.strategy}")

    print(f"Document 数量：" f"{result['document_count']}")

    print(f"Chunk 数量：" f"{result['chunk_count']}")

    print(f"Embedding 维度：" f"{result['dimension']}")

    print(f"FAISS Index：" f"{result['index_path']}")

    print(f"Metadata：" f"{result['metadata_path']}")

    print(f"MySQL 持久化：" f"{result.get('mysql_persisted', False)}")

    print(f"ES 持久化：" f"{result.get('es_persisted', False)}")


if __name__ == "__main__":
    main()
