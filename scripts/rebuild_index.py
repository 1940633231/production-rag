"""重建索引脚本：调用 IndexWriter.rebuild 全量重建 FAISS + metadata.json + MySQL + ES。

幂等：先清理旧数据，再从 data/raw/ 加载全部文档写入。

用法:
  .venv\\Scripts\\python.exe scripts\\rebuild_index.py
  .venv\\Scripts\\python.exe scripts\\rebuild_index.py --strategy fixed
"""
import argparse
import sys
from pathlib import Path

# 注入项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env
from app.core.env import load_env
load_env()

from app.ingestion.writer import IndexWriter


def main():
    parser = argparse.ArgumentParser(description="重建索引（FAISS + metadata.json + MySQL + ES）")
    parser.add_argument(
        "--strategy", choices=["fixed", "recursive"], default="recursive",
        help="分块策略（默认 recursive）",
    )
    args = parser.parse_args()

    writer = IndexWriter()
    result = writer.rebuild(strategy=args.strategy)

    print("\n========== 重建结果 ==========")
    print("strategy:        {}".format(args.strategy))
    print("docs:            {}".format(result["document_count"]))
    print("chunks:          {}".format(result["chunk_count"]))
    print("dimension:       {}".format(result["dimension"]))
    print("index_path:      {}".format(result["index_path"]))
    print("metadata_path:   {}".format(result["metadata_path"]))
    print("mysql_deleted:   {}".format(result.get("mysql_deleted", 0)))
    print("es_dropped:      {}".format(result.get("es_dropped", False)))
    print("mysql_persisted: {}".format(result["mysql_persisted"]))
    print("es_persisted:    {}".format(result["es_persisted"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
