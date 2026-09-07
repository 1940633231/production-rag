# -*- coding: utf-8 -*-
"""派生索引对账脚本：以 MySQL chunks 为事实源，比对并修复派生索引。

用法:
  python scripts/reconcile.py --strategy recursive --tenant default [--dry-run]

选项:
  --strategy  分块策略（fixed / recursive），默认 recursive
  --tenant    租户 ID，默认 default
  --dry-run   只报告不修复（fix=False）

示例:
  python scripts/reconcile.py --strategy recursive
  python scripts/reconcile.py --dry-run --tenant tenant_a
"""
import argparse
import json
import sys

from app.core.config import Config
from app.core.logger import get_logger
from app.storage.reconciler import Reconciler

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="派生索引对账（最终一致修复）")
    parser.add_argument("--strategy", default="recursive",
                        choices=["fixed", "recursive"], help="分块策略")
    parser.add_argument("--tenant", default="default", help="租户 ID")
    parser.add_argument("--dry-run", action="store_true",
                        help="只报告差异，不执行修复")
    args = parser.parse_args()

    if not Config().storage_mysql_enabled:
        logger.error("MySQL 未启用，对账以 MySQL 为事实源，无法执行")
        return 2

    report = Reconciler().reconcile(
        strategy=args.strategy, tenant_id=args.tenant, fix=not args.dry_run
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
