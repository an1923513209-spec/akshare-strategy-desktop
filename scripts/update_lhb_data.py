"""CLI for first-time and incremental Dragon-Tiger List downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_decision.lhb_data import update_lhb_data, update_result_dict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并增量更新 A 股龙虎榜与机构席位数据")
    parser.add_argument("--start", default=None, help="开始日期 YYYYMMDD；首次默认读取配置")
    parser.add_argument("--end", default=None, help="结束日期 YYYYMMDD；默认今天")
    parser.add_argument("--force", action="store_true", help="强制重拉指定日期，不按可用性清单跳过")
    args = parser.parse_args()
    result = update_lhb_data(args.start, args.end, force=args.force)
    print(json.dumps(update_result_dict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
