#!/usr/bin/env python3
"""
test_de_api.py — 测试 DeApiClient 能否正常查询数据

用法:
  python3 backend/scripts/test_de_api.py
  python3 backend/scripts/test_de_api.py --sql "SELECT neType FROM ... LIMIT 5" --table "..."
  python3 backend/scripts/test_de_api.py --metric "OLT总数"
  python3 backend/scripts/test_de_api.py --metric "OLT槽位利用率分布"

示例输出（--metric "OLT总数"）:
  [1] {"OLT总数": "1602"}

示例输出（--metric "OLT槽位利用率分布"）:
  [1] {"档位": "低(<20%)", "设备数量": "399"}
  [2] {"档位": "中(20%~50%)", "设备数量": "274"}
  [3] {"档位": "中高(50%~70%)", "设备数量": "326"}
  [4] {"档位": "高(>=70%)", "设备数量": "736"}

前置条件:
  backend/config.yaml 已按 config.example.yaml 填写完整
"""

import argparse
import json
import logging
import os
import sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _BACKEND_DIR)

from services.de_sql_execution_client import DeApiClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── 默认测试 SQL ────────────────────────────────────────────────────────
DEFAULT_SQL   = "SELECT neType FROM ads_aggr_unb_eval_ip_networkelement LIMIT 3"
DEFAULT_TABLE = "ads_aggr_unb_eval_ip_networkelement"


def run_sql_test(sql: str, table: str) -> None:
    """直接执行指定 SQL 并打印结果"""
    logger.info("=== SQL 测试 ===")
    logger.info("SQL  : %s", sql)
    logger.info("Table: %s", table)

    with DeApiClient() as client:
        result = client.execute_sql_query(sql, table)

    if result is None:
        logger.error("查询失败，result 为 None")
        sys.exit(1)

    logger.info("查询成功，共 %d 条", len(result))
    print("\n── 结果 ──────────────────────────────")
    for i, row in enumerate(result, 1):
        print(f"  [{i}] {json.dumps(row, ensure_ascii=False)}")
    print()


def run_metric_test(metric_name: str) -> None:
    """从 sample_query_sql.json 查找指标名对应的 SQL 并执行"""
    kb_dir = os.path.join(_BACKEND_DIR, "expert_knowledge")
    sql_file = os.path.join(kb_dir, "评估指标.json")

    if not os.path.exists(sql_file):
        logger.error("找不到 %s", sql_file)
        sys.exit(1)

    with open(sql_file, encoding="utf-8") as f:
        records = json.load(f)

    # 按 name 查找
    matched = next((r for r in records if r.get("name") == metric_name), None)
    if not matched:
        names = [r.get("name") for r in records[:10]]
        logger.error("未找到指标 %r\n前10个指标名: %s", metric_name, names)
        sys.exit(1)

    answer = json.loads(matched.get("answer", "{}"))
    exec_sql = answer.get("exec_sql", "")
    tables   = json.loads(answer.get("extracted_table", "[]"))
    table    = tables[0] if tables else ""

    logger.info("=== 指标测试 ===")
    logger.info("指标  : %s  (id=%s)", metric_name, matched.get("id"))
    logger.info("SQL   : %s", exec_sql)
    logger.info("Table : %s", table)

    with DeApiClient() as client:
        result = client.execute_sql_query(exec_sql, table)

    if result is None:
        logger.error("查询失败")
        sys.exit(1)

    logger.info("查询成功，共 %d 条", len(result))
    print("\n── 结果 ──────────────────────────────")
    for i, row in enumerate(result, 1):
        print(f"  [{i}] {json.dumps(row, ensure_ascii=False)}")
    print()


def main():
    parser = argparse.ArgumentParser(description="测试 DeApiClient")
    parser.add_argument("--sql",    help="自定义 SQL 语句")
    parser.add_argument("--table",  help="自定义表名")
    parser.add_argument("--metric", help="从 sample_query_sql.json 按指标名查询，例如 'AEC覆盖用户数'")
    args = parser.parse_args()

    if args.metric:
        run_metric_test(args.metric)
    elif args.sql:
        table = args.table or args.sql.split("FROM")[-1].strip().split()[0]
        run_sql_test(args.sql, table)
    else:
        # 无参数：跑默认 SQL
        run_sql_test(DEFAULT_SQL, DEFAULT_TABLE)


if __name__ == "__main__":
    main()
