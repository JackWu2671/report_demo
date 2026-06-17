"""
prefetch_mock_data.py — 批量执行评估指标 SQL，将结果写入 mock_data 字段。

直接运行: python3 scripts/prefetch_mock_data.py
增量策略：已有 mock_data 的条目跳过；改 FORCE=True 强制全量重跑。
"""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from services.de_sql_execution_client import DeApiClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("prefetch")

INPUT_FILE  = os.path.join(_BACKEND, "reference", "评估指标.json")
OUTPUT_FILE = os.path.join(_BACKEND, "reference", "评估指标_mock.json")
MAX_WORKERS = 5
FORCE       = False   # 改为 True 则忽略已有 mock_data，全量重跑


def _parse_record(record: dict) -> tuple[str, str]:
    try:
        answer = json.loads(record.get("answer", "{}"))
        sql = answer.get("exec_sql", "")
        tables = json.loads(answer.get("extracted_table", "[]"))
        table = tables[0] if tables else ""
        return sql, table
    except (json.JSONDecodeError, TypeError):
        return "", ""


def fetch_one(record: dict) -> dict:
    name = record.get("name", record.get("id", "?"))
    sql, table = _parse_record(record)

    if not sql:
        logger.warning("[%s] 无 exec_sql，mock_data=null", name)
        return {**record, "mock_data": None}

    try:
        with DeApiClient() as client:
            rows = client.execute_sql_query(sql, table)
        logger.info("[%s] 完成 — %d 行", name, len(rows) if rows else 0)
        return {**record, "mock_data": rows}
    except Exception as e:
        logger.error("[%s] 查询异常: %s", name, e)
        return {**record, "mock_data": None}


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        records = json.load(f)

    to_fetch = [r for r in records if FORCE or "mock_data" not in r]
    cached   = [r for r in records if not FORCE and "mock_data" in r]
    logger.info("共 %d 条，需执行: %d，跳过: %d", len(records), len(to_fetch), len(cached))

    results = {r["id"]: r for r in cached}

    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, r): r for r in to_fetch}
        for future in as_completed(futures):
            result = future.result()
            results[result["id"]] = result
            done += 1
            logger.info("进度: %d / %d", done, len(to_fetch))

    output = [results.get(r["id"], r) for r in records]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in output if r.get("mock_data") is not None)
    logger.info("完成 — 成功: %d / %d，已写入 %s", success, len(records), OUTPUT_FILE)


if __name__ == "__main__":
    main()
