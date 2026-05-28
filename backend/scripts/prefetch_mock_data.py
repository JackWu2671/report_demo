"""
prefetch_mock_data.py — 批量执行评估指标 SQL，将结果写入 mock_data 字段。

用法:
  python3 tools/prefetch_mock_data.py                        # 默认路径
  python3 tools/prefetch_mock_data.py input.json output.json
  python3 tools/prefetch_mock_data.py --force                # 忽略已有 mock_data，全量重跑

默认输入:  expert_knowledge/评估指标.json
默认输出:  expert_knowledge/评估指标_mock.json

增量策略：已有 mock_data（包括 null）的条目跳过，除非 --force。
并发数由 MAX_WORKERS 控制，默认 5。
"""

import argparse
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

MAX_WORKERS = 5


def _parse_record(record: dict) -> tuple[str, str]:
    """从 answer 字段解析出 (sql, table_name)。"""
    try:
        answer = json.loads(record.get("answer", "{}"))
        sql = answer.get("exec_sql", "")
        tables = json.loads(answer.get("extracted_table", "[]"))
        table = tables[0] if tables else ""
        return sql, table
    except (json.JSONDecodeError, TypeError):
        return "", ""


def fetch_one(record: dict) -> dict:
    """执行单条指标的 SQL，返回附带 mock_data 的新 record。"""
    name = record.get("name", record.get("id", "?"))
    sql, table = _parse_record(record)

    if not sql:
        logger.warning("[%s] 无 exec_sql，mock_data=null", name)
        return {**record, "mock_data": None}

    try:
        with DeApiClient() as client:
            rows = client.execute_sql_query(sql, table)
        count = len(rows) if rows else 0
        logger.info("[%s] 完成 — %d 行", name, count)
        return {**record, "mock_data": rows}
    except Exception as e:
        logger.error("[%s] 查询异常: %s", name, e)
        return {**record, "mock_data": None}


def prefetch(input_path: str, output_path: str, force: bool = False) -> None:
    with open(input_path, encoding="utf-8") as f:
        records = json.load(f)

    total = len(records)
    logger.info("共 %d 条指标，force=%s", total, force)

    to_fetch = [r for r in records if force or "mock_data" not in r]
    cached   = [r for r in records if not force and "mock_data" in r]
    logger.info("需要执行: %d 条，跳过（已缓存）: %d 条", len(to_fetch), len(cached))

    results: dict[str, dict] = {r["id"]: r for r in cached}

    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, r): r for r in to_fetch}
        for future in as_completed(futures):
            result = future.result()
            results[result["id"]] = result
            completed += 1
            logger.info("进度: %d / %d", completed, len(to_fetch))

    # 保持原始顺序
    output = [results.get(r["id"], r) for r in records]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    success = sum(1 for r in output if r.get("mock_data") is not None)
    logger.info("完成 — 成功: %d / %d，已写入 %s", success, total, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量执行评估指标 SQL 并缓存结果到 mock_data 字段")
    parser.add_argument("input",  nargs="?",
                        default=os.path.join(_BACKEND, "expert_knowledge", "评估指标.json"))
    parser.add_argument("output", nargs="?",
                        default=os.path.join(_BACKEND, "expert_knowledge", "评估指标_mock.json"))
    parser.add_argument("--force", action="store_true",
                        help="强制重新执行全部，忽略已有 mock_data")
    args = parser.parse_args()

    prefetch(args.input, args.output, force=args.force)
