"""
sql_executor.py — L5 指标名 → 执行 SQL → 返回结构化结果

职责:
  - 加载 expert_knowledge/评估指标.json，按 name 建索引
  - execute_metric(name)   → {rows, render_type, col_x, col_y}
  - get_scalar(name)       → 第一行第一列的数值（用于 condition 判断）
  - rows_to_markdown(rows) → Markdown 表格字符串
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR  = os.path.dirname(_SERVICES_DIR)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from services.de_sql_execution_client import DeApiClient

logger = logging.getLogger(__name__)

_KB_DIR       = os.path.join(_BACKEND_DIR, "expert_knowledge")
_METRICS_FILE = os.path.join(_KB_DIR, "评估指标.json")


class SqlExecutor:
    """
    L5 指标名到 SQL 执行的封装。

    使用方式（推荐在 async context 中配合 DeApiClient）:
        executor = SqlExecutor()
        with DeApiClient() as client:
            result = executor.execute_metric("AEC覆盖用户数", client)
    """

    def __init__(self):
        self._index: Dict[str, Dict] = {}   # name → 指标记录
        self._load()

    # ── 公共接口 ──────────────────────────────────────────────

    def execute_metric(self, name: str, client: DeApiClient) -> Optional[Dict[str, Any]]:
        """
        按指标名执行 SQL，返回:
          {
            "rows":        List[Dict],   # 原始行数据
            "render_type": str,          # "TABLE" / "BAR" / "LINE" / None
            "col_x":       str | None,
            "col_y":       str | None,
          }
        失败返回 None。
        """
        record = self._index.get(name)
        if not record:
            logger.warning("[SqlExecutor] 未找到指标: %r", name)
            return None

        sql, table = self._parse_sql(record)
        if not sql:
            logger.warning("[SqlExecutor] 指标 %r 无 exec_sql", name)
            return None

        rows = client.execute_sql_query(sql, table)
        if rows is None:
            return None

        return {
            "rows":        rows,
            "render_type": record.get("renderType"),
            "col_x":       record.get("colX"),
            "col_y":       record.get("colY"),
        }

    def get_scalar(self, name: str, client: DeApiClient) -> Optional[float]:
        """
        执行指标并返回第一行第一列的数值，用于 condition 判断。
        失败或无数据返回 None。
        """
        result = self.execute_metric(name, client)
        if not result or not result["rows"]:
            return None
        first_row = result["rows"][0]
        first_val = next(iter(first_row.values()), None)
        try:
            return float(first_val)
        except (TypeError, ValueError):
            return None

    def eval_condition(self, condition: str, client: DeApiClient) -> bool:
        """
        评估 L4 的 condition 表达式，决定该节是否展示。

        支持: ${number("指标名") > 0}  / =0 / >=N / <N 等
        无法解析时默认返回 True（展示）。
        """
        if not condition:
            return True

        m = re.match(
            r'\$\{number\("([^"]+)"\)\s*([><=!]+)\s*([^}]+)\}',
            condition.strip()
        )
        if not m:
            logger.debug("[SqlExecutor] 无法解析 condition: %r，默认展示", condition)
            return True

        metric_name = m.group(1)
        operator    = m.group(2)
        threshold   = m.group(3).strip()

        scalar = self.get_scalar(metric_name, client)
        if scalar is None:
            logger.debug("[SqlExecutor] 条件指标 %r 无数据，默认展示", metric_name)
            return True

        try:
            t = float(threshold)
            ops = {">": scalar > t, ">=": scalar >= t,
                   "<": scalar < t,  "<=": scalar <= t,
                   "=": scalar == t,  "!=": scalar != t}
            result = ops.get(operator, True)
            logger.info("[SqlExecutor] condition %r → scalar=%.2f %s %.2f → %s",
                        condition, scalar, operator, t, result)
            return result
        except (ValueError, TypeError):
            return True

    @staticmethod
    def rows_to_markdown(rows: List[Dict]) -> str:
        """将行数据转为 Markdown 表格字符串。空数据返回空字符串。"""
        if not rows:
            return ""

        headers = list(rows[0].keys())
        lines   = ["| " + " | ".join(str(h) for h in headers) + " |",
                   "| " + " | ".join("---" for _ in headers) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        return "\n".join(lines)

    # ── 内部 ──────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(_METRICS_FILE):
            logger.warning("[SqlExecutor] 找不到 %s", _METRICS_FILE)
            return
        with open(_METRICS_FILE, encoding="utf-8") as f:
            records = json.load(f)
        self._index = {r["name"]: r for r in records if r.get("name")}
        logger.info("[SqlExecutor] 加载 %d 条指标", len(self._index))

    @staticmethod
    def _parse_sql(record: Dict) -> tuple[str, str]:
        """从 answer 字段解析 exec_sql 和 table_name。"""
        try:
            answer = json.loads(record.get("answer", "{}"))
            sql    = answer.get("exec_sql", "")
            tables = json.loads(answer.get("extracted_table", "[]"))
            table  = tables[0] if tables else ""
            return sql, table
        except (json.JSONDecodeError, TypeError, IndexError):
            return "", ""
