"""
sql_executor.py — L5 指标名 → 执行 SQL → 返回结构化结果

职责:
  - 加载 expert_knowledge/node.json（过滤 level==5），按 name 建索引
  - 若存在 评估指标_mock.json，按 id 叠加 mock_data 字段
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

_KB_DIR            = os.path.join(_BACKEND_DIR, "expert_knowledge")
_NODE_FILE         = os.path.join(_KB_DIR, "node.json")
_METRICS_MOCK_FILE = os.path.join(_KB_DIR, "评估指标_mock.json")   # mock_data 来源，不变


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
            "rows":        List[Dict],
            "render_type": str,
            "col_x":       str | None,
            "col_y":       str | None,
          }
        FORCE_MOCK=true 时直接用 mock_data，跳过真实 SQL。
        否则优先走真实 API；查询失败或无数据时回落到 mock_data；都没有返回 None。
        """
        record = self._index.get(name)
        if not record:
            logger.warning("[SqlExecutor] 未找到指标: %r", name)
            return None

        def _wrap(rows):
            return {
                "rows":        rows,
                "render_type": record.get("renderType"),
                "col_x":       record.get("colX"),
                "col_y":       record.get("colY"),
            }

        force_mock = os.environ.get("FORCE_MOCK", "").lower() in ("1", "true", "yes")

        if not force_mock:
            sql    = record.get("exec_sql", "")
            tables = record.get("extracted_table") or []
            table  = tables[0] if tables else ""
            if sql:
                rows = client.execute_sql_query(sql, table)
                if rows:
                    logger.info("[SqlExecutor] 真实查询成功: %r，%d 行", name, len(rows))
                    return _wrap(rows)
                logger.warning("[SqlExecutor] 真实查询返回空结果: %r，尝试 mock_data", name)
            else:
                logger.warning("[SqlExecutor] 指标 %r 无 exec_sql，尝试 mock_data", name)
        else:
            logger.info("[SqlExecutor] FORCE_MOCK=true，跳过真实 SQL: %r", name)

        # 回落到 mock_data
        mock = record.get("mock_data")
        if mock is not None:
            logger.info("[SqlExecutor] 使用 mock_data: %r", name)
            return _wrap(mock)

        return None

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

    def eval_condition(self, condition: str, client: DeApiClient,
                       collected: Dict[str, List] = None) -> bool:
        """
        评估 L4 的 condition 表达式，决定该节是否展示。

        优先从 collected（已查询缓存）取值，避免重复 SQL；
        collected 中无数据时回落到 get_scalar 查询。
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

        # 优先用已收集的查询结果
        scalar = None
        if collected and metric_name in collected:
            rows = collected[metric_name]
            if rows:
                first_val = next(iter(rows[0].values()), None)
                try:
                    scalar = float(first_val)
                except (TypeError, ValueError):
                    pass
            logger.debug("[SqlExecutor] condition 指标 %r 从 collected 取值: %s", metric_name, scalar)

        if scalar is None:
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

        dict_rows = [r for r in rows if isinstance(r, dict)]
        if not dict_rows:
            # API 返回了非 dict 行（如裸字符串），降级为逐行输出
            return "\n".join(str(r) for r in rows)

        rows = dict_rows
        headers = list(rows[0].keys())
        lines   = ["| " + " | ".join(str(h) for h in headers) + " |",
                   "| " + " | ".join("---" for _ in headers) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        return "\n".join(lines)

    # ── 内部 ──────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(_NODE_FILE):
            logger.warning("[SqlExecutor] 找不到 node.json")
            return
        with open(_NODE_FILE, encoding="utf-8") as f:
            all_nodes = json.load(f)
        l5 = [n for n in all_nodes if n.get("level") == 5 and n.get("name")]
        self._index = {n["name"]: n for n in l5}

        # 将 评估指标_mock.json 中的 mock_data 按 id 叠加进索引
        mock_count = 0
        if os.path.exists(_METRICS_MOCK_FILE):
            with open(_METRICS_MOCK_FILE, encoding="utf-8") as f:
                mock_records = json.load(f)
            id_to_mock = {r["id"]: r["mock_data"] for r in mock_records if r.get("id") and "mock_data" in r}
            for node in self._index.values():
                if node.get("id") in id_to_mock:
                    node["mock_data"] = id_to_mock[node["id"]]
                    mock_count += 1

        logger.info("[SqlExecutor] 加载 %d 条 L5 指标（%d 条含 mock_data）",
                    len(self._index), mock_count)
