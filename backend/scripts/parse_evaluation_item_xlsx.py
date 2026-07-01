#!/usr/bin/env python3
"""
parse_evaluation_item_xlsx.py — 将评估项.xlsx 转换为 评估项.json

输入: reference/评估项.xlsx
输出: reference/评估项.json

Excel 列说明:
  SCENEKEY          — 场景标识（备用，取 name 优先）
  CONTENT           — 原始 JSON 字符串
  CONDITION         — 整体展示条件（手动填写，如 ${number("AEC覆盖用户数")>0}）
                      为空时自动从 expandLogic showWhen 行提取
  CONDITION_QUERIES — 条件相关的指标名，逗号分隔（如 AEC覆盖用户数）
  DESCRIPTION       — 章节导语（手动撰写，无占位符）
                      为空时回退到 CONTENT.description

输出字段说明:
  uuid              — 原始 UUID
  id                — 短编号（如 L4_001）
  name              — 评估项名称
  level             — 固定 4
  description       — 章节导语（无占位符，展示给用户）
  keywords          — 关键词列表
  sampleIssue       — 示例提问
  condition         — 整体展示条件表达式
  condition_queries — 条件相关指标名列表（用于执行前判断是否展示本节）
  summarySuggestion — LLM 总结指令
  descriptionSuggestion — expandLogic 原文（含 ${} 占位符，执行引擎渲染报告）
  children          — 关联的评估指标名列表
"""

import json
import logging
import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_KB_DIR = os.path.join(_BACKEND_DIR, "reference")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── 配置区 ────────────────────────────────────────────────────────────
INPUT_FILE = os.path.join(_KB_DIR, "评估项.xlsx")
OUTPUT_FILE = os.path.join(_KB_DIR, "评估项.json")
ID_PREFIX = "L4"   # 短 id 前缀，生成 L4_001 / L4_002 ...
ID_START = 1       # 起始序号
# ─────────────────────────────────────────────────────────────────────


# ── id 生成 ───────────────────────────────────────────────────────────

def make_short_id(index: int) -> str:
    return f"{ID_PREFIX}_{index:03d}"


# ── 字段解析 ──────────────────────────────────────────────────────────

def extract_condition(expand_logic: str) -> str:
    """
    从 expandLogic 第一行提取 showWhen 条件（备用，Excel 有值时不调用）。
    示例：## 标题.showWhen(${number("AEC覆盖用户数")=0})
    → ${number("AEC覆盖用户数")=0}
    """
    if not expand_logic:
        return ""
    first_line = expand_logic.split("\n")[0]
    m = re.search(r"\.showWhen\((.+)\)\s*$", first_line)
    return m.group(1).strip() if m else ""


def parse_condition_queries(text: str) -> list[str]:
    """
    将 CONDITION_QUERIES 列解析为指标名列表。
    支持中英文逗号或换行分隔，自动去除空项。
    示例：'AEC覆盖用户数' → ['AEC覆盖用户数']
    """
    if not text or not text.strip():
        return []
    parts = re.split(r"[,，\n]", text)
    return [p.strip() for p in parts if p.strip()]


def extract_condition_queries(condition: str) -> list[str]:
    """
    从 condition 表达式中提取所有 number("指标名") 里的指标名。
    示例：'${number("AEC覆盖用户数")>0}' → ['AEC覆盖用户数']
    Excel 无 CONDITION_QUERIES 列时作为回退。
    """
    if not condition:
        return []
    return re.findall(r'number\("([^"]+)"\)', condition)


# ── 行转换 ────────────────────────────────────────────────────────────

def convert_row(
    scene_key: str,
    content_str: str,
    xl_condition: str,
    xl_condition_queries: str,
    xl_description: str,
    index: int,
) -> dict | None:
    try:
        obj = json.loads(content_str)
    except (json.JSONDecodeError, TypeError):
        logging.warning("SCENEKEY=%r CONTENT 解析失败，跳过", scene_key)
        return None

    expand_logic = obj.get("expandLogic", "")

    # condition: Excel 列优先，为空时从 expandLogic 提取
    condition = xl_condition.strip() if xl_condition and xl_condition.strip() \
        else extract_condition(expand_logic)

    # condition_queries: Excel 列优先，为空时从 condition 表达式自动提取
    condition_queries = parse_condition_queries(xl_condition_queries) \
        or extract_condition_queries(condition)

    # description: Excel 列优先，为空时留空
    description = xl_description.strip() if xl_description and xl_description.strip() else ""

    return {
        "uuid": obj.get("id", ""),
        "id": make_short_id(index),
        "name": obj.get("name", scene_key),
        "level": 4,
        "description": description,
        "keywords": obj.get("keyWords") or [],
        "sampleIssue": obj.get("sampleIssue", ""),
        "condition": condition,
        "condition_queries": condition_queries,
        "summarySuggestion": obj.get("summarySuggestion") or "",
        "descriptionSuggestion": expand_logic,
        "children": obj.get("metrics") or [],
    }


# ── 主流程 ────────────────────────────────────────────────────────────

def main():
    try:
        import openpyxl
    except ImportError:
        logging.error("请先安装 openpyxl: pip install openpyxl")
        sys.exit(1)

    if not os.path.exists(INPUT_FILE):
        logging.error("文件不存在: %s", INPUT_FILE)
        sys.exit(1)

    wb = openpyxl.load_workbook(INPUT_FILE, data_only=True)
    ws = wb.active

    headers = [str(c.value).strip().upper() if c.value else "" for c in ws[1]]

    # 必要列
    try:
        idx_key = headers.index("SCENEKEY")
        idx_content = headers.index("CONTENT")
    except ValueError:
        logging.error("找不到必要列 SCENEKEY/CONTENT，实际表头: %s", headers)
        sys.exit(1)

    # 可选新增列（兼容旧版 Excel）
    def _col(name: str) -> int:
        return headers.index(name) if name in headers else -1

    idx_condition = _col("CONDITION")
    idx_condition_queries = _col("CONDITION_QUERIES")
    idx_description = _col("DESCRIPTION")

    if idx_condition < 0:
        logging.info("未找到 CONDITION 列，将从 expandLogic 自动提取")
    if idx_condition_queries < 0:
        logging.info("未找到 CONDITION_QUERIES 列，condition_queries 将为空列表")
    if idx_description < 0:
        logging.info("未找到 DESCRIPTION 列，description 将留空")

    def _cell(row, idx: int) -> str:
        return str(row[idx]).strip() if idx >= 0 and row[idx] is not None else ""

    items = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        scene_key = _cell(row, idx_key)
        content_str = _cell(row, idx_content)
        if not scene_key and not content_str:
            continue
        item = convert_row(
            scene_key,
            content_str,
            _cell(row, idx_condition),
            _cell(row, idx_condition_queries),
            _cell(row, idx_description),
            ID_START + len(items),
        )
        if item:
            items.append(item)

    has_cond = sum(1 for it in items if it["condition"])
    has_cq = sum(1 for it in items if it["condition_queries"])
    has_desc = sum(1 for it in items if it["description"])

    logging.info("转换完成，共 %d 条 [评估项]", len(items))
    logging.info("  有 condition: %d", has_cond)
    logging.info("  有 condition_queries: %d", has_cq)
    logging.info("  有 description: %d", has_desc)
    logging.info("  id 范围: %s ~ %s", make_short_id(ID_START), make_short_id(ID_START + len(items) - 1))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    logging.info("→ %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
