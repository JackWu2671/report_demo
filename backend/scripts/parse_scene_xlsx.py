#!/usr/bin/env python3
"""
parse_scene_xlsx.py — 将场景类 xlsx 转换为 JSON

直接修改下方 ── 配置区 ── 里的三个变量，然后运行：
  python3 parse_scene_xlsx.py

字段说明:
  id          — 原始 UUID
  nodeId      — 短编号，LLM 大纲可见（L1_001 / L2_001 / L3_001）
  name        — 场景名称（同 SCENEKEY）
  level       — 由 LEVEL 变量指定
  description — 一句话描述
  detail      — 详细拓展逻辑（原字段名 expandLogic）
  keywords    — 关键词列表（原字段名 keyWords）
  sampleIssue — 示例提问
  condition   — 触发条件（原数据无此字段，默认空字符串）
  dimensions  — 子维度列表，每项只保留 id / name / rank
"""

import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_SCRIPT_DIR)
_KB_DIR = os.path.join(_BACKEND_DIR, "expert_knowledge")

# ── 配置区（按需增删）────────────────────────────────────────────────
CONFIGS = [
    {
        "level":       1,
        "id_prefix":   "L1",
        "id_start":    1,
        "input_file":  os.path.join(_KB_DIR, "场景.xlsx"),
        "output_file": os.path.join(_KB_DIR, "场景.json"),
    },
    {
        "level":       2,
        "id_prefix":   "L2",
        "id_start":    1,
        "input_file":  os.path.join(_KB_DIR, "子场景.xlsx"),
        "output_file": os.path.join(_KB_DIR, "子场景.json"),
    },
    {
        "level":       3,
        "id_prefix":   "L3",
        "id_start":    1,
        "input_file":  os.path.join(_KB_DIR, "评估维度.xlsx"),
        "output_file": os.path.join(_KB_DIR, "评估维度.json"),
    },
]
# ────────────────────────────────────────────────────────────────────


def parse_dimensions(raw) -> list[dict]:
    """只保留每个 dimension 的 id / name / rank。"""
    if not isinstance(raw, list):
        return []
    result = []
    for d in raw:
        result.append({
            "id":   d.get("id",   ""),
            "name": d.get("name", ""),
            "rank": d.get("rank", None),
        })
    # 按 rank 升序排列
    result.sort(key=lambda x: (x["rank"] is None, x["rank"]))
    return result


def parse_content(content_str: str) -> dict | None:
    """解析 CONTENT 列的 JSON 字符串，失败返回 None。"""
    try:
        return json.loads(content_str)
    except (json.JSONDecodeError, TypeError):
        return None


def make_short_id(prefix: str, index: int) -> str:
    return f"{prefix}_{index:03d}"


def convert_row(scene_key: str, content_str: str, level: str,
                id_prefix: str, index: int) -> dict | None:
    """将一行 Excel 数据转换为目标结构。"""
    obj = parse_content(content_str)
    if obj is None:
        print(f"[警告] SCENEKEY={scene_key!r} 的 CONTENT 解析失败，跳过", file=sys.stderr)
        return None

    return {
        "id":          obj.get("id", ""),
        "nodeId":      make_short_id(id_prefix, index),
        "name":        obj.get("name", scene_key),
        "level":       level,
        "description": obj.get("description", ""),
        "detail":      obj.get("expandLogic", ""),
        "keywords":    obj.get("keyWords", []),
        "sampleIssue": obj.get("sampleIssue", ""),
        "condition":   "",
        "dimensions":  parse_dimensions(obj.get("dimensions", [])),
    }


def process_one(cfg: dict) -> None:
    """处理单个 xlsx，转换后写入对应 json。"""
    level       = cfg["level"]
    input_file  = cfg["input_file"]
    output_file = cfg["output_file"]

    if not os.path.exists(input_file):
        print(f"[跳过] 文件不存在: {input_file}")
        return

    import openpyxl
    wb = openpyxl.load_workbook(input_file, data_only=True)
    ws = wb.active

    headers = [str(cell.value).strip().upper() if cell.value else "" for cell in ws[1]]
    try:
        idx_key     = headers.index("SCENEKEY")
        idx_content = headers.index("CONTENT")
    except ValueError:
        print(f"[错误] [{level}] 找不到必要列，实际表头: {headers}", file=sys.stderr)
        return

    id_prefix = cfg["id_prefix"]
    id_start  = cfg["id_start"]

    scenes = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        scene_key   = str(row[idx_key]).strip()    if row[idx_key]     else ""
        content_str = str(row[idx_content]).strip() if row[idx_content] else ""
        if not scene_key and not content_str:
            continue
        item = convert_row(scene_key, content_str, level,
                           id_prefix, id_start + len(scenes))
        if item:
            scenes.append(item)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scenes, f, ensure_ascii=False, indent=2)

    print(f"[{level}] 完成，共 {len(scenes)} 条 → {output_file}")


def main():
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[错误] 请先安装 openpyxl: pip install openpyxl", file=sys.stderr)
        sys.exit(1)

    for cfg in CONFIGS:
        process_one(cfg)


if __name__ == "__main__":
    main()
