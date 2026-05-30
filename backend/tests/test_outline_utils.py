"""
test_outline_utils.py — 独立测试大纲三种视图的转化函数。

运行：
    cd backend
    python tests/test_outline_utils.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "skills", "_lib"))
from outline_utils import to_markdown, to_yaml, from_yaml, to_clean_json, VIRTUAL_ROOT_ID

# ── 测试数据（模拟 subtree.build_subtree 的输出）──────────────────

MOCK_TREE = {
    "id": "L1_001",
    "name": "传送网络专项分析",
    "level": 1,
    "description": "对全省传送网络进行综合评估",
    "keywords": ["OTN", "传送", "覆盖"],   # 内部字段，to_clean_json 应去掉
    "score": 0.95,                          # 内部字段，to_clean_json 应去掉
    "children": [
        {
            "id": "L2_003",
            "name": "OTN覆盖评估",
            "level": 2,
            "description": "评估OTN节点的地理覆盖情况",
            "children": [
                {
                    "id": "L3_011",
                    "name": "覆盖率指标",
                    "level": 3,
                    "description": "各地市OTN覆盖比例",
                    "children": [],
                },
                {
                    "id": "L3_012",
                    "name": "区域覆盖分析",
                    "level": 3,
                    "description": "",
                    "children": [],
                },
                {
                    "id": "L3_013",
                    "name": "盲区识别",
                    "level": 3,
                    "description": "定位未被OTN覆盖的区域",
                    "children": [],
                },
            ],
        },
        {
            "id": "L2_004",
            "name": "设备利用率",
            "level": 2,
            "description": "评估关键设备的负载水平",
            "condition": "${number(\"端口占用率\")>0}",
            "condition_queries": ["端口占用率"],
            "children": [
                {
                    "id": "L3_014",
                    "name": "端口占用率",
                    "level": 3,
                    "description": "",
                    "children": [],
                },
                {
                    "id": "L5_015",
                    "name": "流量峰值分析",
                    "level": 5,
                    "description": "",
                    "exec_sql": "SELECT * FROM traffic",
                    "apiName": "getTraffic",
                    "children": [],
                },
            ],
        },
    ],
}


# ── 测试 ──────────────────────────────────────────────────────────

def test_to_markdown():
    result = to_markdown(MOCK_TREE)
    print("=" * 60)
    print("【to_markdown — 用户视图】")
    print("=" * 60)
    print(result)

    assert "# 传送网络专项分析" in result
    assert "## OTN覆盖评估" in result
    assert "### 覆盖率指标" in result
    assert "L1_001" not in result, "用户视图不应出现 id"
    print("✓ PASS\n")


def test_to_yaml():
    result = to_yaml(MOCK_TREE)
    print("=" * 60)
    print("【to_yaml — LLM 上下文视图】")
    print("=" * 60)
    print(result)

    assert "L1_001" in result
    assert "L2_003" in result
    assert "L3_011" in result
    assert "盲区识别" in result
    # level 和 SQL 字段不应出现
    assert "level:" not in result, "level 不应出现在 YAML 视图"
    assert "exec_sql" not in result, "exec_sql 不应出现在 YAML 视图"
    assert "apiName" not in result, "apiName 不应出现在 YAML 视图"
    # 空 description 不应出现
    assert "区域覆盖分析" in result
    # condition 和 condition_queries 应出现
    assert "condition:" in result
    assert "condition_queries:" in result
    print("✓ PASS\n")


def test_from_yaml_roundtrip():
    print("=" * 60)
    print("【from_yaml — YAML 逆向解析 roundtrip】")
    print("=" * 60)

    yaml_text = to_yaml(MOCK_TREE)
    recovered = from_yaml(yaml_text)

    assert recovered is not None, "from_yaml 不应返回 None"
    assert recovered["id"] == "L1_001"
    assert recovered["name"] == "传送网络专项分析"
    assert recovered["level"] == 1
    assert len(recovered["children"]) == 2

    l2 = recovered["children"][1]
    assert l2["id"] == "L2_004"
    assert l2["level"] == 2
    assert l2["condition"] != ""
    assert l2["condition_queries"] == ["端口占用率"]

    l5 = l2["children"][1]
    assert l5["id"] == "L5_015"
    assert l5["level"] == 5

    print(f"  根节点: {recovered['id']} level={recovered['level']}")
    print(f"  L2_004 condition_queries={l2['condition_queries']}")
    print(f"  L5_015 level={l5['level']}")
    print("✓ PASS\n")


def test_from_yaml_multi_root():
    """多个顶层节点应包裹虚拟根节点。"""
    yaml_text = """
- id: L1_001
  name: 场景A
- id: L1_002
  name: 场景B
"""
    result = from_yaml(yaml_text)
    assert result is not None
    assert result["id"] == VIRTUAL_ROOT_ID
    assert len(result["children"]) == 2
    assert result["children"][0]["level"] == 1
    assert result["children"][1]["level"] == 1
    print("【from_yaml multi-root】虚拟根包裹 ✓\n")


def test_to_clean_json():
    result = to_clean_json(MOCK_TREE)
    print("=" * 60)
    print("【to_clean_json — 程序视图】")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    assert "keywords" not in result, "keywords 应被去除"
    assert "score" not in result,    "score 应被去除"
    assert result["id"] == "L1_001"
    assert len(result["children"]) == 2
    print("✓ PASS\n")


def test_yaml_omits_virtual_root():
    """虚拟根节点应被跳过，直接输出子节点列表。"""
    wrapped = {
        "id": VIRTUAL_ROOT_ID, "name": "", "level": 0, "description": "",
        "children": [
            {"id": "L1_001", "name": "场景A", "level": 1, "description": "", "children": []},
        ],
    }
    result = to_yaml(wrapped)
    assert VIRTUAL_ROOT_ID not in result, "YAML 不应出现虚拟根 id"
    assert "L1_001" in result
    print("【to_yaml 跳过虚拟根节点】✓\n")


if __name__ == "__main__":
    test_to_markdown()
    test_to_yaml()
    test_from_yaml_roundtrip()
    test_from_yaml_multi_root()
    test_to_clean_json()
    test_yaml_omits_virtual_root()
    print("All tests passed.")
