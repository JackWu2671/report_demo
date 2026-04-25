"""
test_outline_utils.py — 独立测试大纲三种视图的转化函数。

运行：
    cd backend
    python tests/test_outline_utils.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from outline_utils import to_markdown, to_markdown_with_ids, to_clean_json

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
                    "params": {
                        "threshold": {"value": "85", "unit": "%"}
                    },
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
            "children": [
                {
                    "id": "L3_014",
                    "name": "端口占用率",
                    "level": 3,
                    "description": "",
                    "children": [],
                },
                {
                    "id": "L3_015",
                    "name": "流量峰值分析",
                    "level": 3,
                    "description": "分析高峰时段流量分布",
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
    assert "> 参数设置 — threshold: 85%" in result
    print("✓ PASS\n")


def test_to_markdown_with_ids():
    result = to_markdown_with_ids(MOCK_TREE)
    print("=" * 60)
    print("【to_markdown_with_ids — LLM 上下文视图】")
    print("=" * 60)
    print(result)

    assert "[L1 L1_001]" in result
    assert "[L2 L2_003]" in result
    assert "[L3 L3_011]" in result
    assert "threshold=85%" in result
    assert "盲区识别" in result
    # 无 description 的节点不应有冒号
    lines = {l.strip() for l in result.splitlines()}
    assert any("区域覆盖分析" in l and "：" not in l for l in lines), \
        "无 description 节点不应出现冒号"
    print("✓ PASS\n")


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
    assert result["children"][0]["children"][0]["params"]["threshold"]["value"] == "85"
    print("✓ PASS\n")


def test_roundtrip_id_stability():
    """to_clean_json → to_markdown_with_ids 的 id 应该一致。"""
    clean = to_clean_json(MOCK_TREE)
    md_ids = to_markdown_with_ids(clean)
    assert "L3_011" in md_ids
    assert "L3_015" in md_ids
    print("【Roundtrip】clean_json → markdown_with_ids id 一致 ✓\n")


if __name__ == "__main__":
    test_to_markdown()
    test_to_markdown_with_ids()
    test_to_clean_json()
    test_roundtrip_id_stability()
    print("All tests passed.")
