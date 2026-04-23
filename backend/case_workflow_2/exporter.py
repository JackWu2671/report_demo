"""
exporter.py — Step 10: 将大纲树导出为干净的 JSON 结构。

输入: outline_tree dict（来自 subtree.build_subtree 或 patcher.apply_patch）
输出: 干净的 JSON dict，供下游系统执行 L5 查询

保留字段: id, name, level, description, params（有参数时才出现）, children
去除字段: keywords, score 等内部/检索字段
"""


def export_json(outline_tree: dict) -> dict:
    """
    将大纲树导出为干净的 JSON 结构。

    保留: id, name, level, description, params, children
    去除: keywords, score 等检索/内部字段

    Args:
        outline_tree: build_subtree() 或 apply_patch() 返回的树 dict

    Returns:
        可直接序列化的干净 dict，params 仅在节点有参数标注时出现
    """
    _KEEP = {"id", "name", "level", "description", "params"}
    node = {k: v for k, v in outline_tree.items() if k in _KEEP}
    node["children"] = [export_json(c) for c in outline_tree.get("children", [])]
    return node
