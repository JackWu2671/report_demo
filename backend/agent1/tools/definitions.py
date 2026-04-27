"""
definitions.py — OpenAI tool schemas for agent1.

Three tools for the expert knowledge → template pipeline:
  1. analyze_expert_knowledge  — Steps 1-4: metadata + outline + new nodes
  2. modify_outline            — shared: patch current outline
  3. save_outline_template     — persist confirmed template
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "analyze_expert_knowledge",
            "description": (
                "分析专家输入的业务场景描述，从知识库检索相关节点，生成报告大纲。"
                "同时标注出大纲中知识库暂未收录的新概念节点（[new]节点）。"
                "专家提供场景描述后首先调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expert_text": {
                        "type": "string",
                        "description": "专家的业务场景描述，原文传入",
                    }
                },
                "required": ["expert_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_outline",
            "description": (
                "对当前大纲执行修改：删除章节、聚焦方向、调整结构等。"
                "仅当已存在大纲时可用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "专家的自然语言修改指令，原文传入",
                    }
                },
                "required": ["instruction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_outline_template",
            "description": (
                "将当前大纲保存为可复用模板。"
                "仅在专家明确确认（如说'保存'、'好的就这样'）时调用，不得主动触发。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
