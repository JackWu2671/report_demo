"""
definitions.py — OpenAI tool schemas for agent2.

Six tools, in the order the agent should try them for a new outline request:
  1. match_outline_template    — vector search + LLM judge on pre-built templates
  2. search_outline_templates  — vector search only, returns top-N candidates (no LLM)
  3. load_template_outline     — load full outline for a specific template by scene_name
  4. init_outline_from_graph   — FAISS → anchor → subtree (when no template matches)
  5. search_graph_tree         — FAISS search KB → build ancestor paths → return tree
  6. modify_outline            — patch current outline via natural-language instruction
"""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "match_outline_template",
            "description": (
                "在预制大纲模板库中检索并由 LLM 判断最匹配的模板，决策是否可复用。"
                "status=pending_confirm 表示找到可用模板；status=not_found 表示无匹配，需改用 init_outline_from_graph。"
                "用户提出新的分析需求时，优先调用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的分析需求描述，原文传入",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_outline_templates",
            "description": (
                "仅做向量检索，返回模板库中与需求最相似的 top-N 候选模板列表（不经 LLM 判断）。"
                "用于用户想直接浏览有哪些可用模板时调用，或在 match_outline_template 结果存疑时补充参考。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的分析需求描述，原文传入",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回候选数量，默认 5",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_template_outline",
            "description": (
                "按模板名称直接加载指定模板的完整大纲内容，跳过向量检索和 LLM 判断。"
                "当用户已从候选列表中看到某个模板名称，想查看其具体大纲结构时调用。"
                "scene_name 必须与 search_outline_templates 返回的候选名称完全一致。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_name": {
                        "type": "string",
                        "description": "模板场景名称，与候选列表中的 scene_name 完全一致",
                    },
                },
                "required": ["scene_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "init_outline_from_graph",
            "description": (
                "从知识库实时检索：FAISS向量检索 → 锚节点选择 → 子树展开 → 初始修正，生成报告大纲。"
                "仅在 match_outline_template 返回 not_found 后调用，或用户明确要求重新生成。"
                "status=not_found 表示知识库无相关内容，应告知用户系统暂不支持该场景。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的分析需求描述，原文传入",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_graph_tree",
            "description": (
                "从知识图谱中检索与问题相关的节点，返回带祖先路径的树状结构（含节点 id、描述、FAISS 命中分数）。"
                "用于用户想直接浏览知识库中有哪些相关节点时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用户的分析需求描述，原文传入",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_outline",
            "description": (
                "对当前报告大纲执行修改：删除章节、聚焦方向、设置参数阈值等。"
                "仅当已存在大纲（之前成功调用过 match_outline_template、load_template_outline 或 init_outline_from_graph）时可用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "用户的自然语言修改指令，原文传入",
                    }
                },
                "required": ["instruction"],
            },
        },
    },
]
