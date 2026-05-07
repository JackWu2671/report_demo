# Agent2 设计文档

> 面向零基础开发人员
>
> 本文档覆盖 `backend/agent2/` 的设计思路、每个模块的职责，以及关键设计决策背后的原因。

---

## 0. 这个项目是什么

**report_demo** 是一个 AI 辅助报告生成系统，帮助用户快速生成结构化的分析报告大纲。

整个系统分两个角色：

- **专家**：把自己的业务知识输入给 Agent1，Agent1 提炼成大纲模板并保存到知识库。
- **普通用户**：告诉 Agent2 想分析什么，Agent2 从知识库检索模板或实时生成大纲，用户可以对话式地修改，满意后导出报告。

技术栈：Python + FastAPI 后端，大模型通过 OpenAI 兼容接口调用（当前接入 Qwen），向量检索用 FAISS。

> **商用迁移计划**：当前 Python 实现为原型验证版本。后续将以本文档的设计为基础，迁移到 **Java** 进行商业化落地，接口协议（SSE 事件格式、工具定义）保持不变。

---

## 1. 为什么用 Agent

**Workflow（旧做法）** 把流程写死在代码里，无法多轮修改，没有记忆。

**Agent（现做法）** 大模型自己决定调哪个工具：

```
用户说"帮我分析 fgOTN 部署"  →  大模型调 match_outline_template
用户说"没有合适的，重新生成"  →  大模型调 search_graph_tree，选锚节点，调 build_outline_from_anchor
用户说"把第二节删掉"          →  大模型调 modify_outline
```

流程不是 Python if/else 写死的，是大模型根据对话上下文推理出来的。

---

## 2. 目录结构

```
backend/
├── memory/
│   └── store.py              ← 共享状态管理（AgentMemory 基类）
│
├── tools/                    ← 共享工具实现（纯业务逻辑，无 LLM 调用）
│   ├── analyze_expert.py         agent1 用：提取 + 检索 + 生成大纲 + 解析新节点
│   ├── search_graph_tree.py      FAISS 检索知识库，返回带祖先路径的树状结构
│   ├── build_outline_from_anchor.py  纯 Python：以锚节点为根展开子树，生成初始大纲
│   ├── search_template.py        检索预制模板（含 LLM judge）
│   ├── modify_outline.py         纯 Python：按 ops 列表修改大纲
│   └── save_template.py          agent1 用：保存模板
│
├── agent1/                   ← 专家知识沉淀 Agent
│   └── ...
│
└── agent2/                   ← 大纲对话生成 Agent
    ├── agent.py              主循环
    ├── prompt.txt            系统提示词（含工具调用 SOP）
    ├── agent_test.py         命令行交互测试
    ├── DESIGN.md             本文档
    └── tools/
        ├── definitions.py    工具 JSON Schema（大模型看到的工具说明）
        ├── handlers.py       工具调度表 + memory 写入
        └── __init__.py
```

`backend/tools/` 是工具的**实现**，`agent2/tools/` 是工具的**定义和调度**，两者分离，实现可跨 agent 复用。

---

## 3. 核心概念：工具调用（Function Calling）

大模型原生只能输出文字。**Function Calling** 是 OpenAI 定义的一种协议，让大模型可以表达"我想调用某个函数"。

### 3.1 如何告诉大模型有哪些工具

调 API 时多传一个 `tools` 参数（JSON Schema 格式的工具说明）：

```python
# agent2/agent.py
await llm._client.chat.completions.create(
    model=llm.default_model,
    messages=messages,
    tools=TOOLS,          # ← 工具说明书，来自 agent2/tools/definitions.py
    tool_choice="auto",   # ← 让大模型自己决定要不要调
)
```

`TOOLS` 里每个工具长这样（`agent2/tools/definitions.py`）：

```python
{
    "type": "function",
    "function": {
        "name": "build_outline_from_anchor",
        "description": "以指定节点为根，从知识图谱展开子树，生成初始报告大纲...",
        "parameters": {
            "type": "object",
            "properties": {
                "anchor_id": {
                    "type": "string",
                    "description": "锚节点 id，从 search_graph_tree 返回的树中选取，如 'L4_001'"
                }
            },
            "required": ["anchor_id"]
        }
    }
}
```

### 3.2 大模型输出什么

大模型判断"需要调工具"时，输出的**不是文字**，而是：

```json
{
  "role": "assistant",
  "finish_reason": "tool_calls",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "build_outline_from_anchor",
        "arguments": "{\"anchor_id\": \"L4_001\"}"
      }
    }
  ]
}
```

关键字段：
- `finish_reason = "tool_calls"` — 大模型没说完，要调工具
- `tool_calls[].function.name` — 调哪个工具
- `tool_calls[].function.arguments` — 参数，是一个 **JSON 字符串**（需要 `json.loads()` 解析）
- `tool_calls[].id` — 这次调用的唯一 ID，执行完要原样带回

大模型判断"直接回答用户"时，输出普通文字：

```json
{
  "role": "assistant",
  "content": "已根据您的需求生成大纲，请查看。",
  "finish_reason": "stop"
}
```

### 3.3 prompt.txt 和 tools 参数的区别

| | 作用 |
|---|---|
| `tools` 参数（JSON Schema） | 告诉大模型工具的**结构**：名称、参数类型、required 字段 |
| `prompt.txt`（自然语言） | 告诉大模型工具的**语义**：什么场景用、什么时候不用、先后顺序 |

`tools` 控制格式，`prompt.txt` 控制决策。

---

## 4. Agent2 的工具

### 4.1 生成大纲：两步走

生成大纲没有一步到位的工具，而是拆成两步：

**第一步：`search_graph_tree(question)`**

FAISS 向量检索知识库，返回与问题相关的节点，以带祖先路径的树状结构展示：

```
[L1 L1_001] 政企业务
  [L2 L2_001] fgOTN 升级
    [L3 L3_001] 传送网络覆盖分析 (score=0.91)
    [L4 L4_001] 高价值行业覆盖缺口 (score=0.87)
```

返回值是文本形态的树（`tree_text`），大模型可以读懂，从中选出最合适的锚节点。

**第二步：`build_outline_from_anchor(anchor_id)`**

纯 Python，不调大模型。以选定节点为根，把知识库中该节点下的**所有子节点递归展开**，生成初始大纲。

**为什么拆成两步？**

- 锚节点的选择需要"理解用户意图"，这是大模型擅长的事
- 展开子树是纯数据操作，不需要大模型参与，更快、更稳定
- 大模型在 `search_graph_tree` 的结果里看到完整的节点信息（id、名称、层级），选完后直接传 `anchor_id`，比让一个内置 LLM 做选择更透明可控

### 4.2 锚节点选择规则（写在 prompt.txt 里）

- 选与用户需求**最直接对应**的节点，而不是它的祖先节点
- 优先选 L3～L5 层节点，避免选 L1/L2 等过于宽泛的顶层节点
- `build_outline_from_anchor` 会把锚节点下**所有子节点全部展开**，生成后需要对照用户需求判断是否需要裁剪，如需要则立即调用 `modify_outline`

### 4.3 模板检索：match_outline_template

`search_graph_tree` → `build_outline_from_anchor` 是从零生成大纲的路径，但如果知识库里已经有对应的预制模板，直接复用更快。

`match_outline_template(question)` 内部做两件事：
1. FAISS 检索候选模板
2. 内置一个 LLM judge 判断候选是否真的匹配用户需求

返回 `status=pending_confirm`（找到可用模板）或 `status=not_found`（无匹配，走 KB 生成路径）。

**为什么用单独的 judge LLM 而不是让 Agent LLM 判断？**

让 Agent LLM 判断需要把所有候选模板全文塞进对话，token 消耗大；judge LLM 专门做一件事，输入精简、可靠性更高。

### 4.4 修改大纲：modify_outline

纯 Python，不调大模型。接收一个 `ops` 列表，按顺序对 `outline_tree` 执行操作：

| op | 参数 | 含义 |
|---|---|---|
| `add_node` | `node_id, parent_id` | 从知识库取该节点（递归展开子树）插入指定父节点下 |
| `delete_node` | `node_id` | 删除节点及其全部子树 |
| `modify_node_name` | `node_id, value` | 修改节点名称 |
| `modify_node_description` | `node_id, value` | 修改节点描述（也用于写入阈值、范围等说明） |
| `keep_only_node` | `node_id` | 保留该节点，同级其他节点全部删除 |

**为什么 ops 由 Agent LLM 直接构造，而不是让 modify_outline 内部调 LLM？**

Agent LLM 本身就看着当前大纲（system prompt 末尾有 `md_with_ids`），已经知道节点 ID 和层级关系，直接输出 ops 最自然。如果在 `modify_outline` 内部再调一次 LLM，相当于重复推理，多一次延迟、多一层出错点。

---

## 5. Agent 主循环（agent.py）

```python
async def chat_stream(self, user_message: str):
    self.memory.add_message({"role": "user", "content": user_message})

    for _round in range(_MAX_TOOL_ROUNDS):   # 最多 6 轮，防止死循环
        response = await self._call_llm()
        choice = response.choices[0]
        msg = choice.message

        self.memory.add_message(msg.model_dump(exclude_none=True))

        if choice.finish_reason == "tool_calls":
            for tc in msg.tool_calls:
                yield {"type": "step", "name": tc.function.name, "status": "running"}

                result_dict, llm_str = await self._execute_tool(tc)

                # 工具产出了大纲 → 立刻推送给前端，不等 LLM
                if result_dict.get("outline_tree"):
                    yield {"type": "outline", "markdown": result_dict["markdown"]}

                yield {"type": "step", "name": tc.function.name, "status": "done"}

                self.memory.add_message({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": llm_str,   # 工具执行结果的精简摘要
                })
            continue  # 回到循环顶部，让大模型继续推理

        # finish_reason == "stop"：大模型直接回答
        if msg.content:
            yield {"type": "text", "chunk": msg.content}
        yield {"type": "done", "seconds": round(time.time() - t0, 1)}
        return
```

这个"调 LLM → 执行工具 → 把结果喂回 LLM → 再调"的循环就是 **ReAct 模式**（Reason + Act）。

---

## 6. 工具调度（handlers.py）

`_execute_tool()` 按工具名查 `HANDLERS` 字典，找到对应的 handler 函数执行：

```python
HANDLERS = {
    "match_outline_template":    handle_match_outline_template,
    "search_outline_templates":  handle_search_outline_templates,
    "load_template_outline":     handle_load_template_outline,
    "search_graph_tree":         handle_search_graph_tree,
    "build_outline_from_anchor": handle_build_outline_from_anchor,
    "modify_outline":            handle_modify_outline,
}
```

每个 handler 做三件事：

```python
async def handle_build_outline_from_anchor(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    # 1. 调工具实现
    result = await build_outline_from_anchor(args["anchor_id"])

    # 2. 成功就更新 memory
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])

    # 3. 返回两份数据
    llm_str = f"[build_outline_from_anchor] status={result['status']}\n\n{result.get('md_with_ids', '')}"
    return result, llm_str
```

**为什么返回两份数据？**

- `result`（完整）给 agent.py 用，从中取 `outline_tree` 和 `markdown` 推送前端
- `llm_str`（精简）放进对话历史给大模型看，只含带 ID 的紧凑大纲，不含大段 Markdown，节省 token

---

## 7. 大纲的三种表示

大纲数据以三种形态存在，服务不同消费者：

| 字段 | 格式示例 | 谁用 |
|------|---------|------|
| `outline_tree` | JSON dict（含 id/name/level/description/children） | 代码逻辑（modify_outline 的输入输出） |
| `markdown` | `# fgOTN部署\n## 传送网络覆盖分析` | 前端渲染给用户看 |
| `md_with_ids` | `[L2 L2_001] fgOTN部署\n  [L3 L3_001] ...` | LLM 上下文（可精确引用节点 ID） |

三者由 `backend/outline_utils.py` 从同一个 `outline_tree` 派生：

```
outline_tree  →  to_markdown()          →  markdown
outline_tree  →  to_markdown_with_ids() →  md_with_ids
```

**为什么 LLM 不能直接看普通 Markdown？**

修改大纲时，LLM 需要精确引用节点（如 `"delete_node L3_002"`），普通 Markdown 没有节点 ID，LLM 只能靠名称描述，容易定位错。

---

## 8. Memory（状态管理）

```python
class AgentMemory:
    def __init__(self):
        self.outline_tree: dict = {}   # 程序用的 JSON 树
        self.markdown: str = ""        # 用户看的 Markdown
        self.md_with_ids: str = ""     # LLM 看的带 ID 版本
        self.kb_tree_text: str = ""    # search_graph_tree 返回的原始树文本
        self._history: list = []       # 对话历史（不包含大纲）
```

### 大纲不进对话历史

大纲会被修改，历史里存的是旧版本，LLM 会被旧版本误导；另外大纲可能很长，每轮都放历史里浪费 token。

### 大纲如何注入 LLM 上下文

每次调 LLM 前，`build_messages()` 把最新的 `md_with_ids` 拼进 **第一条 system 消息** 里：

```python
def build_messages(self, system_prompt: str) -> list[dict]:
    content = system_prompt
    if self.has_outline:
        content += f"\n\n## 当前大纲（可通过节点ID引用）\n\n{self.md_with_ids}"
    return [{"role": "system", "content": content}, *self._history]
```

**为什么不追加一条新的 system 消息？**

Qwen 等模型要求 system 消息只能出现在最开头，追加到末尾会报错：
`400 Bad Request: System message must be at the beginning`

### kb_tree_text 不进 system prompt

`search_graph_tree` 的结果（`kb_tree_text`）存在 memory 里，但不注入 system prompt。它已经出现在工具调用的历史消息里，LLM 可以直接从对话历史里引用，不需要重复。

---

## 9. 事件协议（SSE Events）

`chat_stream()` 是一个 async generator，每发生一件事就 `yield` 一个字典，由 `api_server.py` 序列化成 SSE 推给前端：

```python
{"type": "step",    "name": "search_graph_tree",         "status": "running"}
{"type": "step",    "name": "search_graph_tree",         "status": "done"}
{"type": "step",    "name": "build_outline_from_anchor", "status": "running"}
{"type": "step",    "name": "build_outline_from_anchor", "status": "done"}
{"type": "outline", "markdown": "# fgOTN部署\n## ..."}   # 工具完成后立刻推
{"type": "step",    "name": "modify_outline",            "status": "running"}
{"type": "step",    "name": "modify_outline",            "status": "done"}
{"type": "outline", "markdown": "# fgOTN部署\n## ..."}   # 修改后再推一次
{"type": "text",    "chunk": "已生成大纲，请查看。"}
{"type": "done",    "seconds": 4.1}
{"type": "error",   "message": "工具执行失败: ..."}
```

**为什么 outline 走独立事件，不让 LLM 逐字输出？**

大纲是工具计算出来的**已有数据**，没有理由让 LLM 再"重新打一遍"。`outline` 事件在工具执行完后立刻推送，前端瞬间渲染；如果让 LLM 流式输出 2000 字大纲，用户要等几十秒。

LLM 的文字职责只有一件事：**用 1-2 句话说明刚才做了什么**。

---

## 10. 一次完整对话的消息流

以「从知识库生成大纲 → 修改大纲」两轮为例：

```
第一轮：用户说"分析 fgOTN 高价值行业覆盖"

  LLM 决策：match_outline_template(question="...")
  → status=not_found，无预制模板

  LLM 决策：search_graph_tree(question="...")
  → 返回树文本，memory.kb_tree_text 更新，树文本进入 tool 消息历史

  LLM 决策：build_outline_from_anchor(anchor_id="L4_001")
  → 纯 Python 展开子树，memory.set_outline(...) 更新
  → agent yield: outline event  ← 前端立刻渲染

  LLM 检查大纲与用户需求 → 发现有两个无关节点
  LLM 决策：modify_outline(ops=[{op: "delete_node", node_id: "L5_003"}, ...])
  → 纯 Python 修改，memory.set_outline(...) 更新
  → agent yield: outline event  ← 前端渲染修改后的大纲

  LLM 输出文字："已根据您的需求生成大纲并裁剪无关节点，共 3 个分析维度。"

────────────────────────────────────────────────────

第二轮：用户说"把高价值行业覆盖缺口改成重点行业覆盖缺口"

  messages 传给 LLM（system 里含最新 md_with_ids）:
    [system: prompt + 最新 md_with_ids]
    [user/assistant/tool: 第一轮的完整历史]
    [user: "把高价值行业覆盖缺口改成重点行业覆盖缺口"]

  LLM 直接从 system prompt 里的大纲找到节点 ID
  LLM 决策：modify_outline(ops=[{op: "modify_node_name", node_id: "L4_001", value: "重点行业覆盖缺口"}])
  → 纯 Python 修改
  → agent yield: outline event  ← 前端渲染

  LLM 输出文字："已将节点名称修改为"重点行业覆盖缺口"。"
```

---

## 11. 设计决策速查

| 决策 | 原因 |
|------|------|
| 大纲不进对话历史，通过 memory 注入 system prompt | 避免旧版本误导 LLM，节省 token |
| outline 走独立事件通道 | 大纲是计算结果，无需 LLM 逐字输出，前端可瞬间渲染 |
| 生成大纲拆成 search_graph_tree + build_outline_from_anchor 两步 | 锚点选择靠 Agent LLM（理解意图），子树展开纯 Python（快且稳定） |
| build_outline_from_anchor 后接 modify_outline 裁剪 | 展开工具不做筛选，裁剪交给 Agent LLM 按需判断 |
| modify_outline 纯 Python，ops 由 Agent LLM 构造 | Agent LLM 已看到当前大纲，直接输出 ops 无需二次推理 |
| match_outline_template 内置 judge LLM | 专注判断一件事比让 Agent LLM 兼职更可靠，输入也更精简 |
| handler 返回 (result, llm_str) 两份数据 | result 给 agent 处理，llm_str（精简版）给 LLM 历史，避免历史膨胀 |
| outline context 拼入 system prompt 而非追加 system 消息 | Qwen 不允许多条 system 消息 |
| kb_tree_text 不注入 system prompt | 已在工具历史消息里，重复注入浪费 token |
