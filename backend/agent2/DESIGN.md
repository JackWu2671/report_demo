# Agent 设计文档

> 面向零基础 Agent 开发人员
>
> 本文档覆盖 `backend/agent1/` 和 `backend/agent2/` 的设计思路、每个模块的职责，
> 以及关键设计决策背后的原因。

---

## 1. Workflow vs Agent：为什么要改

**Workflow（旧做法）** — 代码写死执行顺序：

```python
# 每次请求必定跑这 4 步，顺序固定，大模型只是被调用的工具
extraction = await extract_from_expert(expert_text)   # Step 1
hits       = await dual_search(extraction, ...)        # Step 2
outline_md = await generate_outline(expert_text, ...)  # Step 3
new_nodes  = parse_new_nodes(outline_md)               # Step 4
```

问题：流程由代码控制，无法多轮修改，每次请求从头跑，没有记忆。

**Agent（现做法）** — 大模型自己决定调哪个工具：

```
用户说"分析 fgOTN 部署情况"  →  大模型决定调 analyze_expert_knowledge
用户说"把第二节删掉"          →  大模型决定调 modify_outline
用户说"好，保存"              →  大模型决定调 save_outline_template
```

流程不是 Python if/else 写死的，是大模型根据对话上下文推理出来的。

---

## 2. 目录结构

```
backend/
├── memory/
│   └── store.py          ← 共享状态管理（AgentMemory 基类）
│
├── tools/                ← 共享工具实现（agent1 / agent2 都可以用）
│   ├── analyze_expert.py     Steps 1-4: 提取 + 检索 + 生成大纲 + 解析新节点
│   ├── generate_outline.py   从知识库生成大纲（agent2 用）
│   ├── search_template.py    检索预制模板（agent2 用）
│   ├── modify_outline.py     修改大纲（agent1 / agent2 共用）
│   └── save_template.py      保存模板（agent1 用）
│
├── agent1/               ← 专家知识沉淀 Agent
│   ├── agent.py              主循环
│   ├── memory.py             Agent1Memory（继承 AgentMemory，扩展专家字段）
│   ├── prompt.txt            系统提示词
│   ├── agent_test.py         命令行交互测试
│   └── tools/
│       ├── definitions.py    工具 JSON Schema（大模型看到的工具说明）
│       ├── handlers.py       工具调度表 + memory 写入
│       └── __init__.py
│
└── agent2/               ← 大纲对话生成 Agent
    ├── agent.py              主循环
    ├── prompt.txt            系统提示词
    ├── agent_test.py         命令行交互测试
    ├── DESIGN.md             本文档
    └── tools/
        ├── definitions.py    工具 JSON Schema
        ├── handlers.py       工具调度表 + memory 写入
        └── __init__.py
```

`backend/tools/` 里是工具的**实现**，`agentX/tools/` 里是工具的**定义和调度**，两者分离，实现可跨 agent 复用。

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
        "name": "generate_outline",
        "description": "从知识库实时检索并生成报告大纲...",   # ← 大模型靠这句理解用途
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "用户的分析需求，原文传入"
                }
            },
            "required": ["question"]
        }
    }
}
```

### 3.2 大模型输出什么

大模型判断"这轮需要调工具"时，输出的**不是文字**，而是：

```json
{
  "role": "assistant",
  "content": null,
  "finish_reason": "tool_calls",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "generate_outline",
        "arguments": "{\"question\": \"帮我分析 fgOTN 的部署情况\"}"
      }
    }
  ]
}
```

关键字段：
- `finish_reason = "tool_calls"` — 告诉我们大模型没说完，要调工具
- `tool_calls[].function.name` — 调哪个工具
- `tool_calls[].function.arguments` — 参数，是一个 **JSON 字符串**（需要 `json.loads()` 解析）
- `tool_calls[].id` — 这次调用的唯一 ID，执行完要原样带回

大模型判断"直接回答用户"时，输出普通文字：

```json
{
  "role": "assistant",
  "content": "已根据您的需求生成大纲，共发现 2 个新节点。",
  "finish_reason": "stop",
  "tool_calls": null
}
```

> 想亲眼看这两种输出：`cd backend && python tests/tool_test.py "你的问题"`

### 3.3 prompt.txt 和 tools 参数的区别

两者都在"告诉大模型工具的信息"，但各司其职：

| | 作用 |
|---|---|
| `tools` 参数（JSON Schema） | 告诉大模型工具的**结构**：名称、参数类型、required 字段，大模型按此格式输出 |
| `prompt.txt`（自然语言） | 告诉大模型工具的**语义**：什么场景用、什么时候不用、先后顺序 |

`tools` 控制格式，`prompt.txt` 控制决策。

---

## 4. Agent 主循环（agent.py）

两个 agent 的主循环结构完全一致，都在 `chat_stream()` 里：

```python
async def chat_stream(self, user_message: str):
    self.memory.add_message({"role": "user", "content": user_message})

    for _round in range(_MAX_TOOL_ROUNDS):   # 最多 6 轮，防止死循环
        response = await self._call_llm()
        choice = response.choices[0]
        msg = choice.message

        self.memory.add_message(msg.model_dump(exclude_none=True))  # 记录大模型回复

        if choice.finish_reason == "tool_calls":
            for tc in msg.tool_calls:
                yield {"type": "step", "name": tc.function.name, "status": "running"}

                result_dict, llm_str = await self._execute_tool(tc)  # 真正执行工具

                # 工具产出了大纲 → 立刻推送给前端，不等 LLM
                if result_dict.get("outline_tree"):
                    yield {"type": "outline", "markdown": result_dict["markdown"]}

                yield {"type": "step", "name": tc.function.name, "status": "done"}

                # 把执行结果告诉大模型，让它继续推理
                self.memory.add_message({
                    "role": "tool",
                    "tool_call_id": tc.id,   # 对应上面的 call_abc123
                    "content": llm_str,       # 工具执行结果的文字摘要
                })
            continue  # 回到循环顶部，让大模型再推理一轮

        # finish_reason == "stop"：大模型直接回答
        if msg.content:
            yield {"type": "text", "chunk": msg.content}
        yield {"type": "done", "seconds": round(time.time() - t0, 1)}
        return
```

这个"调 LLM → 执行工具 → 把结果喂回 LLM → 再调"的循环就是 **ReAct 模式**（Reason + Act）。

---

## 5. 工具调度（handlers.py）

`_execute_tool()` 按工具名查 `HANDLERS` 字典，找到对应的 handler 函数执行：

```python
# agent2/tools/handlers.py
HANDLERS = {
    "search_outline_template": handle_search_outline_template,
    "generate_outline":         handle_generate_outline,
    "modify_outline":           handle_modify_outline,
}
```

每个 handler 做三件事：

```python
async def handle_generate_outline(args: dict, memory: AgentMemory) -> tuple[dict, str]:
    # 1. 调工具实现（在 backend/tools/ 里）
    result = await generate_outline(args["question"])

    # 2. 成功就更新 memory
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])

    # 3. 返回两份数据
    #    result   → agent 检查是否有大纲，决定是否 yield outline 事件
    #    llm_str  → 喂给大模型历史的精简版（只含 md_with_ids，不含完整 Markdown）
    llm_str = f"[generate_outline] status={result['status']}\n\n{result.get('md_with_ids', '')}"
    return result, llm_str
```

**为什么返回两份数据？**

- `result`（完整）给 agent.py 用，agent 从里面取 `outline_tree` 和 `markdown` 推送前端
- `llm_str`（精简）放进对话历史给大模型看，只含带 ID 的紧凑大纲，不含大段 Markdown，节省 token

---

## 6. 大纲的三种表示

大纲数据在系统中以三种形态存在，服务不同消费者：

| 字段 | 格式示例 | 谁用 |
|------|---------|------|
| `outline_tree` | JSON dict（有 id/name/level/children） | 代码逻辑（修改、保存） |
| `markdown` | `# fgOTN部署\n## 传送网络覆盖分析` | 前端渲染给用户看 |
| `md_with_ids` | `[L2 L2_001] fgOTN部署\n  [L3 L3_001] ...` | LLM 上下文（可精确引用节点 ID） |

三者由 `backend/outline_utils.py` 从同一个 `outline_tree` 派生：

```
outline_tree  →  to_markdown()          →  markdown
outline_tree  →  to_markdown_with_ids() →  md_with_ids
```

**为什么 LLM 不能直接看普通 Markdown？**
修改大纲时，LLM 需要精确引用节点（"把 `L3_002` 删掉"），普通 Markdown 没有节点 ID，LLM 只能用名称描述，容易定位错。

---

## 7. Memory（状态管理）

`backend/memory/store.py` 定义 `AgentMemory` 基类，agent1 的 `Agent1Memory` 在此基础上扩展了专家输入相关字段。

### 7.1 大纲不进对话历史

```python
class AgentMemory:
    def __init__(self):
        self.outline_tree: dict = {}   # 程序用的 JSON 树
        self.markdown: str = ""        # 用户看的 Markdown
        self.md_with_ids: str = ""     # LLM 看的带 ID 版本
        self._history: list = []       # 对话历史（不包含大纲）
```

大纲单独存，不放进 `_history`，原因：
- 大纲会被修改，历史里存的是旧版本，LLM 会被旧版本误导
- 大纲可能很长，每轮都放历史里浪费 token

### 7.2 大纲如何注入 LLM 上下文

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

---

## 8. 事件协议（SSE Events）

`chat_stream()` 是一个 async generator，每发生一件事就 `yield` 一个字典，由 `api_server.py` 序列化成 SSE 推给前端：

```python
{"type": "step",      "name": "generate_outline", "status": "running"}
{"type": "step",      "name": "generate_outline", "status": "done"}
{"type": "outline",   "markdown": "# fgOTN部署\n## ..."}   # 工具完成后立刻推
{"type": "text",      "chunk": "已生成大纲，共 3 个分析维度。"}
{"type": "done",      "seconds": 4.1}
{"type": "error",     "message": "工具执行失败: ..."}

# Agent1 额外有：
{"type": "new_nodes", "nodes": [{"name": "高价值行业覆盖缺口", "level": 4, ...}]}
{"type": "saved",     "scene_name": "fgOTN部署", "path": "templates/fgOTN部署.json"}
```

**为什么 outline 走独立事件，不让 LLM 逐字输出？**
大纲是工具计算出来的**已有数据**，没有理由让 LLM 再"重新打一遍"。`outline` 事件在工具执行完后立刻推送，前端瞬间渲染；如果让 LLM 流式输出 2000 字大纲，用户要等几十秒看 LLM 逐字吐出来。

LLM 的文字职责只有一件事：**用 1-2 句话说明刚才做了什么**。

---

## 9. 两个 Agent 的对比

| | Agent1（专家知识沉淀） | Agent2（大纲对话生成） |
|---|---|---|
| 使用者 | 专家，输入业务场景描述 | 普通用户，提出分析需求 |
| 核心工具 | `analyze_expert_knowledge`（Steps 1-4 打包）| `search_outline_template` + `generate_outline` |
| 额外工具 | `save_outline_template` | — |
| Memory 类 | `Agent1Memory`（扩展了 extraction / new_nodes） | `AgentMemory`（基类） |
| 额外事件 | `new_nodes`、`saved` | — |
| 输出 | 大纲模板保存到 `templates/` | 大纲用于当轮展示 |

---

## 10. 一次完整对话的消息流

以 Agent2「生成大纲 → 修改大纲」两轮为例：

```
第一轮：用户说"分析 fgOTN 部署"

  messages 传给 LLM:
    [system: prompt + (无大纲)]
    [user: "分析 fgOTN 部署"]

  LLM 输出:  finish_reason=tool_calls
    tool_calls: search_outline_template(question="分析 fgOTN 部署")

  handler 执行 → memory.set_outline(tree, markdown, md_with_ids)
  agent yield: outline event  ← 前端立刻渲染大纲

  messages 追加 tool 结果后再调 LLM:
    [system: prompt + md_with_ids]
    [user: ...]
    [assistant: tool_call]
    [tool: "status=found, [L2 L2_001] fgOTN部署 ..."]

  LLM 输出:  finish_reason=stop
    content: "已找到匹配大纲，共 2 个分析维度。"

───────────────────────────────────────────────────────

第二轮：用户说"删掉传送网络容量分析"

  messages 传给 LLM（system 里已含最新 md_with_ids）:
    [system: prompt + 最新 md_with_ids]
    [user: "分析 fgOTN 部署"]
    [assistant: tool_call]
    [tool: "status=found ..."]
    [assistant: "已找到匹配大纲..."]
    [user: "删掉传送网络容量分析"]

  LLM 输出:  finish_reason=tool_calls
    tool_calls: modify_outline(instruction="删掉传送网络容量分析")

  handler 执行 → memory.set_outline(new_tree, ...)
  agent yield: outline event  ← 前端渲染修改后的大纲

  LLM 输出:  "已删除传送网络容量分析，大纲现有 1 个分析维度。"
```

---

## 11. 设计决策速查

| 决策 | 原因 |
|------|------|
| 大纲不进对话历史，通过 memory 注入 system prompt | 避免旧版本误导 LLM，节省 token |
| outline 走独立事件通道 | 大纲是计算结果，无需 LLM 逐字输出，前端可瞬间渲染 |
| `search_template` 内部封装一个 judge LLM | 专注判断一件事比让 Agent LLM 兼职更可靠 |
| 工具返回 `status` 字段 | 让 Agent LLM 做路由决策（found→展示，not_found→走 KB） |
| handler 返回 `(result, llm_str)` 两份数据 | result 给 agent 处理，llm_str（精简版）给 LLM 历史，避免历史膨胀 |
| outline context 拼入 system prompt 而非追加 system 消息 | Qwen 不允许多条 system 消息 |
| agent1 把 Steps 1-4 打包成一个工具 | 各步骤紧耦合（搜索结果是生成大纲的输入），拆开对 LLM 没有意义 |
