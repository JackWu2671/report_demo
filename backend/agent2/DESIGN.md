# Agent2 设计文档

> 面向零基础 Agent 开发人员
>
> 本文档解释 `backend/agent2/` 的设计思路、每个模块的职责，以及关键设计决策背后的原因。

---

## 1. 什么是 Agent？

传统程序的执行路径是固定的：`A → B → C → 结束`。

Agent 的执行路径是**由大模型在运行时动态决定的**：大模型根据当前对话状态，自主选择"下一步调用哪个工具"，直到任务完成。

```
用户输入
  ↓
大模型思考：我应该做什么？
  ↓
选择并调用工具（或直接回答）
  ↓
看到工具结果后，再次思考
  ↓
... 循环直到任务完成 ...
  ↓
输出最终回复
```

这个"思考 → 行动 → 观察 → 再思考"的循环就是 **ReAct 模式**（Reason + Act）。

---

## 2. 本项目的业务背景

用户需要生成网络分析报告大纲，这个过程：

1. **有知识库**：已有结构化的网络分析节点，存储在 FAISS 向量索引里
2. **有模板**：过去积累的大纲可以复用，不必每次从零生成
3. **需要多轮修改**：用户看到大纲后，会说"删掉这一节"、"加个阈值参数"
4. **大纲是结构化数据**：不只是文字，需要以 JSON 形式存储供下游执行

Agent 的价值在于：**让大模型决定走"复用模板"还是"从知识库生成"**，而不是写死 if/else。

---

## 3. 整体架构

```
backend/agent2/
├── agent.py           ← Agent 主循环（ReAct loop）
├── prompt.txt         ← 系统提示词（告诉大模型有哪些工具）
├── agent_test.py      ← 命令行交互测试入口
│
├── memory/
│   └── store.py       ← 状态管理（大纲 + 对话历史）
│
└── tools/
    ├── definitions.py     ← 工具的 JSON Schema（大模型看到的工具说明）
    ├── search_template.py ← 工具实现：检索预制大纲模板
    ├── generate.py        ← 工具实现：从知识库生成大纲
    ├── modify.py          ← 工具实现：修改现有大纲
    └── handlers.py        ← 工具调度表 + 状态写入
```

---

## 4. 核心概念：工具调用（Tool Calling）

大模型原生只能输出文字。**工具调用**是一种约定：

- 我们把工具描述（名称 + 参数 + 用途说明）以 JSON Schema 的形式传给大模型
- 大模型可以在回复里说"我想调用 `generate_outline`，参数是 `question=政企OTN升级`"
- 我们的代码检测到这个意图，真正执行函数，把结果返回给大模型
- 大模型看到结果，决定继续调工具还是直接回答用户

```python
# tools/definitions.py — 大模型看到的工具说明（不是实现，只是描述）
{
    "type": "function",
    "function": {
        "name": "generate_outline",
        "description": "从知识库实时检索并生成报告大纲...",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "用户需求"}
            }
        }
    }
}
```

工具的**实际执行代码**在 `tools/generate.py`，两者通过 `tools/handlers.py` 连接。

---

## 5. Agent 主循环（agent.py）

这是整个系统的心脏，代码逻辑只有一件事：**循环调用大模型，直到它不再要求调用工具为止**。

```
while 未达到最大轮数:
    调用大模型（携带对话历史 + 工具列表）
    
    if 大模型想调用工具:
        执行工具
        把工具结果追加到对话历史
        continue（让大模型再次思考）
    
    else（大模型直接回答）:
        输出最终文字
        break
```

对应代码中的 `chat_stream()` 方法，是一个 **async generator**（异步生成器），每发生一件事就 `yield` 一个事件出去，而不是等全部完成再返回。

### 为什么用 async generator？

因为整个流程耗时较长（向量检索、多次 LLM 调用），用户需要实时看到进度。Generator 允许我们边执行边推送事件，前端可以即时渲染。

---

## 6. 三种工具的设计

### 6.1 search_outline_template — 检索预制模板

```
embed_query(question)        → 把问题转成向量
search_templates(vec, top_k) → 向量相似度检索，找最像的模板
select_template(question, candidates) → 内部 LLM 判断：这些模板够用吗？

返回: {"status": "found" | "not_found", ...}
```

注意：`select_template` 内部有一次 LLM 调用，它是一个**专注于单一判断任务**的小 LLM 调用，不是 Agent 主循环里的那个 LLM。

**为什么不让外层 Agent LLM 自己判断？**

如果让 Agent LLM 看到原始候选列表自己决定，它需要同时处理"选工具"和"评估模板质量"两件事，容易出错。内部封装一个专注的 judge LLM 更可靠，对外只暴露 `found/not_found` 这个干净的结论。

### 6.2 generate_outline — 从知识库生成

```
embed_query(question)                  → 问题向量化
load_resources()                       → 加载 FAISS 索引 + 知识图谱 JSON
search_nodes(vec, faiss_svc)           → 检索相关节点
build_candidate_paths(hits, ...)       → 补全祖先路径
select_anchor(question, candidates)    → LLM 选锚节点
build_subtree(anchor_id, ...)         → 递归构建子树
parse_patch(question, tree)           → LLM 解析参数修改意图
apply_patch(tree, ops)                → 应用修改

返回: {"status": "success" | "not_found", ...}
```

`not_found` 意味着知识库里没有相关节点，这时 Agent 应该告知用户"系统暂不支持该分析场景"，**不应重试**。

### 6.3 modify_outline — 修改现有大纲

```
parse_patch(instruction, tree)   → LLM 把自然语言指令转成结构化操作
apply_patch(tree, ops)           → 纯 Python 执行操作（不再调用 LLM）

支持的操作: delete（删节点）、set_param（设参数）、keep_only（只保留指定节点）
```

---

## 7. 大纲的三种表示

大纲数据有三种形态，服务不同的消费者：

| 表示 | 格式 | 用途 | 谁使用 |
|------|------|------|------|
| `outline_tree` | JSON dict | 程序执行的数据结构 | 代码、下游系统 |
| `markdown` | `# 标题\n## 子章节` | 人类阅读 | 用户界面 |
| `md_with_ids` | `[L1 L1_001] 传送网：描述` | 带节点 ID 的引用视图 | LLM 上下文 |

三者由 `outline_utils.py` 中的函数互相转化，`outline_tree` 是唯一的数据源：

```
outline_tree → to_markdown()        → markdown
outline_tree → to_markdown_with_ids() → md_with_ids
```

**为什么 LLM 不能直接看普通 Markdown？**

因为修改大纲时，LLM 需要精确引用节点（"把 `L2_003` 节点删掉"），普通 Markdown 没有节点 ID，LLM 只能用名称描述，容易出错或定位歧义。

---

## 8. Memory（状态管理）

`AgentMemory` 管理两件事：

### 8.1 大纲状态（不进入对话历史）

```python
memory.outline_tree   # JSON 树（源数据）
memory.markdown       # 给用户看的 Markdown
memory.md_with_ids    # 给 LLM 看的带 ID Markdown
```

大纲不存进对话历史消息里，原因：
- 大纲可能很长（数千字），每轮都放在历史里会浪费 token
- 大纲会被修改，历史里保存的是"旧版本"，容易混淆

### 8.2 对话历史（精简版）

历史里存的 tool 结果只包含 `md_with_ids`（带 ID 的紧凑视图），不包含完整 Markdown。这样历史始终保持较小体积。

### 8.3 上下文注入

每次调用 LLM 前，`build_messages()` 把当前 `md_with_ids` 拼进系统提示词里：

```python
def build_messages(self, system_prompt):
    if self.has_outline:
        system_content = system_prompt + "\n\n## 当前大纲\n" + self.md_with_ids
    else:
        system_content = system_prompt
    
    return [{"role": "system", "content": system_content}, *self._history]
```

**为什么不追加一条新的 system 消息？**

许多模型（如 Qwen）要求 system 消息只能出现在最前面，追加到末尾会报错 `400: System message must be at the beginning`。把内容拼进第一条 system 消息是更兼容的做法。

---

## 9. 事件协议（Event Protocol）

`chat_stream()` 是一个 async generator，yield 以下类型的事件：

```python
{"type": "step",    "name": "search_outline_template", "status": "running"}
{"type": "step",    "name": "search_outline_template", "status": "done"}
{"type": "outline", "markdown": "# fgOTN部署\n## 传送网络覆盖..."}
{"type": "text",    "chunk": "已找到匹配大纲，共3个章节。"}
{"type": "done",    "seconds": 3.2}
{"type": "error",   "message": "工具执行失败: ..."}
```

### 为什么不让 LLM 直接输出大纲文字？

大纲由工具计算得出，是**已完成的数据**，不需要 LLM "逐字生成"。如果让 LLM 在文字回复里输出 2000 字的大纲，用户要等几十秒看 LLM 吐字。而 `outline` 事件是工具执行完后**立即推送**的，前端可以瞬间渲染。

LLM 的文字职责只有一件事：**用 1-2 句话说明刚才做了什么**，引导用户下一步操作。

---

## 10. 工具与 Handler 的关系

```
tools/definitions.py   ← 大模型看到的"菜单"（只有描述，没有实现）
tools/search_template.py / generate.py / modify.py  ← 实际执行函数
tools/handlers.py      ← 连接两者的调度表
```

Handler 的职责（以 `handle_generate_outline` 为例）：

```python
async def handle_generate_outline(args, memory):
    # 1. 调用工具函数
    result = await generate_outline(args["question"])
    
    # 2. 如果成功，更新 memory（outline_tree / markdown / md_with_ids）
    if result["status"] == "success":
        memory.set_outline(result["outline_tree"], result["markdown"], result["md_with_ids"])
    
    # 3. 返回两份数据：
    #    result      → agent 检查是否有大纲，决定是否 yield outline 事件
    #    llm_str     → 注入对话历史的简洁版（只含 md_with_ids，不含完整 markdown）
    llm_str = f"[generate_outline] status=success\n\n{result['md_with_ids']}"
    return result, llm_str
```

---

## 11. 大纲在对话中的流转

以"生成大纲 → 修改大纲"两轮对话为例：

```
第一轮：用户说"政企OTN升级"

  Agent → LLM（2条消息: system + user）
  LLM → tool_call: search_outline_template("政企OTN升级")
  handler 执行 → memory.set_outline(tree, markdown, md_with_ids)
  agent → yield {"type": "outline", "markdown": "..."}   ← 大纲直接推送
  Agent → LLM（5条消息: system+outline注入, user, assistant工具调用, tool结果）
  LLM → "已找到匹配大纲，共3个章节。"
  agent → yield {"type": "text", "chunk": "已找到..."}

------------------------------------------------------

第二轮：用户说"删掉传送网络容量分析"

  Agent → LLM（系统提示词里已含最新 md_with_ids，共4条消息）
  LLM → tool_call: modify_outline("删掉传送网络容量分析")
  handler 执行 → memory.set_outline(new_tree, ...)       ← memory 更新
  agent → yield {"type": "outline", "markdown": "..."}   ← 新大纲推送
  Agent → LLM（看到修改后的 md_with_ids）
  LLM → "已删除传送网络容量分析章节，大纲现有2个一级章节。"
```

---

## 12. 设计决策总结

| 决策 | 原因 |
|------|------|
| 大纲不进对话历史，通过 memory 注入 | 避免 token 浪费，确保 LLM 始终看到最新版本 |
| outline 走独立事件通道 | 大纲是计算结果，不应流式逐字输出 |
| search_template 内部封装 LLM 判断 | 专注的 judge LLM 比让 Agent LLM "兼职"更可靠 |
| 工具返回 status 字段 | 让 Agent LLM 做路由决策（found→展示，not_found→改走KB） |
| handlers 返回 (result, llm_str) 两份 | result 给 agent 处理，llm_str（精简版）给 LLM 历史 |
| outline context 拼入 system prompt | 避免多 system 消息导致模型报错 |

---

## 13. 后续扩展方向

**render_report（未实现）**

当用户确认大纲后，可以新增一个 `render_report` 工具：
- 读取 `memory.outline_tree`（JSON 结构）
- 按节点逐一执行 L5 数据查询，生成各章节内容
- 每完成一个章节 yield `{"type": "report_section", "title": "...", "content": "..."}`

整个事件协议无需修改，只是新增一种 type。

**工具内部子步骤可见性**

当前 `generate_outline` 内部有 5-6 个子步骤，对用户显示为一个大的 `step`。若要让用户看到内部进度，可以把工具改为 async generator，yield 子步骤事件。这会增加复杂度，适合工具耗时超过 30 秒时再考虑。

**模板沉淀**

用户确认过的大纲可以保存为新模板（写入 `templates/` 目录），下次遇到相似需求直接复用，无需再走 KB 生成。
