# 大纲与报告的数据设计

系统对"大纲"和"报告"各维护多种并行的数据形式，每种形式服务于不同的消费者。
本文解释每种形式为什么必须存在、不能合并。

---

## 一、大纲三份数据

大纲在前端 `ChatView.jsx` 中以三个平行的 state 存储，由后端推送 `outline` SSE 事件时同时更新。

| 数据 | State 变量 | SSE 字段 | 消费者 |
|------|-----------|---------|--------|
| 用户可读 Markdown | `outlineMd` | `evt.markdown` | 用户阅读 |
| YAML（精简视图） | `outlineLlm` | `evt.outline_yaml` | 大模型上下文 |
| JSON 树 | `outlineJson` + `outlineJsonRef` | `evt.outline_tree` | 前端报告生成逻辑 |

### 为什么三份都不能省

**能不能只存 JSON，让 LLM 直接读 JSON？**

不行。同一份大纲，两种格式的 token 量差异巨大：

```
# outline_yaml（精简 YAML）：~300 tokens
- id: L3_001
  name: 50GPON价值站点分析
  children:
    - id: L4_018
      name: 50GPON升级站点-套餐和超标
      condition: ${number("AEC覆盖用户数")>0}
      condition_queries: [AEC覆盖用户数]
      children:
        - id: L5_001
          name: AEC覆盖用户数

# JSON（完整树）：~2000+ tokens（含 summarySuggestion、renderType、exec_sql 等字段）
{ "id": "L3_001", "name": "50GPON价值站点分析", "level": 3,
  "description": "...", "condition": "", "condition_queries": [],
  "summarySuggestion": "", "renderType": "", "exec_sql": "...", "children": [ ... ] }
```

`outline_yaml` 是专为 LLM 设计的精简视图：只保留 LLM 需要随时引用的字段（id/name/description/condition/condition_queries），`level`、SQL 字段（exec_sql/apiName 等）、summarySuggestion 等字段默认省略——LLM 需要时可调用 `get_node_detail.py <node_id>` 按需拉取，而不是污染每次请求的上下文。

**能不能只存 outline_yaml，不要 JSON？**

不行。`outlineJson` 是程序逻辑必须的结构化数据：`buildSkeleton()` 遍历树生成报告骨架、`findNodeById()` 查节点、`replayCaches()` 校验 summary 缓存是否仍然有效——这些都依赖字段访问和递归遍历，文本格式无法满足。

**能不能只存 JSON 和 outline_yaml，不要用户可读版？**

不行。用户可读版由 `MarkdownOutline.jsx` 渲染，展示层级颜色标签、描述文字、条件提示；outline_yaml 的原始文本格式不适合直接给用户看。

三份数据职责完全不重叠，每份都有唯一消费者，都不能省。

---

### 各份数据详解

#### 用户可读 Markdown — `outlineMd`

纯净的 Markdown 文本，不含节点 ID：

```
# 50GPON价值站点分析
## 50GPON升级站点-套餐和超标
### AEC覆盖用户数
```

由 `MarkdownOutline.jsx` 渲染，展示带颜色层级标签（L1–L5）、描述文字和条件提示。

#### YAML 精简视图 — `outlineLlm`

注入到 Agent 的 system prompt（`build_messages` 中 `## 当前大纲` 部分），让大模型在多轮对话中能通过 ID 调用 `modify_outline.py` 修改特定节点。格式省略 level、SQL 字段和空字段，只保留 LLM 需要随时引用的内容：

```yaml
- id: L3_001
  name: 50GPON价值站点分析
  description: 针对50GPON站点的套餐价值和升级潜力进行评估
  children:
    - id: L4_018
      name: 50GPON升级站点-套餐和超标
      condition: ${number("AEC覆盖用户数")>0}
      condition_queries:
        - AEC覆盖用户数
      children:
        - id: L5_001
          name: AEC覆盖用户数
```

LLM 如需查看某节点的完整信息（summarySuggestion、exec_sql、renderType 等），可调用：

```bash
python3 $SKILLS_DIR/analyze-network/scripts/get_node_detail.py L4_018
```

#### JSON 树 — `outlineJson` / `outlineJsonRef`

完整的结构化树对象，供前端程序逻辑使用：

```json
{
  "id": "L3_001",
  "name": "50GPON价值站点分析",
  "level": 3,
  "description": "...",
  "condition": "",
  "condition_queries": [],
  "summarySuggestion": "",
  "children": [...]
}
```

`outlineJsonRef` 是与 `outlineJson` 同步的 `useRef`，解决 React setState 异步问题。

**背景**：`setState` 不会立刻改变变量，新值要等到下一次 render 才可读。后端有时在同一批 SSE 事件里连续推送 `outline` 和 `start_report`，两个事件在同一次 JavaScript 执行里依次处理，React 来不及 re-render：

```js
// 事件1：收到新大纲
outlineJsonRef.current = evt.outline_tree   // 立刻生效
setOutlineJson(evt.outline_tree)            // 安排更新，但还没 re-render

// 事件2：立刻触发生成报告（React 还没 re-render，outlineJson 仍是旧值）
generateReport()
  → const tree = outlineJsonRef.current   // ✅ 拿到新树
  → const tree = outlineJson              // ❌ 仍是 null（旧值），报告不会生成
```

---

## 二、大纲的四种后端表示

在后端，大纲通过 `outline_utils.py` 在四种表示之间转换：

| 函数 | 输入 | 输出 | 用途 |
|------|------|------|------|
| `to_markdown(tree)` | outline_tree | 纯 Markdown | 前端用户视图 |
| `to_yaml(tree)` | outline_tree | YAML 精简文本 | LLM 上下文（system prompt） |
| `to_clean_json(tree)` | outline_tree | 干净 JSON dict | 前端 JSON 树、存储 |
| `from_yaml(text)` | YAML 文本 | outline_tree | LLM 写 YAML → 解析回树 |

唯一权威数据源是 `outline_tree`（dict）。`to_yaml()` 省略 level 和 SQL 字段；`from_yaml()` 通过 ID 前缀（`L1_xxx` → level 1）自动推断 level。

---

## 三、报告两份数据

报告在前端对应两个 Tab，数据来源于后端 `/api/report` SSE 流。

| 数据 | State 变量 | 消费者 |
|------|-----------|--------|
| 渲染报告 | `report` + `chartData` + `tableData` | 用户阅读 |
| Markdown 源码 | `report`（同上） | 调试 / 复制 |

两个 Tab 共享同一个 `report` state，区别只是渲染方式——"报告"Tab 通过 `ReportView.jsx` 渲染为带图表的页面，"Markdown"Tab 用 `<pre>` 展示原始文本。

**骨架说明**：`buildSkeleton(tree)` 生成的带占位符模板（`<span data-ph="...">` / `<span data-ph-summary="...">`）作为 `generateReport()` 内部局部变量，不单独存储。`report` state 以骨架为初始值，随 SSE 事件持续替换占位符直到报告完成。

---

## 四、数据流总览

```
后端 SSE 'outline' 事件
    ├─ evt.markdown      ──→  outlineMd   ──→  MarkdownOutline（用户阅读）
    ├─ evt.outline_yaml  ──→  outlineLlm  ──→  Agent system prompt（LLM 修改大纲）
    └─ evt.outline_tree  ──→  outlineJson ──→  buildSkeleton() / findNodeById() 等逻辑
                                │
                                ▼
                          buildSkeleton()  （局部变量 sk）
                                │
                          replayCaches()   （复用上一次缓存）
                                │
                                ▼
后端 SSE /api/report 流
    ├─ report_metric  ──→  替换 data-ph 占位符   ─┐
    ├─ report_summary ──→  替换 data-ph-summary  ─┤─→  report ──→ ReportView（渲染）
    └─ report_done    ──→  结束标记               ─┘         └──→ <pre>（Markdown 源码）
```
