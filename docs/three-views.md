# 大纲三视图 & 报告三视图

系统对"大纲"和"报告"各维护三种表示形式（三视图），每种形式服务于不同的消费者。

---

## 一、大纲三视图

大纲三视图对应前端 `ChatView.jsx` 中三个平行的 state，由后端每次推送 `outline` SSE 事件时同时更新。

| 视图 | Tab 名称 | State 变量 | SSE 字段 | 消费者 |
|------|---------|-----------|---------|--------|
| 用户视图（md） | "用户" | `outlineMd` | `evt.markdown` | 用户阅读 |
| LLM 视图（llm） | "LLM" | `outlineLlm` | `evt.md_with_ids` | 大模型对话 |
| 结构视图（json） | "JSON" | `outlineJson` + `outlineJsonRef` | `evt.outline_tree` | 报告生成逻辑 |

### 1. 用户视图 — `outlineMd`

纯净的 Markdown 文本，不含节点 ID，面向用户阅读。

```
# 50GPON价值站点分析
## 50GPON升级站点-套餐和超标
### AEC覆盖用户数
...
```

由 `MarkdownOutline.jsx` 渲染，解析标题级别，展示带颜色层级标签（L1–L5）、描述文字和条件提示。

### 2. LLM 视图 — `outlineLlm`

每个节点都附带 ID，格式为 `[id=L4_018 L4] 节点名`，用于大模型在多轮对话中引用和修改特定节点。

```
[id=L3_001 L3] 50GPON价值站点分析
  [id=L4_018 L4] 50GPON升级站点-套餐和超标 | 条件：${number("AEC覆盖用户数")>0}
    [id=L5_001 L5] AEC覆盖用户数
```

注入到 Agent 的 system prompt（`build_messages` 中 `## 当前大纲` 部分），让大模型知道可以通过哪些 ID 调用 `modify_outline.py`。

### 3. 结构视图 — `outlineJson`

完整的 JSON 树对象，包含所有字段：

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

`outlineJsonRef` 是与 `outlineJson` 同步的 `useRef`，用来解决 React setState 异步问题。

**背景**：React 的 `setState` 不会立刻改变变量，新值要等到下一次 render 才可读。后端有时会在同一批 SSE 事件里连续推送 `outline`（新大纲）和 `start_report`（触发生成报告），两个事件在同一次 JavaScript 执行里依次处理，React 来不及 re-render：

```js
// 事件1：收到新大纲
outlineJsonRef.current = evt.outline_tree   // 立刻生效
setOutlineJson(evt.outline_tree)            // 安排更新，但还没 re-render

// 事件2：立刻触发生成报告（React 还没 re-render，outlineJson 仍是旧值）
generateReport()
  → const tree = outlineJsonRef.current   // ✅ 拿到新树
  → const tree = outlineJson              // ❌ 仍是 null（旧值），报告不会生成
```

因此凡是需要在同一事件批次内"刚更新完就立刻读"的场景，都通过 ref 而不是 state 读取。

报告生成时，`buildSkeleton(tree)` 从这里读取树结构生成骨架；`_process_structural` 遍历这个树调度 SQL 查询。

---

## 二、报告两视图

报告面板在 `ChatView.jsx` 中，对应两个 Tab，数据来源于后端 `/api/report` SSE 流。

| 视图 | Tab 名称 | State 变量 | 消费者 |
|------|---------|-----------|--------|
| 渲染视图（view） | "报告" | `report` + `chartData` + `tableData` | 用户阅读 |
| 源码视图（md） | "Markdown" | `report`（同上） | 调试 / 复制 |

### 1. 渲染视图 — `report` / `chartData` / `tableData`

用户最终看到的报告页面。由 `ReportView.jsx` 负责渲染：

- 调用 `ReactMarkdown` + `rehype-raw` 把带 HTML 标签的 Markdown 转成页面
- 遇到 `<div data-echart="指标名">` → 读 `chartData` 渲染 ECharts 图表（柱/线/饼）
- 遇到 `<div data-table="指标名">` → 读 `tableData` 渲染分页表格
- 左侧自动生成目录导航，标题自动编号

### 2. 源码视图 — `report`（同一 state）

与渲染视图共享同一个字符串 state，只是在 "Markdown" Tab 下用 `<pre>` 直接展示原始文本，方便复制或排查占位符替换是否正确。

**骨架说明**：`buildSkeleton(tree)` 生成的带占位符模板（`<span data-ph="...">` / `<span data-ph-summary="...">`）在内部作为局部变量使用，不再独立展示。`report` state 以骨架为初始值，随 SSE 事件持续替换占位符直到报告完成。如需调试骨架结构，在 `generateReport()` 里临时 `console.log(sk)` 即可。

---

## 三、数据流总览

```
后端 SSE 'outline' 事件
    ├─ evt.markdown    ──→  outlineMd   ──→  MarkdownOutline（"用户"Tab）
    ├─ evt.md_with_ids ──→  outlineLlm  ──→  <pre>（"LLM"Tab）& system prompt
    └─ evt.outline_tree──→  outlineJson ──→  <pre>（"JSON"Tab）& buildSkeleton()
                               │
                               ▼
                         buildSkeleton()
                               │
                               ▼
                           skeleton ──────────────────────→ <pre>（"骨架"Tab）
                               │
                         replayCaches()（复用上一次缓存）
                               │
                               ▼
后端 SSE /api/report 流
    ├─ report_metric ──→  替换 data-ph 占位符  ─┐
    ├─ report_summary──→  替换 data-ph-summary  ─┤─→  report ──→ ReportView（"报告"Tab）
    └─ report_done   ──→  结束标记               ─┘         └──→ <pre>（"Markdown"Tab）

（骨架 buildSkeleton() 结果作为局部变量，不再单独展示为 Tab）
```

---

## 四、为什么要三视图分离

| 为什么不只存一份 | 原因 |
|---------------|------|
| LLM 视图 ≠ 用户视图 | 用户不需要看节点 ID；大模型必须看 ID 才能引用节点 |
| JSON 视图 ≠ 两者 | 报告生成逻辑需要遍历树结构（level、condition_queries、summarySuggestion），文本无法满足 |
| 骨架是内部中间产物 | 骨架是模板，`report` state 是结果；两者共享同一字符串类型，合并为一个 Tab 即可 |
| `outlineJsonRef` 存在 | React setState 异步，同一批 SSE 事件内若需要立即读最新树（如自动触发报告），只能用 ref |
