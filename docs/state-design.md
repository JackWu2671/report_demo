# 大纲与报告的数据设计

系统对"大纲"和"报告"各维护多种并行的数据形式，每种形式服务于不同的消费者。
本文解释每种形式为什么必须存在、不能合并。

---

## 一、大纲三份数据

大纲在前端 `ChatView.jsx` 中以三个平行的 state 存储，由后端推送 `outline` SSE 事件时同时更新。

| 数据 | State 变量 | SSE 字段 | 消费者 |
|------|-----------|---------|--------|
| 用户可读 Markdown | `outlineMd` | `evt.markdown` | 用户阅读 |
| 带 ID 的 Markdown | `outlineLlm` | `evt.md_with_ids` | 大模型修改大纲 |
| JSON 树 | `outlineJson` + `outlineJsonRef` | `evt.outline_tree` | 前端报告生成逻辑 |

### 为什么三份都不能省

**能不能只存 JSON，让 LLM 直接读 JSON？**

不行。同一份大纲，两种格式的 token 量差异巨大：

```
// md_with_ids：~400 tokens
[id=L3_001 L3] 50GPON价值站点分析
  [id=L4_018 L4] 50GPON升级站点-套餐和超标 | 条件：${number("AEC覆盖用户数")>0}
    [id=L5_001 L5] AEC覆盖用户数

// JSON：~2000+ tokens（含 description、condition_queries、summarySuggestion 等字段）
{ "id": "L3_001", "name": "50GPON价值站点分析", "level": 3,
  "description": "...", "condition": "", "condition_queries": [],
  "summarySuggestion": "", "children": [ ... ] }
```

`md_with_ids` 是专为 LLM 设计的紧凑格式：层级用缩进表达、关键信息（ID、名称、条件）密集排在一行，LLM 扫描节点 ID 和判断结构关系的准确率更高。JSON 里的 `condition_queries`、`summarySuggestion` 等字段是给程序用的，注入给 LLM 只会引入噪声。

**能不能只存 md_with_ids，不要 JSON？**

不行。`outlineJson` 是程序逻辑必须的结构化数据：`buildSkeleton()` 遍历树生成报告骨架、`findNodeById()` 查节点、`replayCaches()` 校验 summary 缓存是否仍然有效——这些都依赖字段访问和递归遍历，文本格式无法满足。

**能不能只存 JSON 和 md_with_ids，不要用户可读版？**

不行。用户可读版由 `MarkdownOutline.jsx` 渲染，展示层级颜色标签、描述文字、条件提示；md_with_ids 的原始文本格式（`[id=L4_018 L4] 节点名`）不适合直接给用户看。

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

#### 带 ID 的 Markdown — `outlineLlm`

每个节点附带 ID，注入到 Agent 的 system prompt（`build_messages` 中 `## 当前大纲` 部分），让大模型在多轮对话中能通过 ID 调用 `modify_outline.py` 修改特定节点：

```
[id=L3_001 L3] 50GPON价值站点分析
  [id=L4_018 L4] 50GPON升级站点-套餐和超标 | 条件：${number("AEC覆盖用户数")>0}
    [id=L5_001 L5] AEC覆盖用户数
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

## 二、报告两份数据

报告在前端对应两个 Tab，数据来源于后端 `/api/report` SSE 流。

| 数据 | State 变量 | 消费者 |
|------|-----------|--------|
| 渲染报告 | `report` + `chartData` + `tableData` | 用户阅读 |
| Markdown 源码 | `report`（同上） | 调试 / 复制 |

两个 Tab 共享同一个 `report` state，区别只是渲染方式——"报告"Tab 通过 `ReportView.jsx` 渲染为带图表的页面，"Markdown"Tab 用 `<pre>` 展示原始文本。

**骨架说明**：`buildSkeleton(tree)` 生成的带占位符模板（`<span data-ph="...">` / `<span data-ph-summary="...">`）作为 `generateReport()` 内部局部变量，不单独存储。`report` state 以骨架为初始值，随 SSE 事件持续替换占位符直到报告完成。

---

## 三、数据流总览

```
后端 SSE 'outline' 事件
    ├─ evt.markdown     ──→  outlineMd   ──→  MarkdownOutline（用户阅读）
    ├─ evt.md_with_ids  ──→  outlineLlm  ──→  Agent system prompt（LLM 修改大纲）
    └─ evt.outline_tree ──→  outlineJson ──→  buildSkeleton() / findNodeById() 等逻辑
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
