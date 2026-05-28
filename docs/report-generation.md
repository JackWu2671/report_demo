# 从大纲到报告：完整流程说明

本文档描述报告生成的完整技术流程，涵盖触发方式、骨架构建、SQL 执行、图表渲染等各个环节。

---

## 整体架构

```
用户操作 / Agent 识别意图
        ↓
  trigger_report.py           ← 写 session 标记
        ↓
  agent._detect_events()      ← 检测到标记，推 start_report 事件
        ↓
  前端 generateReport()       ← 立刻构建骨架，同时发起 POST /api/report
        ↓
  后端 report_executor.py     ← 遍历大纲树，并行执行 SQL
        ↓
  SSE 事件流 report_metric    ← 每条指标数据独立推送
        ↓
  前端替换占位符              ← 转圈 → 数字 / 表格 / 图表
```

---

## 第一步：触发报告生成

报告有两种触发方式，最终都走同一条路。

### 方式一：用户点击"生成报告"按钮

前端直接调用 `generateReport()`。

### 方式二：用户在对话中说"可以了 / 生成报告 / ok"

Agent（大模型）识别到确认意图后，调用脚本：

```python
# backend/skills/analyze-network/scripts/trigger_report.py
data["generate_report"] = True
session.write(data)   # 写入 /tmp/report_sessions/{id}.json
```

Agent 主循环在每次 bash 工具执行后会比对 session 文件的前后状态：

```python
# backend/agent_with_skills/agent.py  _detect_events()
if after.get("generate_report") and not before.get("generate_report"):
    after["generate_report"] = False   # 立刻复位，避免重复触发
    session.write(after)
    events.append({"type": "start_report"})
```

这个 `start_report` 事件通过 SSE 推给前端，前端收到后同样调用 `generateReport()`。

---

## 第二步：构建报告骨架（前端）

`generateReport()` 被调用时，第一件事不是等后端，而是**立刻在本地**把大纲 JSON 转成 Markdown 字符串作为骨架，让用户马上看到结构。

```
outline_tree (JSON)  →  buildSkeleton()  →  Markdown 字符串（含占位符）
```

### 标题层级归一化

大纲可能从任意层级（L1~L4）开始，`buildSkeleton` 会先扫描树找最小 level，再相对偏移，确保报告里永远有 `#` 一级标题：

| 大纲起始 | 映射关系 |
|---------|---------|
| 从 L1 开始 | L1→`#`，L2→`##`，L3→`###`，L4→`####` |
| 从 L2 开始 | L2→`#`，L3→`##`，L4→`###` |
| 从 L3 开始 | L3→`#`，L4→`##` |

### 骨架示例

```markdown
# IP 承载网综合评估

## 覆盖能力

### AEC 覆盖评估

#### 基础覆盖指标

**AEC 覆盖用户数**

<span data-ph="AEC 覆盖用户数" class="ph-spin"></span>

**AEC 覆盖率**

<span data-ph="AEC 覆盖率" class="ph-spin"></span>

---
```

每个 L5 指标（query 节点）的位置放一个 `<span>` 标签，`class="ph-spin"` 对应 CSS 里的转圈动画，用户看到的是一个小圆圈表示"加载中"。

`<span data-ph="指标名">` 里的 `data-ph` 属性存着指标名，后续用来精确匹配替换。

---

## 第三步：增量缓存预填

`generateReport()` 在构建骨架后，会检查本次会话内是否已经查过某些指标（上次生成报告时缓存的）：

```js
const cache = metricCacheRef.current          // { "指标名" → "已有数据" }
const allNames = 从骨架里提取所有 data-ph 的名字
const cachedNames = allNames.filter(n => cache[n] !== undefined)

// 已缓存的：立刻预填，不等后端
for (const name of cachedNames) {
    prefilled = prefilled.replace(`<span data-ph="${name}">`, cache[name])
}

// 向后端只请求未缓存的
POST /api/report { outline_tree, cached_names: cachedNames }
```

效果：调整大纲后再次生成，没有变化的指标**瞬间**显示上次结果，只有新增/变化的指标才真正执行 SQL。

---

## 第四步：后端执行 SQL（report_executor.py）

### 请求结构

```
POST /api/report
{
  "outline_tree": { ... },       // 大纲 JSON
  "cached_names": ["指标A", ...] // 前端已缓存、不需要重新执行的指标名
}
```

### 遍历策略

`_walk()` 递归遍历大纲树：
- L1 / L2 / L3 节点：纯结构，只递归子节点
- L4 节点：调用 `_process_l4()`

### L4 节点处理（_process_l4）

每个 L4 节点可能有 `condition` 字段（展示条件）和 `condition_queries` 字段（用于计算条件的辅助指标）。

```
L4 节点
  ├─ condition: "${number("企业数") > 0}"   ← 是否需要判断条件
  ├─ condition_queries: ["企业数"]           ← 计算条件用的辅助指标（不出现在报告里）
  └─ children (L5 query 节点):
       ├─ 企业数         ← condition_query，不直接查询
       ├─ 安全等级分布   ← 正式指标，需要查询
       └─ 覆盖用户数     ← 正式指标，需要查询
```

处理流程：

```
1. 筛选出正式指标（排除 condition_queries）
2. 排除已缓存的指标（来自前端 cached_names）
3. 如果全部都缓存了 → 跳过整个 L4，连 condition 也不检查
4. 串行执行 condition 检查（eval_condition，用共享 client）
5. condition 不满足 → 给所有指标推送"条件不满足"消息
6. condition 满足 → 并行执行所有正式指标查询
```

### 并行执行（ThreadPoolExecutor）

```python
with ThreadPoolExecutor(max_workers=5) as pool:
    futures = { pool.submit(_run_metric, l5, executor, on_event): l5
                for l5 in uncached }
    for future in as_completed(futures):
        future.result()
```

- 同时最多 5 个查询并发执行
- 每个 worker 创建自己的 `DeApiClient`（`requests.Session` 不线程安全）
- `SqlExecutor` 实例共享（初始化后只读，线程安全）
- 结果乱序推送（前端按名字匹配，顺序无关）

### mock_data 离线缓存

如果 `expert_knowledge/评估指标_mock.json` 存在（通过 `scripts/prefetch_mock_data.py` 预先生成），`SqlExecutor` 会优先从 `mock_data` 字段读数据，完全不走 API：

```python
if "mock_data" in record:
    return { "rows": record["mock_data"], ... }  # 直接返回
# 否则才执行 SQL
```

---

## 第五步：SSE 事件流

后端通过 Server-Sent Events 实时推送，每条指标执行完立刻推，不等全部完成：

### 普通数值型

```json
{ "type": "report_metric", "name": "AEC 覆盖用户数", "chunk": "12345\n\n" }
```

### 表格型

```json
{ "type": "report_metric", "name": "设备型号分布", "chunk": "| 型号 | 数量 |\n|---|---|\n| A | 100 |\n" }
```

### 图表型（BAR / LINE / PIE）

```json
{
  "type":        "report_metric",
  "name":        "安全等级分布",
  "chunk":       "| 等级 | 数量 |\n...",  ← Markdown 表格作为降级文本
  "render_type": "PIE",
  "col_x":       "等级",
  "col_y":       "数量",
  "rows":        [{"等级": "高", "数量": 42}, ...]
}
```

图表事件同时携带 Markdown 表格（`chunk`）和原始数据（`rows`），前端优先使用图表渲染，降级时显示表格文本。

---

## 第六步：前端替换占位符

前端收到每个 `report_metric` 事件后，执行字符串替换：

```
旧：<span data-ph="安全等级分布" class="ph-spin"></span>
                    ↓
新（表格/数值）：直接写入 chunk 文本
新（图表）：     <div data-echart="安全等级分布"></div>
```

图表数据同时存入 `chartData` state：

```js
chartData["安全等级分布"] = { render_type: "PIE", col_x: "等级", col_y: "数量", rows: [...] }
```

替换后的 Markdown 字符串示例：

```markdown
**安全等级分布**

<div data-echart="安全等级分布"></div>
```

---

## 第七步：渲染（ReportView.jsx）

### 技术栈

- `ReactMarkdown`：把 Markdown 字符串渲染成 React 组件树
- `remark-gfm`：支持 GFM 语法（表格、任务列表等）
- `rehype-raw`：允许 Markdown 里的 HTML 标签真正渲染（否则 `<div>` 会被当成文本输出）
- `echarts-for-react`：ECharts 图表组件

### 自定义 div 组件

```jsx
div: (props) => {
    const name = props['data-echart']      // 读取属性
    if (name && chartData[name]) {         // 查有没有数据
        const option = buildChartOption(chartData[name])
        return <ReactECharts option={option} style={{ height: 280 }} />
    }
    return <div {...props}>{children}</div> // 普通 div 透传
}
```

当 `ReactMarkdown` 渲染到 `<div data-echart="安全等级分布">` 时，走自定义组件，查到 `chartData` 里的数据后渲染 ECharts 图表。

### 图表类型映射

| render_type | ECharts 类型 | 数据取法 |
|------------|-------------|---------|
| `BAR` | `bar` | `col_x` 为 X 轴分类，`col_y` 为数值 |
| `LINE` | `line`（smooth） | 同上 |
| `PIE` | `pie`（环形） | `col_x` 为名称，`col_y` 为数值 |

### 目录导航

`ReportView` 解析 Markdown 里的 H1~H4 标题，自动生成左侧 TOC 导航，点击平滑滚动到对应位置。自定义标题组件在渲染时注入 `id` 属性，供锚点跳转使用：

```jsx
h1: ({ children, ...props }) => {
    const id = slugify(text)   // "IP 承载网综合评估" → "IP承载网综合评估"
    return <h1 id={id} {...props}>{children}</h1>
}
```

---

## 数据流总结

```
outline_tree (JSON)
    │
    ▼ buildSkeleton() [前端，同步]
Markdown 骨架（含 <span data-ph> 转圈占位符）
    │
    ├── metricCacheRef 预填已缓存指标 [同步]
    │
    ▼ POST /api/report [异步 SSE 流]
    │
    ├── L4 condition 检查 [串行]
    │
    └── L5 metric SQL 执行 [并行，最多 5 个]
            │ 每条完成立刻推送 report_metric 事件
            ▼
前端替换：<span data-ph> → 文字 / <div data-echart>
            │
            ▼
ReactMarkdown 渲染：<div data-echart> → ECharts 图表
```

---

## 相关文件索引

| 文件 | 职责 |
|------|------|
| `frontend/src/pages/ChatView.jsx` | `buildSkeleton()`、`generateReport()`、SSE 事件处理、增量缓存 |
| `frontend/src/components/ReportView.jsx` | Markdown 渲染、ECharts 图表组件、TOC 导航 |
| `backend/api_server.py` | `/api/report` 端点、SSE 流管理 |
| `backend/services/report_executor.py` | 大纲树遍历、condition 检查、并行 SQL 执行 |
| `backend/services/sql_executor.py` | 指标名→SQL 查询、mock_data 离线缓存 |
| `backend/services/de_sql_execution_client.py` | HTTP 客户端，POST 提交任务→GET 轮询结果 |
| `backend/skills/analyze-network/scripts/trigger_report.py` | 对话触发：写 session 标记 |
| `backend/agent_with_skills/agent.py` | 检测 session 标记，推 `start_report` 事件 |
| `backend/scripts/prefetch_mock_data.py` | 离线预取 SQL 数据，写入 `mock_data` 字段 |
