# 报告生成：完整流程说明

## 一、各组件是什么 / 做什么

在看流程之前，先搞清楚每个文件/包的角色。

### 后端

| 组件 | 是什么 | 做什么 |
|------|--------|--------|
| `api_server.py` | FastAPI 服务入口 | 接收前端请求，把 `/api/report` 变成 SSE 流（长连接持续推数据） |
| `report_executor.py` | 报告主调度器 | 遍历大纲树，决定哪些指标要查，并行分发任务 |
| `sql_executor.py` | 指标查询封装 | 知道每个指标叫什么名字、对应什么 SQL、应该用什么图表类型 |
| `de_sql_execution_client.py` | HTTP 客户端 | 负责真正发 HTTP 请求：POST 提交 SQL 任务 → 轮询等结果回来 |
| `trigger_report.py` | 对话触发桥梁 | Agent（大模型）说"生成报告"时，往一个文件里写一个标记 |
| `agent.py` | Agent 主循环 | 监视那个文件，发现标记就通知前端开始生成 |
| `prefetch_mock_data.py` | 离线数据预取脚本 | 提前把所有 SQL 都查一遍，结果存到 JSON 里，网络不通时用 |

### 前端

| 组件 | 是什么 | 做什么 |
|------|--------|--------|
| `ChatView.jsx` | 主页面 | 控制整个报告生成过程：构建骨架、发请求、处理 SSE 事件、替换占位符 |
| `ReportView.jsx` | 报告渲染组件 | 把 Markdown 字符串渲染成可读的报告，包含目录导航和图表 |
| `ReactMarkdown`（npm 包） | Markdown 渲染引擎 | 把 Markdown 文本翻译成 HTML，支持自定义某些标签的渲染逻辑 |
| `echarts-for-react`（npm 包） | 图表组件 | 把 ECharts 包装成 React 组件，传入配置就能画柱状图/折线图/饼图 |
| `rehype-raw`（npm 包） | HTML 解析插件 | 让 ReactMarkdown 能识别 Markdown 里夹杂的 HTML 标签，不然会被当成纯文本 |

---

## 二、整体流程一览

```
用户点击"生成报告" 或 Agent 说"可以了"
            ↓
  前端 buildSkeleton()         ← 本地，纯同步，立刻完成
  把大纲 JSON 变成 Markdown    （用户马上看到报告骨架，每个指标位置有转圈动画）
            ↓
  前端 POST /api/report        ← 发请求给后端，带着大纲 JSON
            ↓
  后端 report_executor.py
  遍历大纲树，每个 L5 指标
  并行（最多5个）执行 SQL       ← 每查完一个，立刻通过 SSE 推给前端
            ↓
  前端收到每条结果
  把转圈动画替换成真实数据      ← 哪个先回来替换哪个，不用等全部完成
            ↓
  ReactMarkdown 渲染
  遇到图表占位符 → ECharts 画图
```

---

## 三、第一步：触发报告

两种触发方式，最终都调用同一个函数 `generateReport()`。

**方式 A：用户点按钮**
前端直接调用 `generateReport()`。

**方式 B：对话中触发（Agent 说"生成报告"）**

```
Agent 执行 trigger_report.py
    → 往 /tmp/report_sessions/{id}.json 写 {"generate_report": true}
    
agent.py 主循环每次检测文件变化
    → 发现 generate_report 变成 true
    → 通过 SSE 推 {"type": "start_report"} 给前端
    
前端收到 start_report 事件 → 调用 generateReport()
```

---

## 四、第二步：构建骨架（前端，纯本地）

`generateReport()` 被调用时，**第一件事不是等后端**，而是立刻在本地把大纲 JSON 转成带占位符的 Markdown。

### 输入（大纲 JSON）

```json
{
  "name": "IP 承载网综合评估",
  "level": 1,
  "children": [
    {
      "name": "覆盖能力",
      "level": 2,
      "children": [
        { "name": "AEC 覆盖用户数", "level": 5 },
        { "name": "AEC 覆盖率",    "level": 5 }
      ]
    }
  ]
}
```

### 输出（Markdown 骨架）

```markdown
# IP 承载网综合评估

## 覆盖能力

**AEC 覆盖用户数**

<span data-ph="AEC 覆盖用户数" class="ph-spin"></span>

**AEC 覆盖率**

<span data-ph="AEC 覆盖率" class="ph-spin"></span>
```

- L1~L4 节点 → Markdown 标题（`#` `##` `###` `####`）
- L5 节点（指标）→ 加粗名称 + `<span data-ph>` 占位符（`ph-spin` 是转圈 CSS 动画）
- `data-ph="指标名"` 是后续定向替换的关键，相当于每个占位符有唯一 ID

### 标题层级归一化

大纲不一定从 L1 开始，但报告必须有 `#` 一级标题。`buildSkeleton` 先扫最小层级，然后整体偏移：

| 大纲从哪开始 | 报告映射 |
|------------|--------|
| 从 L1 | L1→`#`，L2→`##`，L3→`###`，L4→`####` |
| 从 L2 | L2→`#`，L3→`##`，L4→`###` |
| 从 L3 | L3→`#`，L4→`##` |

---

## 五、第三步：增量缓存预填

骨架构建完后，在发请求给后端之前，前端先检查这次会话里有没有缓存过某些指标的结果（上次生成时存下来的）：

```
metricCacheRef = { "AEC覆盖用户数": "12345\n\n", "安全等级分布": "<div data-echart=...>" }

已缓存的指标 → 直接填入骨架，不等后端
未缓存的指标 → 发给后端执行，POST 时附带 cached_names 列表
```

效果：调整大纲后重新生成，没变化的指标**瞬间**显示上次的结果，只有新指标才真正跑 SQL。

---

## 六、第四步：后端执行 SQL

### 请求

```
POST /api/report
{
  "outline_tree":  { ... },           // 大纲 JSON
  "cached_names":  ["AEC覆盖用户数"]  // 前端已缓存的，跳过不查
}
```

### 遍历策略

`report_executor._walk()` 递归遍历大纲树：

```
L1 / L2 / L3 节点 → 纯结构层，只递归子节点，自身不查数据
L4 节点           → 调用 _process_l4()，负责条件判断 + 并行查指标
L5 节点（单独挂在根下的）→ 直接并行查
```

### L4 节点的处理逻辑

```
L4 节点（例："企业专线质量评估"）
  ├─ condition: "${number("企业数") > 0}"     ← 满足条件才展示这一节
  ├─ condition_queries: ["企业数"]            ← 用来算条件的辅助指标（不进报告）
  └─ children（L5）:
       ├─ 企业数          ← 只用来算条件，不进报告正文
       ├─ 专线时延分布    ← 正式指标
       └─ 专线丢包率      ← 正式指标
```

处理步骤：

```
① 分离正式指标和 condition_queries
② 排除已缓存的指标
③ 如果全部都缓存了 → 整个 L4 跳过
④ 串行检查 condition（需要先查 "企业数" 才知道结果）
⑤ condition 不满足 → 推送"该节条件不满足"事件，前端把占位符清空
⑥ condition 满足  → 并行执行所有正式指标查询
```

### 并行执行

```python
with ThreadPoolExecutor(max_workers=5) as pool:
    # 最多 5 个指标同时在查
    futures = [pool.submit(_run_metric, node) for node in uncached_metrics]
    for future in as_completed(futures):
        future.result()  # 哪个先完成哪个先 yield 事件
```

注意：`requests.Session` 不是线程安全的，所以每个 worker 各自创建独立的 `DeApiClient`（内含独立 Session）。

### SQL 查询优先级

```
① 有 exec_sql → 调用 de_sql_execution_client 查真实 API
② 查询失败或无结果 → 降级到 mock_data（评估指标_mock.json 按 id 叠加到 node.json 的字段）
③ 也没有 mock_data → 该指标无结果
```

---

## 七、第五步：SSE 事件流

后端不等所有指标都查完，每查完一个立刻推一条事件。前端通过 `EventSource` 接收（像收短信一样一条一条到）。

### 普通数值或表格

```json
{
  "type":  "report_metric",
  "name":  "AEC 覆盖用户数",
  "chunk": "12345\n\n"
}
```

`chunk` 直接是 Markdown 文本，可以是一个数字、一段文字，或者一个 Markdown 表格。

### 图表类型（BAR / LINE / PIE）

```json
{
  "type":        "report_metric",
  "name":        "安全等级分布",
  "chunk":       "| 等级 | 数量 |\n|---|---|\n| 高 | 42 |",
  "render_type": "PIE",
  "col_x":       "等级",
  "col_y":       "数量",
  "rows":        [{"等级": "高", "数量": 42}, {"等级": "中", "数量": 18}]
}
```

图表事件同时带了：
- `chunk`：Markdown 表格，万一图表渲染失败可降级显示
- `rows` + `col_x` + `col_y`：原始数据，前端用来喂给 ECharts

---

## 八、第六步：前端替换占位符

前端每收到一条 `report_metric` 事件，执行定向替换（用 `data-ph` 精准找到对应位置）：

**普通数值 / 表格：**
```
<span data-ph="AEC覆盖用户数" class="ph-spin"></span>
                    ↓ 替换为 chunk 文本
12345
```

**图表：**
```
<span data-ph="安全等级分布" class="ph-spin"></span>
                    ↓ 替换为钩子 div，同时把数据存入 chartData
<div data-echart="安全等级分布"></div>
```

`chartData` 是 React state，存的是图表原始数据：
```js
chartData["安全等级分布"] = {
  render_type: "PIE",
  col_x: "等级",
  col_y: "数量",
  rows: [{ "等级": "高", "数量": 42 }, ...]
}
```

---

## 九、第七步：渲染（ReportView.jsx）

`ReportView` 接收最终的 Markdown 字符串 + `chartData`，交给 `ReactMarkdown` 渲染。

### 图表如何从 `<div>` 变成真正的图

`ReactMarkdown` 允许"劫持"某个 HTML 标签的渲染。我们劫持了 `div`：

```jsx
<ReactMarkdown
  components={{
    div: ({ node, ...props }) => {
      const name = props['data-echart']   // 读取钩子上的指标名
      if (name && chartData[name]) {      // 查有没有对应数据
        return <ReactECharts option={buildChartOption(chartData[name])} />
      }
      return <div {...props} />           // 普通 div 正常渲染
    }
  }}
>
  {markdownText}
</ReactMarkdown>
```

当渲染到 `<div data-echart="安全等级分布">` 时，发现 `chartData` 里有数据，转交给 ECharts 画图。

### `buildChartOption` 做什么

把"行数据"翻译成 ECharts 的配置格式：

```js
// 输入
rows   = [{ "等级": "高", "数量": 42 }, { "等级": "中", "数量": 18 }]
col_x  = "等级"
col_y  = "数量"

// 输出（ECharts option）
{
  xAxis:  { data: ["高", "中"] },
  series: [{ type: 'bar', data: [42, 18] }]
}
```

| render_type | 图表类型 |
|------------|--------|
| `BAR` | 柱状图 |
| `LINE` | 折线图（平滑） |
| `PIE` | 环形图（col_x 为名称，col_y 为数值） |

### rehype-raw 为什么必须有

ReactMarkdown 默认只渲染 Markdown 语法，Markdown 里夹杂的 HTML（如 `<div data-echart="...">`）会被当成纯文本输出，不解析成真正的 HTML 元素。加上 `rehype-raw` 插件后，才能让 HTML 标签真正生效，从而触发我们的自定义 `div` 组件。

### 目录导航（TOC）

`ReportView` 扫描 Markdown 里的 H1~H4 标题，自动生成左侧目录。自定义标题组件渲染时注入 `id`：

```jsx
h1: ({ children }) => <h1 id={slugify(text)}>{children}</h1>
```

目录里的链接点击后，页面平滑滚动到对应标题。

---

## 十、数据流总结

```
大纲 JSON
    │
    ▼ buildSkeleton()  [前端，同步，毫秒级]
Markdown 骨架
（L1-L4 → 标题，L5 → <span data-ph="指标名"> 转圈占位符）
    │
    ├─ metricCacheRef 预填已缓存指标  [同步]
    │
    ▼ POST /api/report + SSE  [异步，按指标逐条返回]
    │
    ├─ condition 检查  [串行]
    │
    └─ SQL 执行  [并行，≤5个]
          │ 每条完成 → SSE 推 report_metric 事件
          ▼
前端替换占位符
  数值/表格 → chunk 文本直接写入 Markdown
  图表      → <div data-echart="名称"> + chartData 存数据
          │
          ▼
ReactMarkdown 渲染
  普通节点  → 标题/段落/表格
  <div data-echart> → buildChartOption() → ReactECharts 画图
```

---

## 十一、相关文件索引

| 文件 | 职责 |
|------|------|
| `frontend/src/pages/ChatView.jsx` | `buildSkeleton()`、`generateReport()`、SSE 事件处理、增量缓存 |
| `frontend/src/components/ReportView.jsx` | Markdown 渲染、ECharts 图表、TOC 导航 |
| `backend/api_server.py` | `/api/report` 端点、SSE 流管理 |
| `backend/services/report_executor.py` | 大纲树遍历、condition 检查、并行 SQL 调度 |
| `backend/services/sql_executor.py` | 指标名 → SQL、mock_data 离线降级 |
| `backend/services/de_sql_execution_client.py` | POST 提交 SQL 任务 → 轮询等结果 |
| `backend/skills/analyze-network/scripts/trigger_report.py` | 对话触发：写 session 标记 |
| `backend/agent_with_skills/agent.py` | 检测 session 标记，推 `start_report` 事件 |
| `backend/scripts/prefetch_mock_data.py` | 离线预取 SQL 数据，写入 `mock_data` 字段 |
