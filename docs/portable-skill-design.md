# 可移植「一句话生成报告」Skill 设计

> 目标：把「一句话 → 完整网络分析报告」的全部能力收敛成一个**可一键安装到开源 agent（opencode 等）**的 skill 包；数据不放进 skill，而是放在与 `skills/` 平行的 `reference/` 目录。
>
> 状态：设计稿（不含实现）。

---

## 1. 目标与约束

| 维度 | 要求 |
|------|------|
| 体验 | 用户给一句话（如「评估某城域网超千兆升级」），无需多轮确认，直接产出一份完整报告 |
| 产出 | 一个**落地文件**（`report.md`，图表内联），而非依赖前端/SSE 的实时流 |
| 可移植 | 能装进 opencode、Claude Code 等任何遵循 Agent Skills（`SKILL.md`）约定的 agent |
| 数据位置 | 知识数据 / 索引 / 共享库放在与 `skills/` **平行**的 `reference/`，不塞进 skill 文件夹 |
| 安装 | 一条命令（git clone / install.sh）即可装好并配置可达的服务端点 |
| 外部依赖 | LLM / Embedding / SQL 执行 API 仍是外部服务，通过环境变量配置，不打包 |

---

## 2. 现状诊断：当前 skill 为什么「装不进」opencode

当前 `backend/skills/analyze-network/` 看似是标准 skill，实则是后端的一层薄 SOP，对外有 5 个硬耦合点：

| 耦合点 | 当前依赖 | opencode 里是否存在 |
|--------|----------|--------------------|
| 脚本 import | `from services.llm_service / faiss_service / report_executor / sql_executor` | ❌ 无此 backend |
| 共享库 | `skills/_lib/*`（retriever / patcher / outline_utils / loader…） | ❌ 不在 skill 目录内 |
| 状态桥 | `/tmp/report_sessions/{id}.json` + `agent.py._detect_events` 检测变化 | ❌ 无此 agent 主循环 |
| 渲染 | `trigger_report.py` 只写 `generate_report=True` 标记 → 前端 `/api/report` SSE 流 → React + ECharts 渲染 | ❌ 无前端、无 SSE |
| 数据 | `expert_knowledge/*` + FAISS 索引 + SQL 执行 API | ❌ 不在 skill 内 |

**关键事实**：报告**根本不是 skill 直接生成的**。`trigger_report.py` 只是写一个信号位，由 `agent.py`（`agent.py:582`）检测到后向前端推 `start_report` 事件，前端再调 `report_executor.run_report()`（`services/report_executor.py:37`）通过 SSE 逐条推送指标和总结，最终由 React 组件渲染成带 ECharts 图表的页面。

这套「信号 + SSE + 前端」机制在 opencode 中完全不存在。因此移植的核心不是搬文件，而是**换一种产出形态**。

---

## 3. 可移植性核心原则：产出「文件」而非「事件」

opencode 给 agent 的能力很朴素：`read` / `write` / `bash`。可移植 skill 必须把整条链收敛成「输入一句话 → 输出一组文件」：

```
输入：一句话（用户需求原文）
  │
  ▼  skill 自带脚本链（检索 → 大纲 → 修剪 → 查询 → 渲染）
  ▼  外部服务：LLM / Embedding / SQL（env 配置）
  ▼  数据：parallel reference/（知识图谱 + FAISS 索引）
  │
  ▼
输出（多视图产物，见 §3.1）：
  outline.json / outline.md / outline.yaml   大纲三视图
  report.md / report.html                     报告两视图
  report.state.json                           增量用隐藏状态
```

没有 session 桥、没有 SSE、没有前端。agent 跑完脚本，磁盘上多出这组文件，opencode 直接打开。这才是「把全部内容写进 skill」的真正含义——**逻辑、SOP、产出全部收敛进 skill 包内**，外部只剩可配置的服务端点与平行 `reference/` 数据。

### 3.1 多视图产物模型（真源 + 投影）

产物按「同一份结构化真源 → 多种渲染投影」组织，**视图之间不并列可编辑**，否则手改某个视图会与真源漂移。

| 产物 | 角色 | 由谁生成 | 用途 |
|------|------|----------|------|
| `outline.json` | **大纲真源**（`outline_tree`） | 检索/编辑流水线 | 机器读、唯一可改对象 |
| `outline.md` | 大纲投影 | `outline_utils.to_markdown` | 人读、可 diff |
| `outline.yaml` | 大纲投影 | `outline_utils.to_yaml` | 人读 / agent 上下文格式 |
| `report.md` | 报告投影（源视图） | `render.py`（Mermaid + 表格） | 可移植、可 diff、opencode 直读 |
| `report.html` | 报告投影（富视图） | `render.py`（ECharts/chart.js） | 与原前端同等图表保真 |
| `report.state.json` | **增量隐藏状态** | `render.py` | 存上次 outline 快照 + `report_data` + `summaries`，供下次 diff |

三条铁律（违反任何一条都会导致视图漂移或谎报）：

1. **唯一真源**：结构化数据（`outline.json` + `report.state.json` 内的报告数据）是唯一可信源；其余视图每次变更**整体重投影**，禁止手编 md/yaml/html。
2. **编辑入口永远是结构**：沿用「修改报告 = 修改大纲」——
   `用户改 → 改 outline_tree → 重生成大纲三视图 → 增量重渲报告 → 重生成报告两视图`。
   不存在「直接编辑 report.md / report.html」的路径。
3. **增量靠持久化状态**：SSE 时代靠前端内存 + session 文件记住「查过哪些指标、结果、哪些总结生成过」（`report_executor._persist_report_data` 写 `report_data`/`report_summaries`）。文件态无常驻前端，故落 `report.state.json`；下次修改 diff 它，复用未变子树、只重渲改动部分（对应现有 `cached_names`/`cached_summary_ids`）。

> **报告两视图为何成立**：`report.md` 用 Mermaid/表格保证可移植与可 diff；`report.html` 内嵌真实图表库，恢复原前端的全保真图表。二者都是同一份报告数据的投影，互不为源。这也取代了早期「只产单 markdown、图表退化」的方案。

> **写入原子性**：一次变更涉及多文件，应「写临时文件 + rename」成组提交，避免出现半更新的视图集合。

---

## 4. 总体架构

采用「**脚本自包含 + 数据外置到平行 `reference/`**」方案：流水线逻辑随 skill 走，重数据与共享库放在平行 `reference/`，重计算服务（LLM/Embedding/SQL）通过 env 指向外部端点。

### 4.1 分发包目录

分发单元（一个 git 仓库 / 一个 zip）：

```
report-skill-package/
├── skills/
│   └── generate-report/
│       ├── SKILL.md                     # frontmatter + 一句话→报告 SOP
│       └── scripts/
│           ├── generate_report.py       # 唯一入口（编排整条流水线）
│           └── pipeline/                # 拆分的流水线步骤（薄封装）
│               ├── retrieve.py          # 检索：embed → FAISS → 候选树
│               ├── build_outline.py     # 锚点选择 + 子树展开 + 自动修剪
│               └── render.py            # 遍历大纲 → 查询 → 渲染 Markdown
├── reference/                           # ← 与 skills/ 平行，存数据与共享库
│   ├── knowledge/
│   │   ├── knowledge_nodes.json
│   │   ├── knowledge_relations.json
│   │   └── evaluation_mock.json         # 指标 SQL 定义 + 离线 mock 数据
│   ├── index/
│   │   ├── faiss.index
│   │   └── faiss_id_map.json
│   └── lib/                             # 从 backend 提取的可移植共享代码
│       ├── loader.py  retriever.py  patcher.py
│       ├── outline_utils.py  subtree.py
│       └── services/                    # llm / embedding / faiss / sql 客户端（去 backend 耦合版）
├── install.sh                           # 一键安装
├── requirements.txt                     # numpy / faiss-cpu / httpx 等
└── README.md
```

### 4.2 为什么数据放平行 `reference/` 而不是 skill 内

- **关注点分离**：`SKILL.md` + `scripts/` 是「怎么做」，`reference/` 是「拿什么做」，互不污染。
- **可复用**：未来若有第二个 skill（如「专家知识沉淀」）也能共享同一份 `reference/`。
- **可替换数据**：换知识库 = 换 `reference/`，skill 不动。
- **符合 Agent Skills 渐进式披露精神**：`SKILL.md` 保持轻，重资源外挂。

---

## 5. 数据与路径契约

脚本如何找到平行的 `reference/`？按优先级：

1. **环境变量优先**：`REPORT_REFERENCE_DIR`（install.sh 写入），最可靠，跨 agent 通用。
2. **相对路径回退**：脚本用 `__file__` 求自身位置，向上回溯找同级或上级的 `reference/`
   （`skills/generate-report/scripts/ → ../../../reference`）。
3. 都找不到 → 明确报错，提示运行 install.sh 或设置 `REPORT_REFERENCE_DIR`。

> opencode 通常把 skill 装在 `.opencode/skills/<name>/`。install.sh 需保证 `reference/`
> 落到一个可解析位置（建议 `.opencode/reference/`，并导出 `REPORT_REFERENCE_DIR`），
> 由此固定「skills 与 reference 平行」的约定。

---

## 6. 一句话 → 报告 流水线

整条链在 `generate_report.py` 内一次跑完（无人机多轮确认），各步骤映射到现有代码以便复用：

| 步骤 | 做什么 | 复用现有 |
|------|--------|----------|
| 0. 解析入参 | 接收用户一句话 + `--out-dir` | 新增 |
| 1. 检索 | embed → FAISS → 候选节点树 | `_lib/retriever.search_graph_tree`（`retriever.py:134`） |
| 2. 选锚点 | 单节点直选 / 多节点选共同祖先（可调 LLM 决策） | SKILL.md 锚点选择原则 → 脚本化 |
| 3. 展开大纲 | 从锚点展开子树为 outline_tree | `_lib/build_outline_from_anchor` |
| 4. 自动修剪 | 删除与需求无关分支（一句话场景由 LLM 一次判定，不等用户） | `_lib/patcher` + LLM |
| 5. 写大纲三视图 | `outline_tree` → json/md/yaml 落盘 | `outline_utils.to_clean_json/to_markdown/to_yaml` |
| 6. 渲染报告 | 遍历 outline_tree，查 SQL、判 condition、生成总结 | `report_executor.run_report`（`report_executor.py:37`） |
| 7. 写报告两视图 + 状态 | 事件 → report.md / report.html + report.state.json | **新增 render sink，替代 SSE** |

> 修改场景从步骤 4 进入：改 `outline.json`（真源）→ 重做步骤 5 → 步骤 6 带 `report.state.json`
> 做增量 → 重做步骤 7。详见 §6.3。

### 6.1 大纲三视图

`outline_tree`（步骤 3/4 的产物）即真源，三视图全部由它投影，每次变更整体重写：

```python
# 落大纲三视图
clean = to_clean_json(outline_tree)
write_atomic("outline.json", json.dumps(clean, ensure_ascii=False, indent=2))
write_atomic("outline.md",   to_markdown(clean))
write_atomic("outline.yaml", to_yaml(clean))
```

### 6.2 报告两视图：渲染从「推事件」改为「写文件」

现在 `report_executor.run_report(outline_tree, on_event=...)` 通过回调把
`report_metric` / `report_summary` / `report_skip` 事件推给前端
（`report_executor.py:37`）。它的回调是注入的——**这正是解耦点**。

移植时提供一个**收集型 on_event**，把事件累积进内存，最后同时投影出 md 与 html，并写增量状态：

```python
# render.py（伪代码）
metrics, summaries, skipped = {}, {}, set()
def on_event(ev):
    t = ev["type"]
    if t == "report_metric":   metrics[ev["name"]] = ev           # 含 rows/render_type/colX/colY
    elif t == "report_summary": summaries[ev["node_id"]] = ev["chunk"]
    elif t == "report_skip":    skipped.add(ev["node_id"])

# 增量：带上次状态，未变子树复用、只重渲改动部分（见 §6.3）
prev = load_state("report.state.json")
run_report(outline_tree, on_event=on_event, session_id="",
           cached_names=prev.cached_names, cached_summary_ids=prev.cached_summary_ids)

write_atomic("report.md",   assemble_md(outline_tree, metrics, summaries, skipped))   # Mermaid+表格
write_atomic("report.html", assemble_html(outline_tree, metrics, summaries, skipped)) # ECharts
write_atomic("report.state.json", dump_state(outline_tree, metrics, summaries))
```

> `run_report` 已把 LLM/SQL 客户端做成可独立创建（`DeApiClient` / `LLMService.from_env` /
> `SqlExecutor`），env 配好端点即可在 skill 进程内直接运行，无需 backend 主进程。

**两视图的图表策略**（同一份 `rows / render_type / colX / colY`，两种投影）：

| 视图 | BAR/LINE/PIE | TABLE / 兜底 | 特点 |
|------|--------------|--------------|------|
| `report.md` | **Mermaid**（`pie` / `xychart-beta`） | Markdown 表格 | 纯文本、可 diff、opencode 直读 |
| `report.html` | **ECharts/chart.js**（与原前端同保真） | HTML 表格 | 富渲染，离线自包含（图表库内联或 CDN） |

`render_type → 图表`的映射逻辑集中在 `render.py`，md/html 各一套 emitter，数据入口共用。

### 6.3 增量修改（report.state.json）

`report.state.json` 是文件态对原 session 文件 `report_data`/`report_summaries` 的等价物
（`report_executor._persist_report_data`，`report_executor.py:63`），结构：

```jsonc
{
  "outline_tree": { ... },              // 上次的大纲快照，用于 diff
  "report_data":  { "指标名": [rows] }, // 上次各指标查询结果
  "summaries":    { "node_id": "文本" } // 上次各节点总结
}
```

修改时 `render.py` 用它推出 `cached_names`（数据可复用的指标）与 `cached_summary_ids`
（总结可复用的节点）传给 `run_report`，未变子树不再查 SQL / 不再调 LLM，只重渲改动部分——
直接复用 `report_executor` 既有的缓存语义，无需新增执行逻辑。

---

## 7. SKILL.md 设计

`generate-report/SKILL.md` 对 opencode / Claude 通用：

```yaml
---
name: generate-report
description: >
  一句话生成传送网络分析报告。当用户用自然语言描述网络分析/评估/规划需求
  （超千兆升级、城域网规划、OTN/fgOTN/OSU、覆盖与容量评估、政企专线承载等）时使用。
  自动检索知识库、构建并修剪大纲、执行指标查询，产出大纲三视图与报告两视图文件。
  用户要求修改报告时同样用本 skill：改大纲真源 → 三视图同步、报告增量重渲。
  不适用于与传送网络无关的一般对话。
---
```

正文 SOP 极简——「一句话」体验的核心是**只需一步**：

```markdown
# 一句话生成报告

## 首次生成
用户描述网络分析需求时，直接运行入口脚本，全链路自动完成：

    python3 $SKILL_DIR/scripts/generate_report.py "<用户需求原话>" --out-dir ./out

脚本内部依次完成：检索 → 选锚点 → 展开大纲 → 自动修剪 → 写大纲三视图 → 执行查询 → 写报告两视图。
产物：out/outline.{json,md,yaml}、out/report.{md,html}、out/report.state.json。
完成后告知用户报告已生成（指向 out/report.md 与 out/report.html），用 1-2 句概述主要章节。

## 修改报告（= 修改大纲）
报告/大纲已存在时，用户的后续输入优先理解为修改指令。改的是大纲真源，绝不直接编辑渲染文件：

    python3 $SKILL_DIR/scripts/modify_report.py --out-dir ./out --ops '<结构化修改>'

脚本改 outline.json → 大纲三视图同步重写 → 借 report.state.json 增量重渲报告两视图
（未变章节复用、不重复查 SQL/调 LLM）。

## 失败处理
- 检索无命中（not_found）→ 如实告知知识库未覆盖该场景，不要编造、不要重试。
- 服务不可达（LLM/Embedding/SQL）→ 提示用户检查 .env 中的端点配置。

## 不要做
- 不要在对话里粘贴大纲或报告全文；产物在文件里。
- 不要手编 outline.md/yaml 或 report.md/html——它们是投影，改动只走 outline.json。
- 用户只是闲聊时，不调用脚本。
```

> 进阶（结构化修改操作清单）可作为 Level 2 文档（`reference.md`）渐进披露，保持主 SOP 聚焦「一句话」。

---

## 8. 入口脚本接口

```
generate_report.py "<需求原话>" [--out-dir ./out]
                    [--reference-dir DIR] [--no-mock] [--top-k N]
modify_report.py    --out-dir ./out --ops '<json>'   # 改大纲真源 → 三视图同步 + 报告增量重渲

退出码：0 成功 / 2 检索无命中 / 3 服务不可达 / 1 其他错误
stdout：成功时打印产物路径清单 + 章节概览（JSON 或纯文本，供 agent 转述）
产物目录：outline.{json,md,yaml}、report.{md,html}、report.state.json（见 §3.1）
```

设计要点：
- **幂等**：同输入同输出，便于测试与缓存。
- **环境变量**：`LLM_BASE_URL` / `EMBEDDING_BASE_URL` / `EMBEDDING_DIM` / SQL API 配置 / `REPORT_REFERENCE_DIR` / `FORCE_MOCK`。
- **离线可跑**：`FORCE_MOCK=true` 时用 `reference/knowledge/evaluation_mock.json` 的离线数据，无需 SQL API，方便演示与 CI。

---

## 9. 一键安装机制

### 9.1 opencode 的 skill 发现路径

opencode 会扫描（项目级优先，其次全局）：

```
.opencode/skills/<name>/SKILL.md      # 项目级（主）
.claude/skills/<name>/SKILL.md        # 兼容 Claude 约定
~/.config/opencode/skills/<name>/SKILL.md   # 全局
~/.claude/skills/<name>/SKILL.md            # 全局兼容
```

格式即 Anthropic Agent Skills：一个文件夹 + 一个带 YAML frontmatter 的 `SKILL.md`。

### 9.2 安装方式

**方式一 · 直接 clone（最简）**

```bash
git clone <repo> .opencode/skills/generate-report           # skill 本体
# reference/ 随仓库一起，install.sh 负责摆位 + 写 env
```

**方式二 · install.sh（推荐，处理平行 reference 与 env）**

```bash
curl -fsSL <repo>/install.sh | bash
# 行为：
#   1. 拷贝 skills/generate-report → <target>/.opencode/skills/
#   2. 拷贝 reference/            → <target>/.opencode/reference/   （与 skills 平行）
#   3. 写入/追加 REPORT_REFERENCE_DIR 等 env 到 shell 或 opencode 配置
#   4. pip install -r requirements.txt
#   5. 提示用户填写 LLM_BASE_URL / EMBEDDING_BASE_URL / SQL API
```

**方式三 · marketplace（可选远期）**：发布到 opencode skills 市场，支持 `opencode skill add ...`。

---

## 10. 外部服务与配置

skill 不打包重计算服务，仅通过 env 指向：

| 变量 | 用途 | 必需 |
|------|------|------|
| `LLM_BASE_URL` / `LLM_MODEL` | 大纲修剪、condition 判定、章节总结 | 是 |
| `EMBEDDING_BASE_URL` / `EMBEDDING_DIM` | 问题向量化（检索） | 是 |
| SQL API（base_url/operator/user_id…） | 指标实时查询 | 否（可 `FORCE_MOCK`） |
| `REPORT_REFERENCE_DIR` | 平行 reference 定位 | install.sh 写入 |
| `FORCE_MOCK` | 全程用离线 mock 数据 | 否 |

---

## 11. 与现有 backend 的关系（避免双份维护）

`reference/lib/` 不应是手抄副本。三种维护策略：

1. **构建脚本同步（推荐起步）**：写一个 `scripts/build_skill_package.py`，从 `backend/skills/_lib`、`backend/services`、`backend/expert_knowledge`、FAISS 索引**抽取**并改写 import，生成分发包。单一真源仍是 backend。
2. **抽成独立 pip 包**：把可移植内核（retriever/patcher/outline_utils/services 客户端）发布为 `report-core`，skill 与 backend 都依赖它。最干净，工作量最大。
3. **git subtree / submodule**：把 `_lib` 作为子树双向同步。中间方案。

> 现有代码已对解耦友好：`report_executor` 用注入式 `on_event`，`LLMService.from_env()` /
> `DeApiClient` / `SqlExecutor` 均可独立实例化。主要改造量在「去除对 `backend` 根目录
> `sys.path` 的硬依赖」与「新增文件渲染 sink」。

---

## 12. 分阶段实施计划

| 阶段 | 内容 | 产出 |
|------|------|------|
| P0 | 抽取可移植内核到 `reference/lib`，去 backend sys.path 耦合 | 内核可独立 import |
| P1 | 实现 `render.py` 文件渲染 sink（事件 → Markdown，含 Mermaid/表格） | 给定 outline_tree 能出 report.md |
| P2 | 实现 `generate_report.py` 端到端编排 + `pipeline/*` | 一句话 → report.md 跑通（FORCE_MOCK 离线） |
| P3 | 写 `SKILL.md` + install.sh + requirements + README | 可一键装入 opencode |
| P4 | 接真实 LLM/Embedding/SQL 端点联调；CI 用 mock 冒烟 | 线上可用 + 回归保障 |
| P5 | （可选）build_skill_package.py 自动从 backend 生成分发包 | 单真源、可持续维护 |

---

## 13. 待决问题与风险

1. **图表保真度**：`report.md` 的 Mermaid 覆盖大部分 BAR/PIE/LINE，复杂图型（多系列/双轴）可能退化为表格；`report.html` 用 ECharts 全保真。md 退化是否接受？
2. **HTML 自包含 vs CDN**：`report.html` 内嵌图表库（离线可用、体积大）还是引 CDN（轻、需联网）？
3. **FAISS 索引随包分发**：索引文件较大且与 Embedding 模型绑定；换模型需重建。随包还是首次运行时构建？
4. **一句话 vs 可控性**：跳过多轮确认提升体验，但用户失去大纲干预机会。`modify_report.py` 是否够，还是需 `--interactive`？
5. **维护双源**：`reference/lib` 与 backend 的同步策略需尽早定（见 §11），否则会漂移。
6. **SQL API 不可移植**：政企 SQL 执行 API 属内网服务；对外演示场景默认 `FORCE_MOCK`。

---

## 附：端到端数据流（目标态）

```
用户一句话
   │
   ▼ generate_report.py
   ├─ retrieve.py ──→ Embedding 服务 + reference/index/faiss ──→ 候选节点树
   ├─ build_outline.py ──→ reference/knowledge/*.json ──→ outline_tree（真源）
   ├─ (LLM) 自动修剪无关分支
   ├─ 写大纲三视图 ──→ outline.json / outline.md / outline.yaml
   ├─ render.py ──→ run_report(on_event=收集器, cached_* 来自 report.state.json)
   │                   ├─ SQL API / reference/knowledge/evaluation_mock.json（FORCE_MOCK）
   │                   └─ LLM：condition 判定 + 章节总结
   └─ 写报告两视图 + 状态 ──→ report.md（Mermaid+表格） / report.html（ECharts） / report.state.json
   ▼
opencode 打开 report.md / report.html

后续修改：
modify_report.py --ops ... ──→ 改 outline.json（真源）
   ├─ 大纲三视图同步重写
   └─ render.py + report.state.json 增量 ──→ 仅改动章节重渲，report 两视图更新
```
