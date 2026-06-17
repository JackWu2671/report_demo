# 索引构建与知识检索

## 一、涉及的组件是什么

| 组件 | 是什么 | 做什么 |
|------|--------|--------|
| `reference/node.json` | 知识库节点数据 | 存储所有知识节点（L1~L5），每个节点有 id、name、keywords 等字段 |
| `reference/relation.json` | 知识库关系数据 | 存储节点之间的父子关系（parent → child） |
| `EmbeddingService` | 向量化服务客户端 | 把文本（节点名称）发给 Embedding 模型，换回一串数字（向量） |
| `FAISSService` | 向量索引服务 | 存储和检索向量；给一个问题的向量，返回最相似的节点 |
| `data/faiss.index` | 二进制索引文件 | FAISS 把所有节点向量压缩存在这里，加载后可极速检索 |
| `data/faiss_id_map.json` | 索引映射文件 | 记录索引里每个位置对应哪个节点（FAISS 内部只存数字，不存名字） |
| `scripts/build_index.py` | 手动构建脚本 | 手动跑一次，把 node.json 转成 FAISS 索引文件 |
| `skills/_lib/loader.py` | 启动加载器 | 程序启动时加载索引；发现索引不存在时自动触发构建 |
| `skills/_lib/retriever.py` | 检索器 | 接收用户问题，完成"向量化 → 检索 → 补全路径"的完整流程 |

---

## 二、什么是向量 / Embedding

Embedding 是把文本变成一串数字的过程。

```
"AEC 覆盖用户数"  →  [0.12, -0.34, 0.87, ..., 0.05]  (1024 个数字)
"信号覆盖率"      →  [0.11, -0.31, 0.85, ..., 0.06]  (1024 个数字)
"设备故障率"      →  [-0.45, 0.22, -0.13, ..., 0.71] (1024 个数字)
```

语义相近的词，向量也相近（数字模式相似）。这样就能通过比较数字来找到"意思相近的节点"，而不只是关键词匹配。

---

## 三、FAISS 是什么

FAISS 是 Facebook 开源的向量相似度搜索库。

作用：给你 N 个向量，存进去；然后给一个查询向量，快速找出最相似的 K 个。

本项目使用的是 `IndexFlatIP`（内积索引），配合 L2 归一化后等价于**余弦相似度**检索。相似度范围 0~1，越高越相关。

---

## 四、索引是怎么构建的

### 输入

```
reference/node.json
[
  { "id": "L1_001", "name": "无线网络评估", "keywords": ["覆盖", "信号"], "level": 1 },
  { "id": "L2_003", "name": "AEC 覆盖能力", "keywords": ["AEC", "覆盖率"], "level": 2 },
  ...
]
```

### 构建步骤

```
① 读取 node.json，取出所有节点
         ↓
② 拼接每个节点的检索文本：name + keywords
   "AEC 覆盖能力 AEC 覆盖率"
         ↓
③ 批量发给 Embedding 服务（每批 32 个，避免超时）
   → 每条文本 → 1024 维 float32 向量
   → 对每个向量做 L2 归一化（长度变成 1）
         ↓
④ 用所有向量构建 FAISS 索引（IndexFlatIP）
         ↓
⑤ 保存两个文件：
   data/faiss.index       ← 向量数据（二进制，不能直接读）
   data/faiss_id_map.json ← 位置 → 节点映射 [{id, name, level, ...}, ...]
```

FAISS 内部只存位置（0, 1, 2, ...），不存名字。`faiss_id_map.json` 记录"第 0 号是 L1_001，第 1 号是 L2_003..."，检索结果返回位置后靠它翻译成节点信息。

### 手动构建

```bash
cd backend
python scripts/build_index.py
```

输出：
```
读取到 238 个知识节点
开始获取 238 个节点的 Embedding...
Embedding: 32/238
Embedding: 64/238
...
✅ 索引构建完成，共 238 条向量
   → backend/data/faiss.index
   → backend/data/faiss_id_map.json
```

---

## 五、FAISS 索引文件不存在时自动构建

`data/faiss.index` 是二进制文件，**不提交到 git**（`.gitignore` 里排除了 `backend/data/`）。

每次新克隆仓库或清空 data 目录后，索引文件就不存在了。系统通过两层机制保证索引始终可用。

### 检测逻辑（loader.py）

`_build_index_if_missing()` 是 async 函数，每次 `load_resources()` 被调用时都会先执行检查：

```python
# skills/_lib/loader.py

async def _build_index_if_missing() -> None:
    if os.path.exists(index_path) and os.path.exists(id_map_path):
        return  # 已存在，直接跳过

    # 缺失 → 调 Embedding 服务批量向量化，构建并保存
    embeddings = await emb_svc.get_embeddings_batch(texts, batch_size=32)
    faiss_svc.build(nodes, embeddings)
    faiss_svc.save(index_path, id_map_path)

async def load_resources():
    await _build_index_if_missing()  # ← 先检查，缺失则构建
    faiss_svc = FAISSService(...)
    await faiss_svc.load(...)
    ...
```

### 触发时机：服务器启动时（api_server.py）

FastAPI 提供 `lifespan` 钩子，在服务器启动完成后立即执行初始化逻辑。`api_server.py` 在这里调用 `load_resources()`，确保索引在第一个用户请求到来之前就已就绪：

```python
# api_server.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("[Startup] 检查 FAISS 索引…")
        await load_resources()
        logger.info("[Startup] FAISS 索引就绪")
    except Exception as e:
        logger.warning("[Startup] FAISS 索引初始化失败（不影响启动）: %s", e)
    yield

app = FastAPI(lifespan=lifespan)
```

启动日志示例：

```
# 索引已存在
[Startup] 检查 FAISS 索引…
[Startup] FAISS 索引就绪

# 索引不存在，自动构建
[Startup] 检查 FAISS 索引…
[Step 1] FAISS 索引不存在，开始自动构建…
Embedding: 32/238
Embedding: 64/238
...
[Step 1] FAISS 索引自动构建完成，共 238 条向量
[Startup] FAISS 索引就绪
```

如果 Embedding 服务不可用导致构建失败，只打 warning、不崩服务器，其他功能正常使用。

---

## 六、检索流程（Step 2 → Step 4）

用户发出问题后，检索分四步走。

### Step 2：问题向量化（embed_query）

```python
# retriever.py
vec = await emb_svc.get_embedding(question)
# question: "评估企业专线质量"
# vec:      shape (1, 1024) 的 float32 数组
```

问题文本经过 Embedding 服务，变成和知识节点"同一语言"的向量，才能比较相似度。

### Step 3：FAISS 检索相似节点（search_nodes）

```python
hits = faiss_svc.search(query_embedding, top_k=10, threshold=0.3)
```

FAISS 拿问题向量和索引里所有节点向量做内积运算（因为都归一化了，等价于余弦相似度），返回得分最高的若干个节点：

```
命中节点:
  L4 企业专线质量评估 | score=0.872
  L3 专线业务评估     | score=0.754
  L5 专线时延         | score=0.681
  ...
```

- `top_k=10`：最多返回 10 个
- `threshold=0.3`：相似度低于 0.3 的过滤掉（太不相关）

### Step 4：补全祖先路径（build_candidate_paths）

FAISS 命中的节点可能是树中任意层级。为了让 LLM 理解节点的上下文位置，给每个命中节点补全从根到它的完整路径：

```python
# 对每个命中节点，沿 parent_map 一路向上追溯
chain = []
cur = hit["id"]
while cur:
    chain.append(nodes_dict[cur]["name"])
    cur = parent_map.get(cur)
chain.reverse()
path = " > ".join(chain)
```

结果：

```
企业专线质量评估
  path: "IP 承载网评估 > 企业业务质量 > 企业专线质量评估"
  score: 0.872

专线时延
  path: "IP 承载网评估 > 企业业务质量 > 企业专线质量评估 > 专线时延"
  score: 0.681
```

### 组合接口：search_graph_tree

`retriever.py` 还提供了一个更完整的接口，在 Step 2~4 之后，还会：

1. **还原树状结构**：把命中节点和它们的祖先节点拼成一棵树（而不是扁平列表）
2. **补全后代节点**：对每个命中节点，把它在知识图谱里的所有子孙节点也展开进来，防止 LLM 因阈值过滤漏掉相关指标

最终返回的是一个树状 dict 列表，供 LLM 生成报告大纲时使用：

```json
[
  {
    "id": "L1_001",
    "name": "IP 承载网评估",
    "level": 1,
    "hit": false,
    "score": null,
    "children": [
      {
        "id": "L3_012",
        "name": "企业专线质量评估",
        "level": 3,
        "hit": true,
        "score": 0.872,
        "children": [
          { "id": "L5_045", "name": "专线时延", "level": 5, "hit": true, "score": 0.681, "children": [] },
          { "id": "L5_046", "name": "专线丢包率", "level": 5, "hit": false, "score": null, "children": [] }
        ]
      }
    ]
  }
]
```

`hit: true` 表示 FAISS 直接命中的节点，`hit: false` 表示因为是祖先/后代被补全进来的。

---

## 七、相关文件索引

| 文件 | 职责 |
|------|------|
| `reference/node.json` | 知识节点原始数据（索引的输入） |
| `reference/relation.json` | 节点父子关系（构建路径和树状结构用） |
| `data/faiss.index` | FAISS 二进制索引（不提交 git，运行时自动生成） |
| `data/faiss_id_map.json` | 索引位置 → 节点映射（不提交 git，运行时自动生成） |
| `services/embedding_service.py` | Embedding HTTP 客户端，支持单条和批量 |
| `services/faiss_service.py` | FAISS 封装：build / save / load / search |
| `scripts/build_index.py` | 手动构建索引的脚本 |
| `skills/_lib/loader.py` | 启动时加载索引，索引不存在时自动构建 |
| `skills/_lib/retriever.py` | 完整检索流程：embed → search → build_paths → tree |

---

## 八、环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_BASE_URL` | `http://localhost:8001/v1` | Embedding 服务地址 |
| `EMBEDDING_DIM` | `1024` | 向量维度，需与模型一致（bge-m3 是 1024） |
| `FAISS_SCORE_THRESHOLD` | `0.3` | 检索时过滤低分节点的阈值 |
