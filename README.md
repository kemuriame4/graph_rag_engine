# 多源异构文档智能解析与自适应 Graph-RAG 引擎

> 面向企业知识库与跨境电商场景：**多源文档解析 → 语义切片 → Dense / BM25 / Graph 三路混合召回 → Self-RAG 反思纠错 → Ragas 评测与全链路可观测**。

本仓库实现默认**零外部依赖即可离线跑通**（自动降级为本地向量库、纯 Python BM25、自研 StateGraph 与离线 LLM）；安装可选依赖后可切换 Qdrant、Elasticsearch、Cross-Encoder、Celery 与官方 Ragas。

***

## 1. 核心能力

| 模块       | 实现要点                                                                                                          |
| -------- | ------------------------------------------------------------------------------------------------------------- |
| 多源解析     | PDF（PyMuPDF 优先 / pypdf 降级）、DOCX（python-docx 或 zip+XML 降级）、Markdown、HTML、TXT/CSV；标题层级、表格 Markdown 化、题注识别、页码回填  |
| 动态语义切片   | 按 Semantic Outline 构造 section；表格/公式/图保持原子性；超长段落按句切分；相邻切片句子级 overlap；注入文档名、章节路径、摘要、关键词、实体                      |
| 三路混合召回   | Dense（Qdrant / 本地 NumPy）、Sparse（Elasticsearch BM25 / 本地 BM25）、Graph（NetworkX 实体关系图，1–2 hop）                   |
| 融合与重排    | RRF（Reciprocal Rank Fusion）+ Cross-Encoder（BGE-Reranker）；离线时降级为词法/通道加权启发式重排                                   |
| Self-RAG | 自适应直答、HyDE、Query Rewrite、Is-Rel / Is-Sup / Is-Use 三维反思、严格重生成、Usefulness 补充检索；自研 `StateGraph` 与 LangGraph 语义对齐 |
| 评测       | 内置 Ragas 口径四项指标（Faithfulness / Answer Relevance / Context Recall / Context Precision）；安装 `ragas` 后自动切换官方后端    |
| 可观测性     | Trace 步骤、Token 统计、P50/P95 延迟、JSONL 持久化；WebSocket 推送节点事件；零构建前端看板                                               |
| 工程化      | FastAPI、租户隔离、线程池/Celery 双入库后端、Docker Compose 编排、pytest 测试                                                     |

***

## 2. 系统架构

```text
文档上传 (PDF / DOCX / MD / HTML / TXT)
        │
        ▼
┌────────────────────────────┐
│ 解析层 DocumentParser       │ 版面块 / 表格 / 题注 / 页码
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│ 语义切片 SemanticChunker    │ 章节路径 + 摘要 + 关键词 + 实体
└──────────────┬─────────────┘
               ▼
┌──────────────────────────────────────────────┐
│ 索引层 SearchIndex                            │
│  ├─ Dense  : Qdrant / InMemoryVectorStore    │
│  ├─ Sparse : Elasticsearch / InMemoryBM25    │
│  └─ Graph  : NetworkX 实体-关系图              │
└──────────────┬───────────────────────────────┘
               ▼
┌────────────────────────────┐
│ 检索层 HybridRetriever      │ Dense + Sparse + Graph → RRF → Rerank
└──────────────┬─────────────┘
               ▼
┌────────────────────────────┐
│ Self-RAG 引擎               │ analyze → retrieve → generate
│                            │   → Is-Sup → Is-Use → supplement/finalize
└──────────────┬─────────────┘
               ▼
┌──────────────────────────────────────────────┐
│ 应用层 FastAPI + WebSocket + 前端看板          │
│ 评测层 Evaluator + Trace/Tracer               │
└──────────────────────────────────────────────┘
```

***

## 3. 三档运行模式

| 模式       | 依赖                                                           | 行为                                                                                |
| -------- | ------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| 离线降级（默认） | `requirements.txt`                                           | Hashing 向量 + 本地 NumPy 向量库 + 纯 Python BM25 + 启发式重排 + 离线抽取式 LLM；无需 API Key / 数据库    |
| 生产存储     | 额外安装 `qdrant-client`、`elasticsearch`、`sentence-transformers` | 设置 `QDRANT_URL`、`ES_URL`、`EMBEDDING_BACKEND=sentence-transformers` 后自动切换真实存储与向量模型 |
| 全量增强     | `requirements-full.txt`                                      | 追加 PyMuPDF、python-docx、Cross-Encoder 重排、Celery/Redis 异步入库、可选 Ragas 官方评测           |

> 所有适配器均采用 `auto` 探测：外部服务不可用或依赖缺失时自动回退，不会导致启动失败。

***

## 4. 快速开始

### 4.1 离线模式（推荐首次体验）

```bash
python -m pip install -r requirements.txt

# 端到端演示：示例入库 → 三个问题 → 可选评测
python scripts/demo.py
python scripts/demo.py --eval

# 启动 API 与看板
uvicorn app.main:app --reload --port 8000
```

打开 <http://localhost:8000> 进入可观测性看板；接口文档见 <http://localhost:8000/docs>。

### 4.2 接入真实模型与存储

```bash
python -m pip install -r requirements-full.txt
cp .env.example .env            # Windows: copy .env.example .env
```

`.env` 关键配置：

```ini
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.openai.com/v1
LLM_CHAT_MODEL=gpt-4o-mini
EMBEDDING_BACKEND=sentence-transformers
ST_MODEL=BAAI/bge-m3
QDRANT_URL=http://localhost:6333
ES_URL=http://localhost:9200
RERANKER_BACKEND=cross-encoder
```

### 4.3 Docker Compose

```bash
docker compose up -d --build
```

编排包含 `api`、`worker`（Celery）、`qdrant`、`elasticsearch`、`redis`。

***

## 5. API 一览

前缀 `/api/v1`：

| 方法     | 路径                              | 说明                                            |
| ------ | ------------------------------- | --------------------------------------------- |
| GET    | `/health`                       | 健康检查与当前后端（LLM / Embedding / 向量库 / 稀疏索引 / 评测器） |
| GET    | `/stats`                        | 文档、切片、实体、任务、Trace 统计                          |
| POST   | `/documents`                    | 上传文档（multipart，支持 tenant）                     |
| GET    | `/documents`                    | 文档列表                                          |
| GET    | `/documents/{doc_id}`           | 文档详情                                          |
| GET    | `/documents/{doc_id}/chunks`    | 文档切片（含章节、摘要、关键词、实体）                           |
| GET    | `/documents/{doc_id}/file`      | 原始文件下载                                        |
| DELETE | `/documents/{doc_id}`           | 删除文档并同步清理三路索引                                 |
| GET    | `/jobs`、`/jobs/{job_id}`        | 入库任务进度                                        |
| POST   | `/query`                        | RAG 问答（返回答案、引用、反思分数、召回明细）                     |
| WS     | `/ws/query`                     | 流式推送节点事件与最终结果                                 |
| GET    | `/graph`、`/graph/stats`         | 实体子图与图谱统计                                     |
| GET    | `/traces`、`/traces/{trace_id}`  | Trace 列表与详情                                   |
| GET    | `/metrics`                      | 延迟分位、Token、错误计数与索引统计                          |
| POST   | `/eval/run`、`GET /eval/reports` | 评测执行与报告列表                                     |
| POST   | `/reindex`、`/samples/load`      | 重建索引、一键导入内置示例                                 |

查询示例：

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Battery-X1 的循环寿命和充电时间是多少？","top_k":8}'
```

***

## 6. 目录结构

```text
graph_rag_engine/
├── app/
│   ├── main.py                 # FastAPI 工厂与全部路由
│   └── static/                 # 零构建前端看板（HTML/CSS/JS）
├── graphrag/
│   ├── config.py               # 环境变量配置
│   ├── models.py               # 解析→切片→检索→反思→评测 契约模型
│   ├── text.py                 # 中英混合分词、句子切分、跨语言术语扩展
│   ├── parsing/                # PDF / DOCX / Markdown / HTML / TXT 解析
│   ├── chunking/               # 语义大纲切片与 embedding_text 注入
│   ├── indexing/               # 向量库、BM25、实体抽取、NetworkX 图谱
│   ├── retrieval/              # 三路召回、RRF、重排
│   ├── selfrag/                # StateGraph、Self-RAG 引擎、LLM 适配
│   ├── evaluation/             # Ragas 口径评测
│   ├── observability/          # Trace 采集与持久化
│   ├── tasks/                  # Celery 任务
│   ├── ingestion.py            # 入库编排（线程池 / Celery）
│   └── bootstrap.py            # 进程内依赖装配
├── samples/                    # 示例语料与评测集
├── scripts/demo.py             # 端到端演示
├── tests/                      # pytest 测试
├── requirements.txt            # 离线最小依赖
├── requirements-full.txt       # 全量增强依赖
├── Dockerfile / docker-compose.yml
└── PRD_plan.md                 # 需求与实施计划
```

***

## 7. 关键配置

| 环境变量                                                         | 默认值                      | 说明                                             |
| ------------------------------------------------------------ | ------------------------ | ---------------------------------------------- |
| `DATA_DIR`                                                   | `./data`                 | 索引、上传、Trace、评测数据目录                             |
| `EMBEDDING_BACKEND`                                          | `auto`                   | `hashing` / `openai` / `sentence-transformers` |
| `EMBEDDING_DIM`                                              | `256`                    | hashing 模式维度                                   |
| `TOP_K_DENSE` / `TOP_K_SPARSE` / `TOP_K_GRAPH`               | `20`                     | 三路召回条数                                         |
| `RRF_K`                                                      | `60`                     | RRF 平滑常数                                       |
| `RERANK_TOP_N`                                               | `8`                      | 重排后保留条数                                        |
| `RERANKER_BACKEND`                                           | `auto`                   | `heuristic` / `cross-encoder`                  |
| `GRAPH_MAX_HOPS`                                             | `2`                      | 图谱多跳上限                                         |
| `CHUNK_MAX_TOKENS` / `CHUNK_OVERLAP_SENTENCES`               | `450` / `1`              | 切片长度与句子级重叠                                     |
| `ADAPTIVE_RETRIEVAL`                                         | `true`                   | 简单问题直答，降低检索开销                                  |
| `ENABLE_HYDE`                                                | `true`                   | 抽象/短查询启用 HyDE                                  |
| `IS_REL_THRESHOLD` / `IS_SUP_THRESHOLD` / `IS_USE_THRESHOLD` | `0.22` / `0.72` / `0.60` | 三维反思阈值                                         |
| `MAX_REFLECTION_LOOPS`                                       | `2`                      | 反思回环上限                                         |
| `JOB_BACKEND`                                                | `thread`                 | `thread` / `celery`                            |
| `MAX_UPLOAD_MB`                                              | `200`                    | 单文件上传上限                                        |

完整清单见 [.env.example](.env.example)。

***

## 8. 评测与测试

```bash
python -m pytest                       # 解析/切片、检索、Self-RAG、评测、API 共 14 项

python scripts/demo.py --eval          # 内置评测集（samples/eval_dataset.jsonl）
```

内置评测集为 8 条跨境电商/产品规格问题。离线演示环境中的一次实测结果（3 篇示例文档、27 个切片、76 个实体、102 条关系）：

| 指标                | 数值      |
| ----------------- | ------- |
| Faithfulness      | 0.75    |
| Answer Relevance  | 0.46    |
| Context Recall    | 0.88    |
| Context Precision | 0.25    |
| 平均延迟              | \~17 ms |
| P95 延迟            | \~24 ms |

> 上述数据来自离线降级模式（Hashing Embedding + 启发式重排），仅用于回归对比，不代表接入 BGE-M3 / BGE-Reranker 后的生产表现。

***

## 9. 设计取舍与已知限制

- **离线优先**：核心链路不依赖外部服务，便于本地验证与 CI；生产环境通过环境变量切换真实组件，而不是维护两套代码。
- **自研 StateGraph**：与 LangGraph 的节点/条件边/递归上限语义对齐，避免强依赖；接入 LangGraph 时节点函数可直接复用。
- **租户隔离**：向量库、BM25、图谱查询均带 `tenant` 过滤。
- **解析限制**：当前 PDF 走文本层提取，未集成 OCR/版面视觉模型；扫描件、复杂合并单元格表格的还原精度有限。
- **多模态未实现**：PRD 中的 CLIP/ColPali 图文联合索引属于后续扩展；当前跨语言检索依赖多语言 Embedding，离线模式仅提供内置术语表扩展。
- **评测精度**：Context Precision 在离线 Hashing 模式下偏低，属于预期；接入真实 Embedding/Reranker 与 Ragas 官方后端后更具参考价值。
- **安全与运维**：未内置鉴权、限流与分布式索引一致性方案，需在生产网关层补齐。

