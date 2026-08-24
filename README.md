# Production RAG

生产级 RAG（Retrieval-Augmented Generation）问答服务：文档摄入 → 检索 → 重排 → 上下文管理 → 生成 → 引用溯源。

## 特性

- **多格式文档摄入**：支持 TXT / HTML / PDF / Word(.docx)，含清洗、分块（固定/递归）、embedding 向量化
- **混合检索**：向量检索（FAISS）+ BM25 关键词检索 + RRF 融合，支持 Cross-Encoder 重排
- **上下文管理**：chunk 去重（span 重叠 + Jaccard）、邻接合并、压缩截断、token 预算控制
- **引用溯源**：答案中的 `[1][2]` 引用自动映射到源 chunk，含文件名和字符偏移
- **流式生成**：基于 DashScope 的真流式 SSE 输出，首字延迟≈模型开始输出时间
- **持久化存储**：MySQL 存储文档/chunks（可选），含连接池和 CASCADE 删除
- **可观测性**：Prometheus 指标采集 + 请求追踪（trace_id）+ 深度健康检查
- **后台任务**：大文档上传/索引重建支持后台异步执行 + 任务状态查询
- **评估体系**：检索指标（Recall/Precision/MRR/NDCG）+ 生成质量评估（Faithfulness/Relevance，LLM-as-Judge）

## 项目架构图

![alt text](design.png)

## 项目结构

```
production-rag/
├── app/
│   ├── api/                  # FastAPI 路由层
│   │   ├── chat.py           #   聊天/问答 API（含 SSE 流式）
│   │   ├── health.py         #   深度健康检查
│   │   └── knowledge.py      #   知识库管理（上传/删除/重建/任务查询）
│   ├── ingestion/            # 文档摄入
│   │   ├── loader/           #   文档加载器（txt/html/pdf/word）
│   │   ├── chunker/          #   分块器（fixed/recursive）
│   │   ├── cleaner/          #   文档清洗
│   │   └── pipeline.py       #   摄入流水线
│   ├── embedding/            # Embedding 模型封装
│   ├── search/               # 检索引擎（vector/bm25/hybrid + RRF）
│   ├── rerank/               # Cross-Encoder 重排器
│   ├── context/              # 上下文管理（builder/compressor/manager）
│   ├── generation/           # LLM 生成（Stub/Qwen + 流式）
│   ├── citation/             # 引用提取
│   ├── rag/                  # RAG 编排（pipeline/service）
│   ├── storage/              # 持久化（MySQL + 文件元数据）
│   ├── evaluation/           # 评估（检索指标 + 生成质量）
│   └── core/                 # 基础设施（config/env/logger/metrics/tracing/task_queue）
├── configs/
│   └── config.yaml           # 主配置文件
├── scripts/                  # 命令行脚本
│   ├── ingest.py             #   文档摄入脚本
│   ├── evaluate.py           #   批量评估（检索 + 生成质量）
│   ├── evaluate_retrieval.py #   检索评估
│   ├── query.py              #   命令行查询
│   └── context_demo.py      #   上下文管理演示
├── tests/                    # 单元测试
├── data/                     # 运行时数据（索引/原始文件）
├── reports/                  # 评估报告输出
├── main.py                   # CLI 交互式入口
├── requirements.txt          # 依赖清单
└── .env                      # 环境变量（需自建，参考下文）
```

## 环境要求

- Python 3.10+
- MySQL 8.0+（可选，`storage.enabled=false` 时不需要）
- 阿里云 DashScope API Key（使用 Qwen LLM 时需要）

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件：

```ini
# DashScope（LLM 生成，不使用 Qwen 可不配）
DASHSCOPE_API_KEY=sk-your-api-key-here
DASHSCOPE_MODEL=qwen-plus

# MySQL（不使用 MySQL 可在 config.yaml 中设 storage.enabled=false）
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=production_rag
```

### 3. 初始化 MySQL（可选）

如启用 MySQL 持久化，需先创建数据库：

```sql
CREATE DATABASE IF NOT EXISTS production_rag CHARACTER SET utf8mb4;
```

表结构会在服务首次启动时自动创建，无需手动建表。

### 4. 摄入文档

将文档放入 `data/raw/` 目录，然后运行摄入脚本构建索引：

```bash
# 递归分块策略（默认，推荐）
python scripts/ingest.py --strategy recursive

# 固定分块策略
python scripts/ingest.py --strategy fixed
```

也可通过 API 上传文档（见下方 API 说明）。

### 5. 启动服务

#### 方式一：API 服务（推荐）

```bash
# 默认端口 8000
python -m app.api

# 或使用 uvicorn
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

# 端口被占用时换端口
uvicorn app.api:app --host 127.0.0.1 --port 8080
```

启动后访问：
- API 文档（Swagger）：http://localhost:8000/docs
- 健康检查：http://localhost:8000/api/health
- Prometheus 指标：http://localhost:8000/metrics

#### 方式二：命令行交互式问答

```bash
# 交互式（默认）
python main.py

# 单次查询
python main.py --query "铁矿近期供需如何？"

# 指定策略和模式
python main.py --strategy recursive --mode hybrid --query "铁矿供需"

# 跳过 rerank（更快，精度略降）
python main.py --no-rerank --query "铁矿供需"
```

## 配置说明

主配置文件 `configs/config.yaml`：

```yaml
embedding:
  model_name: BAAI/bge-small-zh-v1.5  # embedding 模型
  batch_size: 32

chunk:
  chunk_size: 100   # 每块字符数
  overlap: 20       # 重叠字符数

retrieval:
  top_k: 5          # 检索返回数

rerank:
  model_name: BAAI/bge-reranker-base
  candidate_pool: 50  # 重排候选池大小

context:
  max_context_tokens: 4096  # 上下文 token 预算
  reserved_tokens: 1024     # 给 query+prompt+answer 预留
  order_strategy: score     # 排序策略：score/document/interleaved

generation:
  backend: qwen           # stub（占位）/ qwen（DashScope）
  model_name: qwen-turbo  # Qwen 模型名
  temperature: 0.3
  max_tokens: 1024

storage:
  enabled: true    # 是否启用 MySQL 持久化
  pool_size: 5     # 连接池大小
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | RAG 问答（同步） |
| POST | `/api/chat/stream` | RAG 问答（SSE 流式） |
| GET | `/api/health` | 深度健康检查 |
| POST | `/api/knowledge/upload` | 上传文档并构建索引 |
| DELETE | `/api/knowledge/{doc_id}` | 删除文档 |
| POST | `/api/knowledge/rebuild` | 重建索引 |
| GET | `/api/knowledge/status` | 查看索引状态 |
| GET | `/api/knowledge/documents` | 列出所有文档 |
| GET | `/api/knowledge/tasks/{task_id}` | 查询后台任务状态 |
| GET | `/api/knowledge/tasks` | 列出后台任务 |
| GET | `/metrics` | Prometheus 指标 |

### 示例：问答请求

```bash
# 同步问答
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "铁矿供需如何？", "strategy": "recursive", "mode": "hybrid", "use_rerank": true}'

# 流式问答（SSE）
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "铁矿供需如何？"}'
```

### 示例：上传文档

```bash
# 同步上传
curl -X POST http://localhost:8000/api/knowledge/upload \
  -F "file=@data/raw/report.txt" \
  -F "strategy=recursive"

# 后台异步上传（大文件推荐）
curl -X POST "http://localhost:8000/api/knowledge/upload?async_=true" \
  -F "file=@data/raw/report.pdf" \
  -F "strategy=recursive"

# 查询任务状态
curl http://localhost:8000/api/knowledge/tasks/{task_id}
```

## 评估

### 检索评估

```bash
# 评估检索指标（Recall/Precision/MRR/NDCG）
python scripts/evaluate.py --mode retrieval --strategy recursive

# 指定检索模式
python scripts/evaluate_retrieval.py --strategy recursive --mode hybrid --rerank
```

### 生成质量评估

```bash
# 评估 Faithfulness + Relevance，输出 CSV 报告
python scripts/evaluate.py --mode generation --strategy recursive --search-mode hybrid

# 指定输出路径
python scripts/evaluate.py --mode generation --output reports/gen_eval.csv
```

评估报告输出到 `reports/` 目录。

## 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行指定模块测试
python -m pytest tests/ingestion/test_chunker.py -v    # 分块器
python -m pytest tests/context/test_context.py -v       # 上下文管理
python -m pytest tests/rag/test_pipeline.py -v         # RAG 编排
python -m pytest tests/evaluation/test_eval.py -v       # 评估指标
python -m pytest tests/storage/test_mysql_crud.py -v    # MySQL CRUD
```

MySQL 测试需要 MySQL 服务运行中，否则会自动跳过。

## 技术栈

| 组件 | 技术选型 |
|------|---------|
| Web 框架 | FastAPI + Uvicorn |
| 向量检索 | FAISS（开发原型），生产可平滑迁移 Milvus 向量数据库 |
| BM25 检索 | 自实现（jieba 分词）,BM25（基于 jieba 中文分词自实现）；生产可替换 Elasticsearch |
| Embedding | sentence-transformers (BGE) |
| 重排 | Cross-Encoder (BGE-reranker) |
| LLM 生成 | DashScope (Qwen) |
| 持久化 | MySQL (pymysql + dbutils 连接池) |
| 指标采集 | prometheus-client |
| 测试 | pytest |
```

