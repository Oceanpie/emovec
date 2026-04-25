# EmoVec 情感向量分析

零样本中文文本 → 8 维情感强度向量，无需训练。

<!-- language selector -->
[**English**](README.md) | [**中文**](README.zh.md)
<!-- /language selector -->

<p align="center">
  <img src="https://img.shields.io/badge/Go-1.25+-00ADD8?logo=go&logoColor=white" alt="Go"/>
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

本仓库仅包含**匹配算法与服务代码**，不包含原型数据。运行查询需自行构建原型库。方法详见[构建指南](docs/原型库构建.md)。

提供两套实现：

| 实现 | 语言 | 用途 |
|------|------|------|
| `src/` + `scripts/match.py` | Python 3.12 | 参考实现 / 开发工具 |
| `emotion-service/` | Go 1.25+ | 生产服务 |

两者算法相同，均依赖同一套原型数据。

## 工作原理

```
输入: "我真的很想你"
         |
[Embedding API]  ──→  1024 维向量
         |
[余弦相似度]       ──→  Top-K 匹配
         |
[Softmax + 加权求和] ──→  8 维向量
         |
输出: {joy: 0.12, anger: 0.03, sadness: 0.68, ...}
```

**算法**（实现于 `src/matching.py` 和 `emotion-service/internal/matching/`）：
1. 通过 Embedding 模型编码输入文本
2. L2 归一化 → 与原型向量点积 → 余弦相似度
3. 取 Top-K（默认 7），Softmax 归一化（τ=0.5）
4. 加权求和原型的 8D 标签 → 最终情感向量

两套标签方案：

| 维度 | Plutchik | 原始方案 |
|------|----------|---------|
| 1 | joy | 高兴 |
| 2 | anger | 愤怒 |
| 3 | sadness | 悲伤 |
| 4 | fear | 害怕 |
| 5 | disgust | 厌恶 |
| 6 | surprise | 忧郁 |
| 7 | trust | 惊讶 |
| 8 | anticipation | 平静 |

## 快速开始

### 1. Python 环境（一条命令）

```bash
# 创建 .venv 并安装依赖
uv sync

# 激活
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 2. 配置 API Key

在项目根目录创建 `.env`：

```
MODELSCOPE_TOKEN=...
SILICONFLOWS_TOKEN=...
```

### 3. 构建原型库

原型数据文件（`output/embeddings/`、`emotion-service/data/`）**不包含在本仓库中**。

按 **[构建指南](docs/原型库构建.md)** 的步骤：
- 选择情绪概念来源
- 通过 LLM 生成 8D 标签
- 编码为 Embedding 向量
- 构建并验证原型库
- 打包以供 Go 服务使用

### 4. Python 查询

```bash
uv run python scripts/match.py "我真的很想你"
uv run python scripts/match.py --dim 1024 --verbose "今天心情不错"
uv run python scripts/match.py --scheme original "气死我了"
uv run python scripts/match.py    # 交互模式
```

### 5. 配置 Go 服务

Go 服务运行时读取 `config.yaml`（可通过 `--config 路径` 覆盖）。

```bash
cd emotion-service

# 将示例配置复制为运行配置
cp config.example.yaml config.yaml
# 编辑 config.yaml：填入 API Key、Provider 地址等
```

配置文件中 `${VAR}` 占位符从环境变量读取，API Key 无需写入文件：

```yaml
embedding:
  providers:
    - name: modelscope
      base_url: "https://api-inference.modelscope.cn/v1"
      api_key: "${MODELSCOPE_TOKEN}"        # 从环境变量读取
      model: "Qwen/Qwen3-Embedding-0.6B"
```

### 6. 运行 Go 服务

> **需要原型数据文件**：Go 服务通过 `go:embed` 在编译时内嵌 `data/model.safetensors`。先将构建好的数据文件复制到 `emotion-service/data/`。

```bash
cd emotion-service

# 复制构建好的原型数据
cp ../output/pack/model.safetensors data/

# 编译
go build -o emotion-service.exe .

# CLI 模式
./emotion-service.exe "我真的很想你"
./emotion-service.exe --scheme original "气死我了"

# HTTP 服务（监听 :8080）
./emotion-service.exe
```

## Go 服务 API

### 单条分析

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "我真的很想你"}'
```

响应：

```json
{
  "ok": true,
  "result": {
    "joy": 0.12, "anger": 0.03, "sadness": 0.68,
    "fear": 0.05, "disgust": 0.01, "surprise": 0.02,
    "trust": 0.06, "anticipation": 0.03
  }
}
```

### 批量分析

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["好开心啊", "气死我了", "有点害怕"]}'
```

### 请求参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `text` | string | 单条文本 |
| `texts` | string[] | 批量文本 |
| `scheme` | string | `"plutchik"` 或 `"original"` |
| `top_k` | int | 覆盖默认 K |
| `tau` | float | 覆盖默认温度 |
| `provider` | string | 指定 embedding provider |

### 健康检查

```bash
curl http://localhost:8080/health
```

## 架构

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  输入文本     │───→│ Embedding API     │───→│   原型库          │
│  (中文)      │    │ 双 Provider       │    │ 向量 + 标签       │
└──────────────┘    │ ModelScope/Silicon│    └────────┬─────────┘
                    └───────────────────┘             │
                                                      ▼
                                              ┌──────────────────┐
                                              │ 匹配器            │
                                              │ L2 归一化 → 点积  │
                                              │ Top-K → Softmax  │
                                              │ 加权求和          │
                                              └────────┬─────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │ 8D 情感向量       │
                                              └──────────────────┘
```

- **Embedding**：双 Provider fallback（ModelScope → SiliconFlow），OpenAI 兼容 API
- **匹配**：纯 Go/Python 矩阵乘法 — 无需向量数据库
- **原型**：150–200 个情绪概念向量，每个带 8D 软标签
- **瓶颈**：Embedding API 调用（~300ms）；匹配 < 1ms

## 项目结构

```
emovec/
├── src/                          Python 参考库
│   ├── config.py                   路径、常量、Provider 配置
│   ├── embedding.py                Embedding API 客户端
│   ├── prototype.py                原型库管理
│   ├── matching.py                 查询管线
│   └── labels.py                   标签加载工具
├── scripts/
│   └── match.py                    查询 CLI
├── emotion-service/              Go 生产服务
│   ├── main.go                     CLI + HTTP 入口
│   ├── embed.go                    go:embed 数据文件
│   ├── config.example.yaml         示例配置
│   ├── internal/
│   │   ├── config/                 配置加载与校验
│   │   ├── matching/               L2Normalize, TopK, Softmax, Match
│   │   ├── store/                  safetensors 加载（逆变换）
│   │   ├── embedding/              多 Provider fallback 客户端
│   │   ├── handler/                HTTP handlers
│   │   ├── safetensors/            纯 Go safetensors 解析器
│   │   └── transform/              SplitMix64 PRNG + 向量变换
│   └── cmd/
│       └── bench_match.go          匹配性能基准
├── docs/
│   ├── prototype-building.md       构建指南（英文）
│   ├── 原型库构建.md                 构建指南（中文）
│   ├── 原型.md                       Phase 0 设计文档
│   ├── 模型训练.md                    MLP 训练备选方案
│   └── 部署设计.md                    部署架构设计
├── pyproject.toml                 Python 依赖配置（uv）
├── .python-version                Python 版本锁定
├── LICENSE                        MIT
└── .gitignore
```

## 测试

```bash
# Go 服务
cd emotion-service && go test ./... -count=1

# 匹配性能基准
go run ./cmd/bench_match.go
# ~155µs/query, ~6439 QPS（纯匹配计算）
```

## 文档

| 文件 | 内容 |
|------|------|
| [docs/prototype-building.md](docs/prototype-building.md) | Build a prototype library from scratch |
| [docs/原型库构建.md](docs/原型库构建.md) | 从零构建原型库 |
| [docs/原型.md](docs/原型.md) | Phase 0 算法设计 |
| [docs/模型训练.md](docs/模型训练.md) | MLP 训练备选方案 |
| [docs/部署设计.md](docs/部署设计.md) | Go 服务部署架构 |
