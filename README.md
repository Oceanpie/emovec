# EmoVec

Zero-shot Chinese text to 8D emotion intensity vector — no training required.

<!-- language selector -->
[**English**](README.md) | [**中文**](README.zh.md)
<!-- /language selector -->

<p align="center">
  <img src="https://img.shields.io/badge/Go-1.25+-00ADD8?logo=go&logoColor=white" alt="Go"/>
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

This repository contains the **matching algorithm and service code** only — no prototype data is included. You need to build the prototype library yourself to run queries. See [prototype-building guide](docs/prototype-building.md) for the methodology.

Two implementations are provided:

| Implementation | Language | Purpose |
|---------------|----------|---------|
| `src/` + `scripts/match.py` | Python 3.12 | Reference / Dev tooling |
| `emotion-service/` | Go 1.25+ | Production service |

Both implement the same algorithm and require the same prototype data to function.

## How It Works

```
Input: "我真的很想你"
         |
[Embedding API]  ──→  1024D vector
         |
[Cosine Similarity] ──→  Top-K prototypes
         |
[Softmax + Weighted Sum] ──→  8D vector
         |
Output: {joy: 0.12, anger: 0.03, sadness: 0.68, ...}
```

**Algorithm** (implemented in `src/matching.py` and `emotion-service/internal/matching/`):
1. Encode input text via an embedding model (e.g., Qwen3-Embedding)
2. L2-normalize → dot product with prototype vectors → cosine similarity
3. Select Top-K (default 7), apply Softmax (τ=0.5)
4. Weighted sum of prototype 8D labels → final emotion vector

Two label schemes available:

| Dim | Plutchik | Original |
|-----|----------|----------|
| 1 | joy | 高兴 |
| 2 | anger | 愤怒 |
| 3 | sadness | 悲伤 |
| 4 | fear | 害怕 |
| 5 | disgust | 厌恶 |
| 6 | surprise | 忧郁 |
| 7 | trust | 惊讶 |
| 8 | anticipation | 平静 |

## Quick Start

### 1. Python Environment (one command)

```bash
# Create .venv and install dependencies
uv sync

# Activate
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 2. Configure API Keys

Copy the template and fill in your keys:

```bash
cp .env.example .env
# Edit .env: set your embedding endpoint URLs and API keys
```

### 3. Build the Prototype Library

The prototype data files (`output/embeddings/`, `emotion-service/data/`) are **not included** in this repository.

Follow the **[prototype building guide](docs/prototype-building.md)** to:
- Source emotional concepts
- Generate 8D labels via LLM
- Encode entries into embedding vectors
- Build and validate the prototype store
- Package for the Go service

### 4. Run Queries (Python)

```bash
uv run python scripts/match.py "我真的很想你"
uv run python scripts/match.py --dim 1024 --verbose "今天心情不错"
uv run python scripts/match.py --scheme original "气死我了"
uv run python scripts/match.py    # interactive mode
```

### 5. Configure the Go Service

The binary reads `config.yaml` at startup (use `--config path` to override).

```bash
cd emotion-service

# Copy the example config into place
cp config.example.yaml config.yaml
# Then edit config.yaml: set your API keys, provider endpoints, etc.
```

The config file defines embedding providers (with fallback order), matching parameters, and server settings. `${VAR}` placeholders are expanded from environment variables — keep secrets in `.env`, not in the yaml:

```yaml
embedding:
  providers:
    - name: selfhost
      base_url: "${EMBEDDING_SELFHOST_URL}"    # read from .env
      api_key: "${EMBEDDING_SELFHOST_KEY}"
      model: "Qwen/Qwen3-Embedding-0.6B"

    - name: local
      base_url: "${EMBEDDING_LOCAL_URL}"       # fallback provider
      api_key: "${EMBEDDING_LOCAL_KEY}"
      model: "Qwen/Qwen3-Embedding-0.6B"
```

### 6. Run Go Service

> **Requires prototype data files**: The Go service embeds `data/model.safetensors` at compile time via `go:embed`. Build the prototype library first, then copy the files to `emotion-service/data/`.

```bash
cd emotion-service

# Copy built prototype data into place
cp ../output/pack/model.safetensors data/

# Build
go build -o emotion-service.exe .

# CLI mode
./emotion-service.exe "我真的很想你"
./emotion-service.exe --scheme original "气死我了"

# HTTP service (listens on :8080)
./emotion-service.exe
```

## Go Service API

### Single Analysis

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "我真的很想你"}'
```

Response:

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

### Batch Analysis

```bash
curl -X POST http://localhost:8080/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["好开心啊", "气死我了", "有点害怕"]}'
```

### Request Parameters

| Param | Type | Description |
|-------|------|-------------|
| `text` | string | Single text |
| `texts` | string[] | Batch texts |
| `scheme` | string | `"plutchik"` or `"original"` |
| `top_k` | int | Override default K |
| `tau` | float | Override default temperature |
| `provider` | string | Specific embedding provider |

### Health Check

```bash
curl http://localhost:8080/health
```

## Architecture

```
┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐
│  Input Text  │───→│ Embedding API     │───→│ Prototype Store  │
│  (Chinese)   │    │ Dual-Provider     │    │ vectors + labels │
└──────────────┘    │ ModelScope/Silicon│    └────────┬─────────┘
                    └───────────────────┘             │
                                                      ▼
                                              ┌──────────────────┐
                                              │ Matcher          │
                                              │ L2 Norm → Dot    │
                                              │ Top-K → Softmax  │
                                              │ Weighted Sum     │
                                              └────────┬─────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │ 8D Emotion Vec   │
                                              └──────────────────┘
```

- **Embedding**: Dual-provider fallback (ModelScope → SiliconFlow), OpenAI-compatible API
- **Matching**: Pure Go/Python matrix multiplication — no vector database needed
- **Prototypes**: 150–200 emotional concept vectors with 8D soft labels
- **Bottleneck**: Embedding API call (~300ms); matching is < 1ms

## Project Structure

```
emovec/
├── src/                          Python reference library
│   ├── config.py                   Paths, constants, provider config
│   ├── embedding.py                Embedding API client (dual fallback)
│   ├── prototype.py                Prototype store management
│   ├── matching.py                 Query pipeline
│   └── labels.py                   Label loading utilities
├── scripts/
│   └── match.py                    Query CLI
├── emotion-service/              Go production service
│   ├── main.go                     CLI + HTTP entry point
│   ├── embed.go                    go:embed data files
│   ├── config.example.yaml         Example configuration
│   ├── internal/
│   │   ├── config/                 Config loading & validation
│   │   ├── matching/               L2Normalize, TopK, Softmax, Match
│   │   ├── store/                  Safetensors loading (inverse transform)
│   │   ├── embedding/              Multi-provider fallback client
│   │   ├── handler/                HTTP handlers
│   │   ├── safetensors/            Pure Go safetensors parser
│   │   └── transform/              SplitMix64 PRNG + vector transform
│   └── cmd/
│       └── bench_match.go          Matching performance benchmark
├── docs/
│   ├── prototype-building.md        Build guide (English)
│   ├── 原型库构建.md                  Build guide (Chinese)
│   ├── 原型.md                       Phase 0 design
│   ├── 模型训练.md                    MLP alternative design
│   └── 部署设计.md                    Deployment architecture
├── pyproject.toml                 Python dependencies (uv)
├── .python-version                Python version pin
├── LICENSE                        MIT
└── .gitignore
```

## Tests

```bash
# Go service
cd emotion-service && go test ./... -count=1

# Matching benchmark
go run ./cmd/bench_match.go
# ~155µs/query, ~6439 QPS (matching only)
```

## Documentation

| File | Description |
|------|-------------|
| [docs/prototype-building.md](docs/prototype-building.md) | Build a prototype library from scratch |
| [docs/原型库构建.md](docs/原型库构建.md) | 构建原型库指南（中文） |
| [docs/原型.md](docs/原型.md) | Phase 0 algorithm design |
| [docs/模型训练.md](docs/模型训练.md) | MLP training alternative design |
| [docs/部署设计.md](docs/部署设计.md) | Go service deployment architecture |
