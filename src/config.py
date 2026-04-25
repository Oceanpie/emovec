"""
config.py — 全局路径、常量、Provider 配置单一来源。

所有模块通过 from src.config import ... 获取配置，不硬编码路径。
"""

from pathlib import Path

# ── 项目根 ────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent

# ── 数据目录 ──────────────────────────────────────────────────

DATA_DIR = BASE / "data"
ENTRIES_DIR = DATA_DIR / "entries"
ENTRIES_METADATA = ENTRIES_DIR / "metadata.json"
RAW_DIR = DATA_DIR / "raw"
NOVELS_DIR = DATA_DIR / "novels"
TEST_DIR = DATA_DIR / "test"

# ── 输出目录 ──────────────────────────────────────────────────

OUTPUT_DIR = BASE / "output"
LABELS_DIR = OUTPUT_DIR / "labels"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
VIZ_DIR = OUTPUT_DIR / "viz"

# 8B 原型库（Python 管线默认使用）
PROTOTYPE_VECTORS = EMBEDDINGS_DIR / "8b" / "prototype_vectors.npy"
PROTOTYPE_META = EMBEDDINGS_DIR / "8b" / "prototype_meta.json"

# ── 环境变量 ──────────────────────────────────────────────────

ENV_PATH = BASE / ".env"

# ── 模型配置 ──────────────────────────────────────────────────

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
FULL_DIMS = 4096
DEFAULT_DIM = 512  # MRL 默认截断维度，可通过 --dim 调整

# 查询端 instruction（英文，Qwen3 官方建议）
QUERY_INSTRUCTION = (
    "Given a Chinese emotional utterance, retrieve the most relevant "
    "emotion description that matches the emotional state expressed"
)

# ── Provider 配置 ─────────────────────────────────────────────
# (名称, base_url, 环境变量 key)
PROVIDERS = [
    ("ModelScope", "https://api-inference.modelscope.cn/v1", "MODELSCOPE_TOKEN"),
    ("SiliconFlow", "https://api.siliconflow.cn/v1", "SILICONFLOWS_TOKEN"),
]

# ── 并发/重试 ────────────────────────────────────────────────

MAX_EMBED_WORKERS = 5
EMBED_RETRY_MAX = 3
EMBED_RETRY_DELAY = 10  # seconds

# ── 方案名称 ──────────────────────────────────────────────────

SCHEME_NAMES = {
    "plutchik": "Plutchik 8D",
    "original": "原始方案 8D",
}

# ── 工具函数 ──────────────────────────────────────────────────


def load_env() -> None:
    """从 .env 文件加载环境变量（不覆盖已有值）。"""
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        import os
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
