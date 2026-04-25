"""
embedding.py — Embedding API 客户端。

双 Provider fallback（ModelScope → SiliconFlow），支持重试。
返回原始截断向量（不归一化），由调用方决定是否归一化。
"""

import os
import time
from typing import TYPE_CHECKING

import numpy as np
from openai import OpenAI

from .config import (
    EMBED_RETRY_DELAY,
    EMBED_RETRY_MAX,
    EMBEDDING_MODEL,
    FULL_DIMS,
    PROVIDERS,
)

if TYPE_CHECKING:
    pass

# ── 模块级客户端缓存 ──────────────────────────────────────────

_clients: list[tuple[str, OpenAI]] = []


def init_clients() -> list[tuple[str, OpenAI]]:
    """
    初始化所有可用 Provider 的 OpenAI 客户端。

    Returns:
        [(provider_name, client), ...] 列表。
        首次调用后缓存结果，后续调用直接返回缓存。
    """
    global _clients
    if _clients:
        return _clients
    for name, base_url, env_key in PROVIDERS:
        token = os.environ.get(env_key)
        if token:
            _clients.append((name, OpenAI(base_url=base_url, api_key=token)))
    return _clients


def get_embedding(
    text: str,
    *,
    instruction: str | None = None,
    dim: int = FULL_DIMS,
    clients: list[tuple[str, OpenAI]] | None = None,
) -> np.ndarray:
    """
    调用 Embedding API 获取向量。

    Args:
        text: 输入文本
        instruction: 可选 instruction 前缀。提供时格式为
                     "Instruct: {instruction}\\nQuery:{text}"
                     不提供时直接发送 text（用于 entry 端编码）。
        dim: MRL 截断维度 (32-4096)。默认 4096 = 不截断。
        clients: 预初始化的客户端列表。None 时使用模块级缓存。

    Returns:
        截断后的 float32 向量，shape=(dim,)，**未归一化**。

    Raises:
        RuntimeError: 所有 Provider 均失败时。
    """
    if clients is None:
        clients = init_clients()

    if not clients:
        raise RuntimeError("No embedding providers available. Check .env keys.")

    # 构造输入文本
    if instruction:
        api_input = f"Instruct: {instruction}\nQuery:{text}"
    else:
        api_input = text

    last_error: Exception | None = None
    for provider_name, client in clients:
        for attempt in range(EMBED_RETRY_MAX + 1):
            try:
                resp = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=api_input,
                    encoding_format="float",
                )
                full_vec = resp.data[0].embedding
                return np.array(full_vec[:dim], dtype=np.float32)

            except Exception as e:
                last_error = e
                err_str = str(e)
                # 400/429 直接 fallback 到下一个 Provider，不浪费重试
                if "429" in err_str or "400" in err_str:
                    break
                if attempt < EMBED_RETRY_MAX:
                    time.sleep(EMBED_RETRY_DELAY)

    raise RuntimeError(f"All embedding providers failed. Last error: {last_error}")
