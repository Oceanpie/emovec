"""
matching.py — 查询管线核心。

截断 → L2 归一化 → 余弦相似度 Top-K → Softmax → 加权标签求和。
返回结构化 QueryResult，不负责打印/格式化。
"""

from dataclasses import dataclass, field

import numpy as np

from .config import FULL_DIMS, QUERY_INSTRUCTION
from .embedding import get_embedding, init_clients


# ── 数据结构 ──────────────────────────────────────────────────


@dataclass
class MatchInfo:
    """单条原型匹配结果。"""
    rank: int
    title: str
    similarity: float
    weight: float
    top_labels: list[tuple[str, float]]  # top 3 (维度名, 分数)


@dataclass
class QueryResult:
    """完整查询结果。"""
    text: str
    scheme: str
    dim: int
    top_k: int
    tau: float
    scores: dict[str, float]          # {维度名: 加权分数}
    matches: list[MatchInfo] = field(default_factory=list)


# ── 核心算法 ──────────────────────────────────────────────────


def top_k_search(
    query_vec: np.ndarray,
    proto_vecs: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    余弦相似度 Top-K 搜索。

    等价于 FAISS IndexFlatIP + L2 归一化后的内积搜索。

    Args:
        query_vec: 已归一化的查询向量, shape=(dim,)
        proto_vecs: 已归一化的原型矩阵, shape=(N, dim)
        top_k: 返回前 K 个结果

    Returns:
        (similarities, indices) 按相似度降序排列。
    """
    sims = proto_vecs @ query_vec  # (N,)
    top_indices = np.argsort(sims)[::-1][:top_k]
    return sims[top_indices], top_indices


def softmax(similarities: np.ndarray, tau: float = 0.5) -> np.ndarray:
    """温度缩放 Softmax。"""
    exp_sims = np.exp(similarities / tau)
    return exp_sims / exp_sims.sum()


# ── 查询管线 ──────────────────────────────────────────────────


def run_query(
    text: str,
    proto_vecs: np.ndarray,
    row_to_entry: dict[int, dict],
    dims: list[str],
    label_key: str,
    *,
    top_k: int = 7,
    tau: float = 0.5,
    dim: int = 512,
    instruction: str = QUERY_INSTRUCTION,
    verbose: bool = False,
) -> QueryResult:
    """
    完整查询管线：嵌入 → 截断 → Top-K → Softmax → 加权求和。

    Args:
        text: 用户输入的中文文本
        proto_vecs: 原型向量矩阵, shape=(N, 4096)，原始未截断
        row_to_entry: {row_index: entry_dict} 映射
        dims: 情感维度名列表，如 ["joy", "anger", ...]
        label_key: "label_plutchik" 或 "label_original"
        top_k: 取前 K 个最近邻
        tau: Softmax 温度
        dim: MRL 截断维度 (32-4096)
        instruction: 查询端 instruction（默认使用 config.QUERY_INSTRUCTION）
        verbose: 是否在 matches 中填充详情（影响 MatchInfo.top_labels）

    Returns:
        QueryResult 包含各维度加权分数和匹配详情。
    """
    # 1. 嵌入查询文本（API 返回截断后未归一化的向量）
    query_raw = get_embedding(text, instruction=instruction, dim=dim)

    # 2. L2 归一化查询向量
    query_norm = np.linalg.norm(query_raw)
    query_vec = query_raw / max(query_norm, 1e-10)

    # 3. 截断 + 归一化原型向量
    proto_trunc = proto_vecs[:, :dim].copy()
    proto_norms = np.linalg.norm(proto_trunc, axis=1, keepdims=True)
    proto_norms = np.maximum(proto_norms, 1e-10)
    proto_normed = proto_trunc / proto_norms

    # 4. Top-K 搜索
    sims, indices = top_k_search(query_vec, proto_normed, top_k)

    # 5. Softmax 权重
    weights = softmax(sims, tau)

    # 6. 加权标签求和
    scores: dict[str, float] = {}
    for d in dims:
        val = 0.0
        for i in range(top_k):
            idx = int(indices[i])
            entry = row_to_entry.get(idx)
            if entry is None:
                continue
            label = entry.get(label_key)
            if label is not None:
                val += float(weights[i]) * label.get(d, 0.0)
        scores[d] = round(val, 4)

    # 7. 构建匹配详情（verbose 模式下填充）
    matches: list[MatchInfo] = []
    if verbose:
        for rank in range(top_k):
            idx = int(indices[rank])
            entry = row_to_entry.get(idx, {})
            label = entry.get(label_key, {})
            top_labels = sorted(
                ((d, label.get(d, 0.0)) for d in dims if label),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            matches.append(MatchInfo(
                rank=rank + 1,
                title=entry.get("title", "?"),
                similarity=float(sims[rank]),
                weight=float(weights[rank]),
                top_labels=top_labels,
            ))

    scheme = "plutchik" if label_key == "label_plutchik" else "original"
    return QueryResult(
        text=text,
        scheme=scheme,
        dim=dim,
        top_k=top_k,
        tau=tau,
        scores=scores,
        matches=matches,
    )


# ── 格式化输出（CLI 用）──────────────────────────────────────


def bar(value: float, width: int = 20) -> str:
    """生成文本进度条。"""
    filled = round(value * width)
    return "#" * filled + "-" * (width - filled)


def format_result(result: QueryResult) -> str:
    """
    将 QueryResult 格式化为可打印的文本。

    用于 CLI 脚本输出，不影响 run_query 的纯计算逻辑。
    """
    from .config import SCHEME_NAMES

    scheme_name = SCHEME_NAMES.get(result.scheme, "")
    lines: list[str] = [
        f'\n输入: "{result.text}"',
        f"方案: {scheme_name}, dim={result.dim}\n",
        "情感强度:",
    ]

    max_dim_len = max(len(d) for d in result.scores)
    for d, v in result.scores.items():
        lines.append(f"  {d:<{max_dim_len}} {v:>6.2f}  {bar(v)}")

    if result.matches:
        lines.append(f"\nTop-{result.top_k} 匹配原型 (--verbose):")
        for m in result.matches:
            top_str = ", ".join(
                f"{d}:{v:.2f}" for d, v in m.top_labels if v > 0
            )
            lines.append(
                f"  {m.rank}. {m.title} "
                f"(sim={m.similarity:.4f}, weight={m.weight:.4f})"
                f" — {top_str}"
            )

    lines.append("")
    return "\n".join(lines)
