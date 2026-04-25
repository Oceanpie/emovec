"""
prototype.py — 原型向量库管理。

存储/加载全维度 (4096D) numpy 向量，增量构建支持，条目解析工具。
维度截断在 matching.py 查询时进行，此处不做任何截断。
"""

import json
from pathlib import Path

import numpy as np

from .config import (
    ENTRIES_DIR,
    FULL_DIMS,
    PROTOTYPE_META,
    PROTOTYPE_VECTORS,
)
from .labels import load_labels


# ── 条目文本解析 ──────────────────────────────────────────────


def extract_description(file_path: Path) -> str:
    """
    从条目 .md 文件提取正文描述。

    跳过标题行（# 开头）和 --- 之后的参见链接部分。

    Args:
        file_path: 条目 .md 文件路径

    Returns:
        提取的正文文本（已 strip）。
    """
    content = file_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    body: list[str] = []
    in_see_also = False
    for line in lines:
        if line.startswith("# "):
            continue
        if line.strip() == "---":
            in_see_also = True
            continue
        if in_see_also:
            continue
        body.append(line)
    return "\n".join(body).strip()


def is_redirect(desc: str) -> bool:
    """判断条目描述是否为重定向（*see*: ...）。"""
    lower = desc.lower()
    return (lower.startswith("*see*:") or lower.startswith("*see*:")) and len(desc) < 100


# ── 向量存储读写 ──────────────────────────────────────────────


def load_prototype_store() -> tuple[np.ndarray, dict]:
    """
    加载原型向量库。

    Returns:
        (vectors, meta)
        - vectors: ndarray[N, 4096]，float32 原始向量
        - meta: dict，包含 entries 列表和元信息

    Raises:
        FileNotFoundError: 向量文件或元数据文件不存在。
    """
    if not PROTOTYPE_VECTORS.exists():
        raise FileNotFoundError(f"Prototype vectors not found: {PROTOTYPE_VECTORS}")
    if not PROTOTYPE_META.exists():
        raise FileNotFoundError(f"Prototype metadata not found: {PROTOTYPE_META}")

    vectors = np.load(str(PROTOTYPE_VECTORS))
    with open(PROTOTYPE_META, encoding="utf-8") as f:
        meta = json.load(f)
    return vectors, meta


def save_prototype_store(vectors: np.ndarray, meta: dict) -> None:
    """
    保存原型向量库。

    Args:
        vectors: ndarray[N, 4096] 原始向量
        meta: 元数据字典（将写入 JSON）
    """
    PROTOTYPE_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(PROTOTYPE_VECTORS), vectors)
    with open(PROTOTYPE_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ── 元数据工具 ────────────────────────────────────────────────


def build_row_to_entry_map(meta: dict) -> dict[int, dict]:
    """
    构建 {row_index: entry} 映射。

    Args:
        meta: load_prototype_store() 返回的元数据

    Returns:
        {row_index: entry_dict}，仅包含有向量的条目。
    """
    result: dict[int, dict] = {}
    for e in meta.get("entries", []):
        row = e.get("row_index", -1)
        if row >= 0:
            result[row] = e
    return result


def build_existing_index_map(meta: dict) -> dict[str, int]:
    """
    构建 {title: row_index} 映射，用于增量构建时判断哪些条目已有向量。

    Args:
        meta: 现有的元数据字典

    Returns:
        {title: row_index}，仅包含非跳过条目。
    """
    result: dict[str, int] = {}
    for e in meta.get("entries", []):
        row = e.get("row_index", -1)
        if row >= 0 and not e.get("skipped", False):
            result[e["title"]] = row
    return result
