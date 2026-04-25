"""
labels.py — 标签加载工具。

加载 output/labels/ 下的 JSON 标签文件，提供统一的查询接口。
"""

from pathlib import Path

from .config import LABELS_DIR


def load_labels(filename: str) -> dict[str, dict[str, float]]:
    """
    加载标签文件，返回 {title: {维度名: 分数}}。

    Args:
        filename: 标签文件名，如 "labels_plutchik.json"

    Returns:
        非跳过条目的标签字典。文件不存在时返回空 dict。
    """
    path = LABELS_DIR / filename
    if not path.exists():
        return {}
    import json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result: dict[str, dict[str, float]] = {}
    for r in data.get("results", []):
        if r.get("skipped") or r.get("label") is None:
            continue
        result[r["title"]] = r["label"]
    return result


def get_dimension_names(entries: list[dict], label_key: str) -> list[str] | None:
    """
    从 entries 列表中提取维度名列表。

    Args:
        entries: prototype_meta.json 中的 entries 列表
        label_key: "label_plutchik" 或 "label_original"

    Returns:
        维度名列表（如 ["joy", "anger", ...]），无有效标签时返回 None。
    """
    for e in entries:
        if e.get(label_key) is not None:
            return list(e[label_key].keys())
    return None
