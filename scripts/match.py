"""
match.py — Phase 0 情感查询 CLI。

薄壳 CLI，调用 src.matching.run_query + format_result。
支持 --dim 参数选择 MRL 截断维度。

Usage:
    .venv\\Scripts\\python scripts\\match.py "我真的很想你"
    .venv\\Scripts\\python scripts\\match.py --dim 1024 --verbose "今天心情不错"
    .venv\\Scripts\\python scripts\\match.py --scheme original "气死我了"
    .venv\\Scripts\\python scripts\\match.py           # 交互模式
"""

import argparse
import io
import sys
from pathlib import Path

# Windows 终端 UTF-8 修复
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 让 scripts/ 能 import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_DIM, FULL_DIMS, SCHEME_NAMES, load_env
from src.labels import get_dimension_names
from src.matching import format_result, run_query
from src.prototype import build_row_to_entry_map, load_prototype_store


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 0 情感查询: 文本 → 8D 情感向量"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="要分析的中文文本（省略则进入交互模式）",
    )
    parser.add_argument(
        "--scheme",
        choices=["plutchik", "original"],
        default="plutchik",
        help="标签方案 (default: plutchik)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=7,
        help="最近邻数量 (default: 7)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="Softmax 温度 (default: 0.5)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DEFAULT_DIM,
        help=f"MRL 截断维度 (default: {DEFAULT_DIM}, 完整: 4096)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示 Top-K 匹配详情",
    )
    return parser.parse_args()


def main():
    load_env()
    args = parse_args()

    # 校验维度
    if args.dim < 32 or args.dim > FULL_DIMS:
        print(f"Error: --dim 范围 32-{FULL_DIMS}，当前 {args.dim}")
        sys.exit(1)

    # 加载原型库
    try:
        proto_vecs, meta = load_prototype_store()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("请先运行 build_prototype.py 构建原型库。")
        sys.exit(1)

    print(f"已加载 {proto_vecs.shape[0]} 条原型向量 ({proto_vecs.shape[1]}D)")

    if args.dim > proto_vecs.shape[1]:
        print(f"Error: --dim {args.dim} 超出存储维度 {proto_vecs.shape[1]}")
        sys.exit(1)

    # 构建查询所需数据
    entries = meta["entries"]
    label_key = "label_plutchik" if args.scheme == "plutchik" else "label_original"
    row_to_entry = build_row_to_entry_map(meta)
    dims = get_dimension_names(entries, label_key)

    if dims is None:
        print(f"Error: 未找到方案 '{args.scheme}' 的有效标签")
        sys.exit(1)

    # 查询函数（共享参数）
    def do_query(text: str):
        result = run_query(
            text=text,
            proto_vecs=proto_vecs,
            row_to_entry=row_to_entry,
            dims=dims,
            label_key=label_key,
            top_k=args.top_k,
            tau=args.tau,
            dim=args.dim,
            verbose=args.verbose,
        )
        print(format_result(result))

    if args.text:
        # 单次查询
        do_query(args.text)
    else:
        # 交互模式
        scheme_name = SCHEME_NAMES[args.scheme]
        print(f"Phase 0 情感查询 (交互模式)")
        print(f"方案: {scheme_name}, K={args.top_k}, tau={args.tau}, dim={args.dim}")
        print('输入 "quit" 或 "exit" 退出。\n')
        while True:
            try:
                text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not text:
                continue
            if text.lower() in ("quit", "exit"):
                print("Bye.")
                break
            do_query(text)


if __name__ == "__main__":
    main()
