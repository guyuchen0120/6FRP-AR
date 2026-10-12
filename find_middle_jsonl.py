"""
find_middle_jsonl.py

从 jsonl 文件（每行一个 JSON 对象）中找到"最中间的那一项"并打印出来。

- 行数为奇数：返回正中间那一行（index = n // 2）。
- 行数为偶数：返回偏后那条中间行（index = n // 2，例如 4 行取第 2 行/0-indexed 索引 2）。

采用流式读取，只需遍历一遍文件，适合大文件（不会一次性载入内存）。

用法:
    python find_middle_jsonl.py <file.jsonl>
    python find_middle_jsonl.py <file.jsonl> --index    # 同时打印行号(0-based)

示例:
    python find_middle_jsonl.py results/online_run1/round1_candidates.jsonl
"""
import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="从 jsonl 文件中找到最中间的那一项。")
    parser.add_argument("file", help="输入的 jsonl 文件路径")
    parser.add_argument("--index", action="store_true",
                        help="同时打印中间项的行号(0-based)")
    args = parser.parse_args()

    # 第一次遍历：统计总行数（跳过空行）
    total = 0
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total += 1

    if total == 0:
        print("错误：文件为空或没有任何有效行。", file=sys.stderr)
        sys.exit(1)

    mid = total // 2  # 中间索引（偶数条时取偏后）

    # 第二次遍历：定位到中间那一行
    cur = -1
    middle_line = None
    with open(args.file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            cur += 1
            if cur == mid:
                middle_line = line
                break

    # 解析为 JSON 对象后打印（更易读），解析失败则原样输出
    try:
        obj = json.loads(middle_line)
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    except json.JSONDecodeError:
        print(middle_line.rstrip("\n"))

    if args.index:
        print(f"\n(总行数: {total}, 中间行号(0-based): {mid})", file=sys.stderr)


if __name__ == "__main__":
    main()
