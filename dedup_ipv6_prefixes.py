"""
dedup_ipv6_prefixes.py

输入两个 IPv6 前缀 txt 文件（每行一个前缀，如 "2001:db8::/32"），
做去重并"只保留短前缀"：如果某个前缀被另一个更长的前缀包含，
则丢弃被包含的（更长的）前缀，只保留范围更大（更短）的版本。

默认行为：把两个文件合并后一起去重，输出一个合并后的去重文件。

用法:
    python dedup_ipv6_prefixes.py <file1> <file2> [--out OUT]
    python dedup_ipv6_prefixes.py <file1> <file2> --separate [--out1 OUT1] [--out2 OUT2]

示例（合并输出一个文件）:
    python dedup_ipv6_prefixes.py seeds_a.txt seeds_b.txt --out merged_dedup.txt

示例（分别输出两个文件）:
    python dedup_ipv6_prefixes.py seeds_a.txt seeds_b.txt --separate \
        --out1 seeds_a_dedup.txt --out2 seeds_b_dedup.txt
"""
import argparse
import sys

# 复用项目内已有的 IPv6 工具函数
from utils.ipv6_utils import maximal_cover, normalize_prefix


def read_prefixes(path: str) -> list:
    """读取 IPv6 前缀文件，忽略空行与注释(#)，返回去空白后的原始字符串列表。"""
    prefixes = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prefixes.append(line)
    return prefixes


def write_prefixes(path: str, prefixes: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in prefixes:
            f.write(p + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="对两个 IPv6 前缀文件去重，仅保留短前缀（移除被包含的长前缀）。"
                    "默认合并两个文件后输出一个去重文件。")
    parser.add_argument("file1", help="第一个 IPv6 前缀输入文件")
    parser.add_argument("file2", help="第二个 IPv6 前缀输入文件")
    parser.add_argument("--out", default="merged_dedup.txt",
                        help="合并去重输出路径（默认 merged_dedup.txt）")
    parser.add_argument("--separate", action="store_true",
                        help="不合并，而是分别对两个文件去重，输出两个文件")
    parser.add_argument("--out1", default=None,
                        help="--separate 模式下第一个文件去重输出路径"
                             "（默认 <file1>.dedup.txt）")
    parser.add_argument("--out2", default=None,
                        help="--separate 模式下第二个文件去重输出路径"
                             "（默认 <file2>.dedup.txt）")
    args = parser.parse_args()

    if args.separate:
        # ---- 分别处理两个文件（不合并） ----
        out1 = args.out1 or (args.file1 + ".dedup.txt")
        out2 = args.out2 or (args.file2 + ".dedup.txt")

        raw1 = read_prefixes(args.file1)
        before1 = len(raw1)
        dedup1 = maximal_cover(raw1)
        write_prefixes(out1, dedup1)
        print(f"[file1] {args.file1}")
        print(f"        原始: {before1} 条, 去重后(保留短前缀): {len(dedup1)} 条 -> {out1}")

        raw2 = read_prefixes(args.file2)
        before2 = len(raw2)
        dedup2 = maximal_cover(raw2)
        write_prefixes(out2, dedup2)
        print(f"[file2] {args.file2}")
        print(f"        原始: {before2} 条, 去重后(保留短前缀): {len(dedup2)} 条 -> {out2}")
    else:
        # ---- 默认：合并两个文件后一起去重，输出一个文件 ----
        raw1 = read_prefixes(args.file1)
        raw2 = read_prefixes(args.file2)
        before_total = len(raw1) + len(raw2)
        merged = raw1 + raw2
        dedup = maximal_cover(merged)
        write_prefixes(args.out, dedup)
        print(f"[merge] {args.file1} + {args.file2}")
        print(f"        原始合计: {before_total} 条, "
              f"合并去重后(保留短前缀): {len(dedup)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
