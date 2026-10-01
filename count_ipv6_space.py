import argparse
import ipaddress
import os

"""conda activate TranSCA && python count_ipv6_space.py addrs/short_test_base.txt --compare addrs/short_test_advance.txt -o /root/embedding/ipv6/ipv6_compare.png 2>&1"""
def fmt_addr(n):
    """整数转 IPv6 地址字符串。"""
    return str(ipaddress.IPv6Address(n))


def fmt_size(n):
    """将地址数格式化为易读字符串（带等效前缀长度）。"""
    for prefix_len in range(128, -1, -1):
        if n == 2 ** (128 - prefix_len):
            return f"{n} (= 1 × /{prefix_len})"
    # 非 2 的幂，找最接近的等效
    for prefix_len in range(128, -1, -1):
        block = 2 ** (128 - prefix_len)
        if n >= block:
            ratio = n / block
            return f"{n} (≈ {ratio:.2f} × /{prefix_len})"
    return str(n)


def merge_and_count(prefixes, verbose=True):
    """将前缀转为 (start, end) 区间，合并重叠后求总地址数。"""
    intervals = []
    for p in prefixes:
        net = ipaddress.ip_network(p, strict=False)
        start = int(net.network_address)
        end = start + net.num_addresses  # 左闭右开
        intervals.append((start, end, p))

    intervals.sort()

    if verbose:
        print("=" * 60)
        print("排序后的前缀列表:")
        print("=" * 60)
        for i, (start, end, p) in enumerate(intervals):
            print(f"  [{i+1}] {p:>20s}  ->  {fmt_addr(start)} ~ {fmt_addr(end - 1)}")
        print()

    # merged 每项: (start, end, [包含的原始前缀列表])
    # merge_log: 记录每步合并操作
    merged = []
    merge_log = []  # 每项: (动作, 详情字符串)

    for start, end, prefix in intervals:
        if merged and start <= merged[-1][1]:
            old_start, old_end, old_members = merged[-1]
            new_end = max(old_end, end)
            merged[-1] = (old_start, new_end, old_members + [prefix])

            if start == old_start and end <= old_end:
                reason = "完全包含"
                detail = f"{prefix} 被 {old_members[0]} 完全包含，跳过"
            elif start < old_end and end > old_end:
                overlap = old_end - start
                reason = "部分重叠"
                detail = (f"{prefix} 与当前区间部分重叠 (重叠 {fmt_size(overlap)})，"
                          f"区间尾部扩展: {fmt_addr(old_end - 1)} -> {fmt_addr(new_end - 1)}")
            elif start == old_end:
                reason = "首尾相接"
                detail = (f"{prefix} 与当前区间首尾相接，"
                          f"区间尾部扩展: {fmt_addr(old_end - 1)} -> {fmt_addr(new_end - 1)}")
            else:
                reason = "被包含"
                detail = f"{prefix} 落在当前区间内，区间不变"

            merge_log.append(("合并", reason, detail,
                              (old_start, old_end), (old_start, new_end)))
        else:
            merged.append((start, end, [prefix]))
            merge_log.append(("新建", "独立区间", f"以 {prefix} 开始新区间",
                              None, (start, end)))

    if verbose:
        print("=" * 60)
        print("合并过程 (逐步):")
        print("=" * 60)
        for i, (action, reason, detail, old_interval, new_interval) in enumerate(merge_log):
            ns, ne = new_interval
            print(f"  步骤 {i+1}: [{action} - {reason}]")
            print(f"    {detail}")
            if old_interval:
                os, oe = old_interval
                print(f"    合并前: {fmt_addr(os)} ~ {fmt_addr(oe - 1)}  ({fmt_size(oe - os)})")
            print(f"    合并后: {fmt_addr(ns)} ~ {fmt_addr(ne - 1)}  ({fmt_size(ne - ns)})")
            print()

    total = sum(end - start for start, end, _ in merged)
    return total, merged


def load_prefixes(filepath):
    """从文件加载前缀列表。"""
    prefixes = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            prefixes.append(line)
    return prefixes


def print_report(prefixes):
    """单文件分析报告。"""
    print(f"输入前缀数: {len(prefixes)}")
    print()

    total, merged = merge_and_count(prefixes)

    print("=" * 60)
    print(f"最终合并结果: {len(merged)} 个独立区间")
    print("=" * 60)

    for i, (start, end, members) in enumerate(merged):
        size = end - start
        print(f"--- 区间 {i+1} ---")
        print(f"  范围: {fmt_addr(start)} ~ {fmt_addr(end - 1)}")
        print(f"  地址数: {fmt_size(size)}")
        print(f"  包含 {len(members)} 个原始前缀:")
        for p in members:
            net = ipaddress.ip_network(p, strict=False)
            print(f"    {p:>20s}  ({net.num_addresses} 个地址)")
        print()

    print("=" * 60)
    print("汇总统计")
    print("=" * 60)
    print(f"  去重后总地址数: {total}")
    print(f"  等效 /48 数: {total / 2**80:.2f}")
    print(f"  等效 /64 数: {total / 2**64:.2f}")
    print(f"  占完整 IPv6 空间比例: {total / 2**128:.15e}")


def compare_plot(file_a, file_b, output=None):
    """对比两个前缀文件的地址覆盖范围，绘制在一张图上。"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    prefixes_a = load_prefixes(file_a)
    prefixes_b = load_prefixes(file_b)

    _, merged_a = merge_and_count(prefixes_a, verbose=False)
    _, merged_b = merge_and_count(prefixes_b, verbose=False)

    label_a = os.path.basename(file_a)
    label_b = os.path.basename(file_b)

    # --- 收集所有区间，确定全局地址范围 ---
    all_intervals = (
        [(s, e) for s, e, _ in merged_a] +
        [(s, e) for s, e, _ in merged_b]
    )
    # --- 按不相交的地址簇分组，避免大片空白 ---
    # 先把所有区间排序，找出不相交的簇
    all_sorted = sorted(all_intervals)
    clusters = []  # 每项: (cluster_min, cluster_max)
    for s, e in all_sorted:
        if clusters and s <= clusters[-1][1]:
            clusters[-1] = (clusters[-1][0], max(clusters[-1][1], e))
        else:
            clusters.append((s, e))

    # 给每个簇加一点边距 (簇宽度的 5%)
    padded_clusters = []
    for cs, ce in clusters:
        width = ce - cs
        if width == 0:
            width = 1
        pad = max(width // 20, 1)  # 整数运算避免浮点精度问题
        padded_clusters.append((cs - pad, ce + pad))

    # --- 构建映射: 真实地址 -> 绘图 x 坐标 (拼接各簇) ---
    # 使用 Fraction 避免超大整数转 float 时精度丢失
    from fractions import Fraction

    gap_width = Fraction(2, 100)
    n_clusters = len(padded_clusters)
    usable_width = Fraction(1) - gap_width * max(n_clusters - 1, 0)
    # 每个簇等宽分配，保证小簇也能看清
    per_cluster_width = usable_width / n_clusters

    cluster_mappings = []  # (real_start, real_end, plot_offset_frac, scale_frac)
    plot_cursor = Fraction(0)
    for cs, ce in padded_clusters:
        span = Fraction(ce - cs)
        scale = per_cluster_width / span
        cluster_mappings.append((cs, ce, plot_cursor, scale))
        plot_cursor += per_cluster_width + gap_width

    def addr_to_plot(addr):
        for real_s, real_e, offset, scale in cluster_mappings:
            if real_s <= addr <= real_e:
                return float(offset + Fraction(addr - real_s) * scale)
        # 超出范围 — 夹到最近簇
        if addr < cluster_mappings[0][0]:
            return float(cluster_mappings[0][2])
        last = cluster_mappings[-1]
        return float(last[2] + Fraction(last[1] - last[0]) * last[3])

    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(16, 5))

    bar_height = 0.35
    y_a = 1.0  # 文件 A 的 y 位置
    y_b = 0.4  # 文件 B 的 y 位置

    color_a = '#2196F3'  # 蓝色
    color_b = '#FF9800'  # 橙色
    color_a_raw = '#90CAF9'  # 浅蓝 (原始前缀)
    color_b_raw = '#FFE0B2'  # 浅橙 (原始前缀)

    def draw_intervals(merged, y, color, color_raw, prefixes):
        """绘制合并区间和原始前缀。"""
        # 先画原始前缀 (浅色细条)
        for p in prefixes:
            net = ipaddress.ip_network(p, strict=False)
            s = int(net.network_address)
            e = s + net.num_addresses
            x_start = addr_to_plot(s)
            x_end = addr_to_plot(e)
            ax.barh(y - bar_height * 0.3, x_end - x_start, left=x_start,
                    height=bar_height * 0.4, color=color_raw, edgecolor='none',
                    alpha=0.7)
        # 再画合并后区间 (深色粗条)
        for s, e, members in merged:
            x_start = addr_to_plot(s)
            x_end = addr_to_plot(e)
            ax.barh(y, x_end - x_start, left=x_start,
                    height=bar_height, color=color, edgecolor='white',
                    linewidth=0.5, alpha=0.85)
            # 标注前缀长度
            mid_x = (x_start + x_end) / 2
            # 尝试找到最大的前缀长度来标注
            if len(members) == 1:
                label = members[0]
            else:
                label = f"{len(members)} prefixes"
            # 只在区间够宽时标注
            if x_end - x_start > 0.02:
                ax.text(mid_x, y, label, ha='center', va='center',
                        fontsize=7, color='white', fontweight='bold')

    draw_intervals(merged_a, y_a, color_a, color_a_raw, prefixes_a)
    draw_intervals(merged_b, y_b, color_b, color_b_raw, prefixes_b)

    # --- 簇分隔线 ---
    for i in range(len(cluster_mappings) - 1):
        cm = cluster_mappings[i]
        pc = padded_clusters[i]
        x_break = float(cm[2] + Fraction(pc[1] - pc[0]) * cm[3] + gap_width / 2)
        ax.axvline(x_break, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.text(x_break, 1.55, '// gap //', ha='center', va='bottom',
                fontsize=7, color='gray', style='italic')

    # --- 绘制簇的地址范围标注 ---
    for i, (cs, ce) in enumerate(clusters):
        x_left = addr_to_plot(cs)
        x_right = addr_to_plot(ce)
        mid_x = (x_left + x_right) / 2
        ax.text(mid_x, -0.15, fmt_addr(cs), ha='center', va='top', fontsize=6, color='#555')
        if x_right - x_left > 0.05:
            ax.text(x_right, -0.15, fmt_addr(ce - 1), ha='center', va='top', fontsize=6, color='#555')

    # --- 样式 ---
    ax.set_yticks([y_a, y_a - bar_height * 0.3, y_b, y_b - bar_height * 0.3])
    ax.set_yticklabels([f'{label_a} (merged)', f'{label_a} (raw)',
                        f'{label_b} (merged)', f'{label_b} (raw)'], fontsize=9)
    ax.set_xlim(-0.02, float(plot_cursor - gap_width) + 0.02)
    ax.set_ylim(-0.35, 1.6)
    ax.set_xlabel('')
    ax.xaxis.set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # 图例
    legend_elements = [
        Patch(facecolor=color_a, alpha=0.85, label=f'{label_a} merged'),
        Patch(facecolor=color_a_raw, alpha=0.7, label=f'{label_a} raw prefixes'),
        Patch(facecolor=color_b, alpha=0.85, label=f'{label_b} merged'),
        Patch(facecolor=color_b_raw, alpha=0.7, label=f'{label_b} raw prefixes'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    # 统计信息
    total_a = sum(e - s for s, e, _ in merged_a)
    total_b = sum(e - s for s, e, _ in merged_b)
    stats_text = (
        f"{label_a}: {len(prefixes_a)} prefixes -> {len(merged_a)} intervals, "
        f"equiv /{_equiv_prefix(total_a)}\n"
        f"{label_b}: {len(prefixes_b)} prefixes -> {len(merged_b)} intervals, "
        f"equiv /{_equiv_prefix(total_b)}"
    )
    ax.set_title('IPv6 Address Coverage Comparison', fontsize=13, fontweight='bold', pad=15)
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=8, color='#555',
             family='monospace')

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if output is None:
        output = 'ipv6_compare.png'
    fig.savefig(output, dpi=200, bbox_inches='tight')
    print(f"对比图已保存: {output}")
    plt.close(fig)


def _equiv_prefix(total_addrs):
    """返回最接近的等效前缀长度字符串。"""
    import math
    if total_addrs == 0:
        return "N/A"
    bits = math.log2(total_addrs)
    prefix_len = 128 - bits
    if prefix_len == int(prefix_len):
        return f"{int(prefix_len)}"
    return f"~{prefix_len:.1f}"


def main():
    parser = argparse.ArgumentParser(description='IPv6 前缀地址空间统计工具')
    parser.add_argument('file', help='前缀文件')
    parser.add_argument('--compare', '-c', metavar='FILE2',
                        help='与另一个前缀文件对比，绘制覆盖范围图')
    parser.add_argument('--output', '-o', metavar='PNG',
                        help='对比图输出路径 (默认: ipv6_compare.png)')

    args = parser.parse_args()

    if args.compare:
        compare_plot(args.file, args.compare, output=args.output)
    else:
        prefixes = load_prefixes(args.file)
        print_report(prefixes)


if __name__ == "__main__":
    main()
