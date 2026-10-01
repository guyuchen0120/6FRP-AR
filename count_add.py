import argparse
import ipaddress
import sys
from typing import Set, List

def load_prefixes(filename: str) -> Set[ipaddress.IPv6Network]:
    """
    从文件加载IPv6前缀，返回IPv6Network对象的集合
    
    参数:
        filename: 输入文件名
    
    返回:
        IPv6Network对象的集合
    """
    prefixes = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # 提取每行第一个字段（前缀）
                prefix_str = line.split()[0]
                
                try:
                    # 解析IPv6前缀
                    network = ipaddress.IPv6Network(prefix_str, strict=False)
                    prefixes.add(network)
                except ValueError as e:
                    print(f"警告: 文件 {filename} 第{line_num}行 '{prefix_str}' 不是有效的IPv6前缀: {e}")
                    
    except FileNotFoundError:
        print(f"错误: 文件 '{filename}' 未找到", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取文件 '{filename}' 时出错: {e}", file=sys.stderr)
        sys.exit(1)
    
    return prefixes

def is_subnet_or_equal(child: ipaddress.IPv6Network, parent: ipaddress.IPv6Network) -> bool:
    """
    检查一个前缀是否是另一个前缀的子网或相等（兼容旧版Python）
    
    参数:
        child: 子前缀
        parent: 父前缀
    
    返回:
        如果child是parent的子网或相等返回True
    """
    # 方法1: 使用subnet_of（Python 3.7+）
    if hasattr(ipaddress.IPv6Network, 'subnet_of'):
        return child.subnet_of(parent)
    
    # 方法2: 手动检查（兼容所有版本）
    # 检查前缀长度：子网的前缀长度必须大于等于父网
    if child.prefixlen < parent.prefixlen:
        return False
    
    # 将两个网络地址转换为整数进行比较
    child_int = int(child.network_address)
    parent_int = int(parent.network_address)
    
    # 计算掩码
    mask = (1 << 128) - (1 << (128 - parent.prefixlen))
    
    # 检查子网是否在父网范围内
    return (child_int & mask) == (parent_int & mask)

def filter_new_prefixes(doc1_prefixes: Set[ipaddress.IPv6Network], 
                        doc2_prefixes: Set[ipaddress.IPv6Network]) -> List[ipaddress.IPv6Network]:
    """
    从doc2中找出不是doc1中任何前缀的子前缀的新前缀
    
    参数:
        doc1_prefixes: 第一个文档的前缀集合
        doc2_prefixes: 第二个文档的前缀集合
    
    返回:
        新的前缀列表（不是doc1子前缀的部分）
    """
    new_prefixes = []
    
    # 优化：先按前缀长度排序，短的在前（覆盖范围大）
    sorted_doc1 = sorted(doc1_prefixes, key=lambda x: x.prefixlen)
    
    for prefix in doc2_prefixes:
        is_subnet = False
        # 检查当前前缀是否是doc1中任何前缀的子前缀
        for existing in sorted_doc1:
            # 如果现有前缀比待检查的前缀还长，说明范围更小，可以跳过后续检查
            if existing.prefixlen > prefix.prefixlen:
                continue
                
            if is_subnet_or_equal(prefix, existing):
                is_subnet = True
                break
        
        if not is_subnet:
            new_prefixes.append(prefix)
    
    return new_prefixes

def save_prefixes(filename: str, prefixes: Set[ipaddress.IPv6Network], mode: str = 'w'):
    """
    保存前缀到文件
    
    参数:
        filename: 输出文件名
        prefixes: 前缀集合
        mode: 文件打开模式 ('w' 写, 'a' 追加)
    """
    with open(filename, mode, encoding='utf-8') as f:
        for prefix in sorted(prefixes):  # 排序后保存
            f.write(f"{prefix}\n")

def print_prefix_stats(prefixes: Set[ipaddress.IPv6Network], name: str):
    """
    打印前缀统计信息
    """
    if not prefixes:
        print(f"{name}: 空集合")
        return
    
    # 按前缀长度分组统计
    prefix_lengths = {}
    for prefix in prefixes:
        plen = prefix.prefixlen
        prefix_lengths[plen] = prefix_lengths.get(plen, 0) + 1
    
    print(f"\n{name} 统计:")
    print(f"  总数: {len(prefixes)}")
    print(f"  前缀长度分布:")
    for plen in sorted(prefix_lengths.keys()):
        print(f"    /{plen}: {prefix_lengths[plen]} 个")

def optimize_prefixes(prefixes: Set[ipaddress.IPv6Network]) -> Set[ipaddress.IPv6Network]:
    """
    优化前缀集合：移除被子网覆盖的前缀
    
    参数:
        prefixes: 前缀集合
    
    返回:
        优化后的前缀集合
    """
    if len(prefixes) <= 1:
        return prefixes
    
    # 按前缀长度排序（短的在前）
    sorted_prefixes = sorted(prefixes, key=lambda x: x.prefixlen)
    optimized = set()
    
    for i, prefix in enumerate(sorted_prefixes):
        is_covered = False
        # 检查是否被已添加的更短前缀覆盖
        for existing in optimized:
            if is_subnet_or_equal(prefix, existing):
                is_covered = True
                break
        
        if not is_covered:
            optimized.add(prefix)
    
    return optimized

def main():
    parser = argparse.ArgumentParser(
        description='过滤IPv6前缀并合并到第一个文档',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s -1 doc1.txt -2 doc2.txt
  %(prog)s -1 base.txt -2 new.txt -o merged.txt -v
  %(prog)s --doc1 prefixes.txt --doc2 more.txt --output combined.txt --verbose

说明:
  1. 检查doc2中的每个前缀是否是doc1中任何前缀的子前缀
  2. 如果不是子前缀，则计数并添加到doc1
  3. 输出合并后的文件
        '''
    )
    
    parser.add_argument(
        '-1', '--doc1',
        type=str,
        required=True,
        help='第一个文档路径（基础前缀集合）'
    )
    
    parser.add_argument(
        '-2', '--doc2',
        type=str,
        required=True,
        help='第二个文档路径（待过滤的前缀集合）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出文件路径（默认: 覆盖第一个文档）'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细信息'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='不保存结果到文件（仅显示统计）'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='优化前缀集合（移除被子网覆盖的前缀）'
    )
    
    args = parser.parse_args()
    
    # 加载两个文档的前缀
    print("正在加载前缀...")
    doc1_prefixes = load_prefixes(args.doc1)
    doc2_prefixes = load_prefixes(args.doc2)
    
    if args.verbose:
        print_prefix_stats(doc1_prefixes, "文档1")
        print_prefix_stats(doc2_prefixes, "文档2")
    
    # 找出新的前缀
    print("\n正在过滤和比较...")
    new_prefixes = filter_new_prefixes(doc1_prefixes, doc2_prefixes)
    
    # 输出统计信息
    print(f"\n=== 过滤结果 ===")
    print(f"文档2中的前缀总数: {len(doc2_prefixes)}")
    print(f"是文档1子前缀的数量: {len(doc2_prefixes) - len(new_prefixes)}")
    print(f"不是文档1子前缀的数量: {len(new_prefixes)}")
    
    if args.verbose and new_prefixes:
        print(f"\n新增的前缀列表（前20个）:")
        for prefix in new_prefixes[:20]:
            print(f"  {prefix}")
        if len(new_prefixes) > 20:
            print(f"  ... 还有 {len(new_prefixes) - 20} 个")
    
    # 保存结果
    if not args.no_save:
        # 合并前缀
        merged_prefixes = doc1_prefixes.union(new_prefixes)
        
        # 可选：优化前缀集合
        if args.optimize:
            print("\n正在优化前缀集合...")
            original_count = len(merged_prefixes)
            merged_prefixes = optimize_prefixes(merged_prefixes)
            print(f"优化后移除了 {original_count - len(merged_prefixes)} 个被覆盖的前缀")
        
        # 确定输出文件
        output_file = args.output if args.output else args.doc1
        
        # 保存合并后的结果
        save_prefixes(output_file, merged_prefixes)
        print(f"\n合并后的前缀已保存到: {output_file}")
        print(f"原有前缀: {len(doc1_prefixes)}, 新增: {len(new_prefixes)}, 总计: {len(merged_prefixes)}")
    
    # 返回新增数量（用于脚本调用）
    return len(new_prefixes)

if __name__ == "__main__":
    sys.exit(main())