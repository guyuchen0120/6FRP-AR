import ipaddress
import csv
import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from typing import Set, Dict, Tuple, Optional
import bisect

class IPv6PrefixIndex:
    """优化的IPv6前缀索引，支持快速查找覆盖关系"""
    
    def __init__(self, prefixes):
        self.prefixes = sorted(prefixes, key=lambda x: x.prefixlen)
        # 按前缀长度分组，提高查找效率
        self.by_length = {}
        for p in self.prefixes:
            length = p.prefixlen
            if length not in self.by_length:
                self.by_length[length] = []
            self.by_length[length].append(p)
    
    def covers(self, route_net):
        """检查路由是否被任何前缀覆盖（使用优化算法）"""
        # 如果路由的前缀长度小于输入前缀的最小长度，不可能被覆盖
        min_len = min(self.by_length.keys()) if self.by_length else 0
        if route_net.prefixlen < min_len:
            return False
        
        # 从长前缀开始检查（长前缀更具体，更容易覆盖）
        # 只检查长度不超过路由前缀长度的前缀
        for length in sorted(self.by_length.keys(), reverse=True):
            if length > route_net.prefixlen:
                continue
            # 用二分查找快速定位可能匹配的前缀
            for prefix in self.by_length[length]:
                # 快速比较：检查路由是否在prefix的范围内
                if route_net.network_address >= prefix.network_address and \
                   route_net.broadcast_address <= prefix.broadcast_address:
                    return True
        return False

def parse_ipv6_prefix_fast(line):
    """快速解析IPv6前缀，缓存已解析的结果"""
    line = line.strip()
    if not line:
        return None
    try:
        return ipaddress.IPv6Network(line, strict=False)
    except ValueError:
        try:
            return ipaddress.IPv6Network(line + '/128', strict=False)
        except ValueError:
            return None

def read_ipv6_prefixes_fast(file_path):
    """快速读取IPv6前缀文件"""
    prefixes = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 使用列表推导式快速读取
            lines = f.readlines()
            # 批量解析
            for line_num, line in enumerate(tqdm(lines, desc="读取IPv6前缀", unit="行"), 1):
                net = parse_ipv6_prefix_fast(line)
                if net:
                    prefixes.append(net)
                else:
                    print(f"警告: 第{line_num}行 '{line.strip()}' 不是有效的IPv6前缀，已跳过", 
                          file=sys.stderr)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取文件 '{file_path}' 时发生异常: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not prefixes:
        print("警告: 未读取到任何有效的IPv6前缀", file=sys.stderr)
    return prefixes

def read_asn_info_fast(file_path, verbose=False):
    """快速读取ASN组织分类信息文件（CSV格式）"""
    asn_info = {}
    error_count = 0
    line_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 跳过第一行表头
            header = f.readline()
            line_count += 1
            
            # 使用csv.reader批量处理
            reader = csv.reader(f)
            for line_num, parts in enumerate(reader, 2):  # 从第2行开始
                line_count += 1
                
                if not parts or len(parts) < 5:
                    error_count += 1
                    continue
                
                asn = parts[0].strip()
                if not asn:
                    continue
                    
                country = parts[2].strip() if len(parts) > 2 else ""
                category = parts[3].strip() if len(parts) > 3 else ""
                sub_category = parts[4].strip() if len(parts) > 4 else ""
                
                asn_info[asn] = (country, category, sub_category)
                
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取文件 '{file_path}' 时发生异常: {e}", file=sys.stderr)
        sys.exit(1)
    
    if verbose:
        print(f"ASN文件统计: 总行数 {line_count}, 有效ASN记录 {len(asn_info)}, 格式错误 {error_count}", 
              file=sys.stderr)
    
    return asn_info

def process_routes_fast(route_file, prefix_index, asn_info, verbose=False):
    """优化的路由处理函数"""
    covered_prefixes = set()
    covered_asns = set()
    covered_countries = set()
    covered_categories = set()
    covered_sub_categories = set()
    
    total_routes = 0
    ipv6_routes = 0
    covered_routes = 0
    not_found_asns = 0
    
    # 缓存ASN信息查找
    asn_info_cache = asn_info
    
    try:
        # 先获取总行数用于进度条
        total_lines = sum(1 for _ in open(route_file, 'r', encoding='utf-8'))
        
        with open(route_file, 'r', encoding='utf-8') as f:
            # 使用更大缓冲区
            with tqdm(total=total_lines, desc="处理路由", unit="条") as pbar:
                for line in f:
                    total_routes += 1
                    
                    # 快速分割
                    parts = line.split('\t')
                    if len(parts) < 2:
                        pbar.update(1)
                        continue
                    
                    prefix_str = parts[0].strip()
                    asn_str = parts[1].strip()
                    
                    # 快速IPv6解析（使用try-except，但先做简单检查）
                    if ':' not in prefix_str:  # 不是IPv6
                        pbar.update(1)
                        continue
                    
                    try:
                        route_net = ipaddress.IPv6Network(prefix_str, strict=False)
                    except ValueError:
                        pbar.update(1)
                        continue
                    
                    ipv6_routes += 1
                    
                    # 使用优化的索引进行覆盖检查
                    if prefix_index.covers(route_net):
                        covered_routes += 1
                        covered_prefixes.add(prefix_str)  # 直接使用原字符串，避免重复转换
                        covered_asns.add(asn_str)
                        
                        # 快速ASN信息查找
                        asn_info_item = asn_info_cache.get(asn_str)
                        if asn_info_item:
                            country, category, sub_category = asn_info_item
                            if country:
                                covered_countries.add(country)
                            if category:
                                covered_categories.add(category)
                            if sub_category:
                                covered_sub_categories.add(sub_category)
                        else:
                            not_found_asns += 1
                            if verbose and not_found_asns <= 10:
                                print(f"警告: ASN '{asn_str}' 在分类文件中未找到", file=sys.stderr)
                    
                    # 更新进度条
                    pbar.update(1)
                    
                    # 定期更新进度条描述
                    if total_routes % 50000 == 0:
                        pbar.set_postfix({
                            'IPv6': ipv6_routes,
                            '覆盖': covered_routes
                        })
                    
    except FileNotFoundError:
        print(f"错误: 文件 '{route_file}' 不存在", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: 读取文件 '{route_file}' 时发生异常: {e}", file=sys.stderr)
        sys.exit(1)
    
    if verbose:
        print(f"\n路由统计: 总路由 {total_routes}, IPv6路由 {ipv6_routes}, 被覆盖 {covered_routes}")
        if not_found_asns > 0:
            print(f"警告: {not_found_asns} 个ASN在分类文件中未找到", file=sys.stderr)
    
    return {
        'covered_prefixes': covered_prefixes,
        'covered_asns': covered_asns,
        'covered_countries': covered_countries,
        'covered_categories': covered_categories,
        'covered_sub_categories': covered_sub_categories
    }

def main():
    parser = argparse.ArgumentParser(
        description='分析IPv6前缀覆盖的ASN路由信息（优化版）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s -p ipv6_prefixes.txt -r IP-ASN32-DAT -a as_org_category.csv -o result.json
  %(prog)s -p prefixes.txt -r routes.txt -a asn.csv -o output.json -v
        '''
    )
    
    parser.add_argument(
        '-p', '--prefix-file',
        required=True,
        help='IPv6前缀文件路径（每行一个CIDR格式的前缀）'
    )
    
    parser.add_argument(
        '-r', '--route-file',
        default='ipasn_20260429.dat',
        help='IP-ASN32-DAT路由文件路径（每行: 前缀\\tASN）'
    )
    
    parser.add_argument(
        '-a', '--asn-file',
        default='as_org_category.csv',
        help='ASN组织分类CSV文件路径（表头: as\\torg_name\\tcountry\\tcategory\\tsub_category）'
    )
    
    parser.add_argument(
        '-o', '--output-file',
        default='result.json',
        help='输出JSON文件路径（默认: result.json）'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细处理信息'
    )
    
    parser.add_argument(
        '--encoding',
        default='utf-8',
        help='文件编码（默认: utf-8）'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='禁用进度条'
    )
    
    parser.add_argument(
        '--skip-asn',
        action='store_true',
        help='跳过ASN分类信息读取（提高速度）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    for file_path in [args.prefix_file, args.route_file]:
        if not Path(file_path).exists():
            print(f"错误: 文件 '{file_path}' 不存在", file=sys.stderr)
            sys.exit(1)
    
    if args.asn_file and not Path(args.asn_file).exists():
        print(f"警告: ASN文件 '{args.asn_file}' 不存在，将跳过ASN分类统计", file=sys.stderr)
        args.skip_asn = True
    
    if args.verbose:
        print("开始处理（优化版）...", file=sys.stderr)
        print(f"输入前缀文件: {args.prefix_file}", file=sys.stderr)
        print(f"路由文件: {args.route_file}", file=sys.stderr)
        if not args.skip_asn:
            print(f"ASN分类文件: {args.asn_file}", file=sys.stderr)
        print(f"输出文件: {args.output_file}", file=sys.stderr)
    
    # 1. 读取IPv6前缀
    if args.verbose:
        print("\n读取IPv6前缀...", file=sys.stderr)
    input_prefixes = read_ipv6_prefixes_fast(args.prefix_file)
    frp_count = len(input_prefixes)
    total_addresses = sum(net.num_addresses for net in input_prefixes)
    
    if args.verbose:
        print(f"读取了 {frp_count} 个IPv6前缀", file=sys.stderr)
        print(f"总地址数: {total_addresses:,}", file=sys.stderr)
    
    # 如果前缀列表为空，提前结束
    if not input_prefixes:
        print("错误: 没有有效的IPv6前缀可供分析", file=sys.stderr)
        sys.exit(1)
    
    # 2. 构建前缀索引（优化查找）
    if args.verbose:
        print("\n构建前缀索引...", file=sys.stderr)
    prefix_index = IPv6PrefixIndex(input_prefixes)
    
    # 3. 读取ASN分类信息（可选）
    asn_info = {}
    if not args.skip_asn and args.asn_file and Path(args.asn_file).exists():
        if args.verbose:
            print("\n读取ASN分类信息...", file=sys.stderr)
        asn_info = read_asn_info_fast(args.asn_file, args.verbose)
    else:
        if args.verbose:
            print("\n跳过ASN分类信息读取", file=sys.stderr)
    
    # 4. 处理路由文件
    if args.verbose:
        print("\n处理路由文件...", file=sys.stderr)
    
    stats = process_routes_fast(args.route_file, prefix_index, asn_info, args.verbose)
    
    # 5. 汇总结果
    result = {
        "frp_count": frp_count,
        "total_addresses": float(total_addresses),
        "covered_prefix_count": len(stats['covered_prefixes']),
        "covered_as_count": len(stats['covered_asns']),
        "covered_country_count": len(stats['covered_countries']),
        "covered_category_count": len(stats['covered_categories']),
        "covered_sub_category_count": len(stats['covered_sub_categories'])
    }
    
    # 6. 输出JSON
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        if args.verbose:
            print(f"\n结果已保存到: {args.output_file}", file=sys.stderr)
            print("\n统计结果:", file=sys.stderr)
            for key, value in result.items():
                print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}", 
                      file=sys.stderr)
    except Exception as e:
        print(f"错误: 写入输出文件 '{args.output_file}' 时发生异常: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()