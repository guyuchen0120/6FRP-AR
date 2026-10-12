"""
IPv6 地址与前缀工具函数
"""
import ipaddress
import random
from typing import List, Tuple

import torch


def ipv6_prefix_to_bits(prefix_str: str) -> Tuple[List[int], int]:
    """
    将 IPv6 前缀字符串转为二进制位列表。

    Args:
        prefix_str: 如 "2001:db8::/32"

    Returns:
        (bits, prefix_len)
        bits: 长度为 prefix_len 的 0/1 列表
        prefix_len: 前缀长度
    """
    network = ipaddress.IPv6Network(prefix_str, strict=False)
    prefix_len = network.prefixlen
    addr_int = int(network.network_address)
    # 取高 prefix_len 位
    bits = []
    for i in range(prefix_len):
        bit = (addr_int >> (127 - i)) & 1
        bits.append(bit)
    return bits, prefix_len


def bits_to_ipv6_prefix(bits: List[int]) -> str:
    """
    将二进制位列表转为 IPv6 前缀字符串。

    Args:
        bits: 0/1 列表，长度即前缀长度

    Returns:
        标准化前缀字符串，如 "2001:db8::/32"
    """
    prefix_len = len(bits)
    addr_int = 0
    for i, b in enumerate(bits):
        addr_int |= (b << (127 - i))
    addr = ipaddress.IPv6Address(addr_int)
    network = ipaddress.IPv6Network(f"{addr}/{prefix_len}", strict=False)
    return str(network)


def bits_tensor_to_prefix(bits_tensor: torch.Tensor, length: int) -> str:
    """
    将 PyTorch tensor 的二进制位转为 IPv6 前缀字符串。

    Args:
        bits_tensor: shape (seq_len,) 的 0/1 tensor
        length: 实际前缀长度（可能 <= seq_len）

    Returns:
        前缀字符串
    """
    bits = bits_tensor[:length].long().tolist()
    return bits_to_ipv6_prefix(bits)


def prefix_to_bits_tensor(prefix_str: str, max_len: int = 128) -> Tuple[torch.Tensor, int]:
    """
    将前缀字符串转为固定长度的 tensor（右侧零填充）。

    Returns:
        (tensor, actual_len)
        tensor: shape (max_len,) float tensor
        actual_len: 实际前缀长度
    """
    bits, prefix_len = ipv6_prefix_to_bits(prefix_str)
    padded = bits + [0] * (max_len - len(bits))
    return torch.tensor(padded, dtype=torch.float32), prefix_len


def generate_apd_addresses(prefix_str: str, count: int = 16) -> List[str]:
    """
    在前缀内生成 APD 探测地址：遍历前缀后紧接的 4 位 (0-15)，
    其余位随机。

    Args:
        prefix_str: IPv6 前缀
        count: 地址数（默认16）

    Returns:
        IPv6 地址列表
    """
    network = ipaddress.IPv6Network(prefix_str, strict=False)
    plen = network.prefixlen
    base_int = int(network.network_address)
    remaining_bits = 128 - plen

    if remaining_bits < 4:
        addresses = []
        for _ in range(count):
            rand_int = random.randint(
                int(network.network_address),
                int(network.broadcast_address)
            )
            addresses.append(str(ipaddress.IPv6Address(rand_int)))
        return addresses

    tail_bits = remaining_bits - 4
    addresses = []
    for nibble in range(16):
        nibble_shifted = nibble << tail_bits
        tail_random = random.randint(0, (1 << tail_bits) - 1) if tail_bits > 0 else 0
        addr_int = base_int | nibble_shifted | tail_random
        addresses.append(str(ipaddress.IPv6Address(addr_int)))
    return addresses


def is_subnet_of(child_str: str, parent_str: str) -> bool:
    """判断 child 是否为 parent 的子前缀（或相同）"""
    child = ipaddress.IPv6Network(child_str, strict=False)
    parent = ipaddress.IPv6Network(parent_str, strict=False)
    if child.prefixlen < parent.prefixlen:
        return False
    return (int(child.network_address) & int(parent.netmask)) == int(parent.network_address)


def normalize_prefix(prefix_str: str) -> str:
    """规范化前缀（后缀归零 + 压缩表示）"""
    return str(ipaddress.IPv6Network(prefix_str, strict=False))


def hamming_distance_bits(a: List[int], b: List[int]) -> int:
    """两个等长二进制列表的汉明距离"""
    min_len = min(len(a), len(b))
    dist = sum(1 for i in range(min_len) if a[i] != b[i])
    dist += abs(len(a) - len(b))
    return dist


def nibble_set(prefix_str: str) -> set:
    """提取前缀的 nibble 集合（用于 Jaccard 相似度）"""
    bits, plen = ipv6_prefix_to_bits(prefix_str)
    nybbles = set()
    for i in range(0, plen, 4):
        chunk = bits[i:i+4]
        val = 0
        for b in chunk:
            val = (val << 1) | b
        nybbles.add((i // 4, val))
    return nybbles


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard 相似度"""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ============================================================================
# Helpers added for v2 (tree alignment, BGP-constrained generation)
# ============================================================================

def parent_prefix(prefix_str: str, delta: int = 1) -> str:
    """
    Return the parent prefix (shorter by `delta` bits).
    Returns None-equivalent empty string if already at root.
    """
    net = ipaddress.IPv6Network(prefix_str, strict=False)
    new_len = net.prefixlen - delta
    if new_len < 0:
        return None
    return str(ipaddress.IPv6Network(
        (int(net.network_address), new_len), strict=False))


def sibling_prefixes(prefix_str: str) -> List[str]:
    """
    Return all immediate siblings of the given prefix under its parent
    (parent is 1 bit shorter; 2 siblings total including self).
    For a coarser grouping (e.g. 4-bit siblings), use `siblings_at_ancestor`.
    """
    net = ipaddress.IPv6Network(prefix_str, strict=False)
    if net.prefixlen == 0:
        return []
    parent_len = net.prefixlen - 1
    parent_net = ipaddress.IPv6Network(
        (int(net.network_address), parent_len), strict=False)
    sibs = []
    # 2 children of parent at net.prefixlen
    bit_shift = 128 - net.prefixlen
    for i in range(2):
        addr = int(parent_net.network_address) | (i << bit_shift)
        sibs.append(str(ipaddress.IPv6Network(
            (addr, net.prefixlen), strict=False)))
    return sibs


def siblings_at_ancestor(prefix_str: str, ancestor_delta: int = 4) -> List[str]:
    """
    Enumerate all prefixes of same length under the ancestor that is
    `ancestor_delta` bits shorter.
    E.g. for prefix /30 with ancestor_delta=4, returns 2^4 = 16 siblings at /30
    under the common /26 parent.
    """
    net = ipaddress.IPv6Network(prefix_str, strict=False)
    if net.prefixlen < ancestor_delta:
        return [str(net)]
    anc_len = net.prefixlen - ancestor_delta
    anc = ipaddress.IPv6Network(
        (int(net.network_address), anc_len), strict=False)
    sibs = []
    bit_shift = 128 - net.prefixlen
    num_sibs = 1 << ancestor_delta
    for i in range(num_sibs):
        addr = int(anc.network_address) | (i << bit_shift)
        sibs.append(str(ipaddress.IPv6Network(
            (addr, net.prefixlen), strict=False)))
    return sibs


def prefix_length(prefix_str: str) -> int:
    return ipaddress.IPv6Network(prefix_str, strict=False).prefixlen


def address_to_prefix(address: str, prefix_len: int) -> str:
    """Compute the /prefix_len prefix containing the given address."""
    addr = ipaddress.IPv6Address(address)
    return str(ipaddress.IPv6Network((int(addr), prefix_len), strict=False))


def prefix_covers_address(prefix_str: str, address: str) -> bool:
    net = ipaddress.IPv6Network(prefix_str, strict=False)
    addr = ipaddress.IPv6Address(address)
    return int(addr) & int(net.netmask) == int(net.network_address)


def maximal_cover(prefixes: List[str]) -> List[str]:
    """
    From a list of prefixes, return only the maximal (shortest) ones:
    any prefix strictly contained in another is removed.
    """
    result = []
    normalized = [normalize_prefix(p) for p in prefixes]
    for i, p in enumerate(normalized):
        dominated = False
        for j, q in enumerate(normalized):
            if i == j:
                continue
            if p == q:
                continue
            # q strictly contains p → drop p
            if is_subnet_of(p, q) and not is_subnet_of(q, p):
                dominated = True
                break
        if not dominated and p not in result:
            result.append(p)
    return result


def random_address_in_prefix(prefix_str: str) -> str:
    net = ipaddress.IPv6Network(prefix_str, strict=False)
    rand_int = random.randint(
        int(net.network_address), int(net.broadcast_address))
    return str(ipaddress.IPv6Address(rand_int))


def sample_address_under_parent_excluding_child(
        parent_str: str, child_str: str) -> str:
    """
    Sample a random IPv6 address inside `parent` but NOT inside `child`.
    Used by tree alignment's lateral expansion.
    """
    parent = ipaddress.IPv6Network(parent_str, strict=False)
    child = ipaddress.IPv6Network(child_str, strict=False)
    for _ in range(50):
        rand_int = random.randint(
            int(parent.network_address), int(parent.broadcast_address))
        if not prefix_covers_address(child_str, str(ipaddress.IPv6Address(rand_int))):
            return str(ipaddress.IPv6Address(rand_int))
    # Fallback: best effort
    return random_address_in_prefix(parent_str)
