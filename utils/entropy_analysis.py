#!/usr/bin/env python3
"""
Bit-level Information Entropy Analysis for IPv6 FRP Prefixes.

Given a seed file (one prefix per line), this module:
  1. Converts each prefix to a 128-bit binary representation
  2. Computes the Shannon entropy H(bit_i) for each bit position i
  3. Derives per-bit temperature weights for the generator:
     - Low entropy (bits are consistent across seeds) → low temperature (conservative)
     - High entropy (bits vary across seeds) → higher temperature (allow variation)

The key insight: in mimic mode, high-order bits (positions 0-11, i.e., the first
3 nibbles) tend to have very low entropy because they represent the network/AS
allocation prefix. The generator should preserve these bits and focus variation
on higher-entropy positions.

Usage:
    # Analyze and save weights
    python3 -m utils.entropy_analysis \
        --seeds output/prepared_new/seeds_prepared.txt \
        --output output/prepared_new/bit_entropy_weights.json

    # Analyze per-AS (group seeds by AS from meta.json)
    python3 -m utils.entropy_analysis \
        --seeds output/prepared_new/seeds_prepared.txt \
        --meta output/prepared_new/meta.json \
        --per-as --output output/prepared_new/as_entropy_weights.json
"""

import argparse
import ipaddress
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent to path for imports
_here = os.path.dirname(os.path.abspath(__file__))
_pkg = os.path.dirname(_here)
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)


# ============================================================================
# Lightweight IPv6 helpers (no torch dependency for standalone usage)
# ============================================================================

def _ipv6_prefix_to_bits(prefix_str: str) -> Tuple[List[int], int]:
    """
    Convert IPv6 prefix string to binary bit list.
    Lightweight version that doesn't require torch.
    """
    network = ipaddress.IPv6Network(prefix_str, strict=False)
    prefix_len = network.prefixlen
    addr_int = int(network.network_address)
    bits = []
    for i in range(prefix_len):
        bit = (addr_int >> (127 - i)) & 1
        bits.append(bit)
    return bits, prefix_len


def _normalize_prefix(prefix_str: str) -> str:
    """Normalize an IPv6 prefix string."""
    try:
        net = ipaddress.IPv6Network(prefix_str.strip(), strict=False)
        return str(net)
    except Exception:
        return prefix_str.strip()


# Try to import from utils.ipv6_utils if available (for consistency),
# fall back to local implementations
try:
    from utils.ipv6_utils import ipv6_prefix_to_bits, normalize_prefix
except (ImportError, ModuleNotFoundError):
    ipv6_prefix_to_bits = _ipv6_prefix_to_bits
    normalize_prefix = _normalize_prefix


# ============================================================================
# Core: Compute per-bit entropy
# ============================================================================

def prefix_to_full_bits(prefix: str, total_bits: int = 128) -> Optional[List[int]]:
    """
    Convert an IPv6 prefix to a full 128-bit list.
    Only the bits within the prefix length are meaningful;
    bits beyond prefix_len are set to 0.
    """
    try:
        bits, plen = ipv6_prefix_to_bits(prefix)
        # Pad to total_bits
        full = list(bits[:total_bits])
        while len(full) < total_bits:
            full.append(0)
        return full
    except Exception:
        return None


def compute_bit_entropy(
    prefixes: List[str],
    max_bit_position: int = 128,
    use_prefix_len_mask: bool = True,
) -> Dict:
    """
    Compute Shannon entropy H(X_i) for each bit position i across all prefixes.

    H(X_i) = -p0*log2(p0) - p1*log2(p1)
    where p0 = fraction of prefixes with bit i = 0, p1 = 1 - p0.

    Only bits within each prefix's declared prefix_len are counted.

    Args:
        prefixes: list of IPv6 prefix strings (e.g., "2001:db8::/32")
        max_bit_position: analyze up to this bit position (default 128)
        use_prefix_len_mask: if True, only count a bit if it's within the
                             prefix's declared length

    Returns:
        dict with:
          - "entropy": list of 128 floats (H per position, 0.0 to 1.0)
          - "p1": list of 128 floats (probability of bit=1 at each position)
          - "count": list of 128 ints (number of prefixes contributing to each position)
          - "num_prefixes": total prefixes analyzed
          - "prefix_len_distribution": dict of prefix_len -> count
    """
    # Count bit=1 and total observations per position
    ones_count = np.zeros(max_bit_position, dtype=np.float64)
    total_count = np.zeros(max_bit_position, dtype=np.float64)
    plen_dist = defaultdict(int)

    for prefix in prefixes:
        try:
            bits, plen = ipv6_prefix_to_bits(normalize_prefix(prefix))
        except Exception:
            continue

        plen_dist[plen] += 1
        effective_len = min(plen, max_bit_position) if use_prefix_len_mask else max_bit_position

        for i in range(effective_len):
            if i < len(bits):
                ones_count[i] += bits[i]
            total_count[i] += 1

    # Compute entropy per position
    entropy = np.zeros(max_bit_position, dtype=np.float64)
    p1_arr = np.zeros(max_bit_position, dtype=np.float64)

    for i in range(max_bit_position):
        if total_count[i] == 0:
            entropy[i] = 0.0
            p1_arr[i] = 0.0
            continue

        p1 = ones_count[i] / total_count[i]
        p0 = 1.0 - p1
        p1_arr[i] = p1

        # Shannon entropy (binary)
        h = 0.0
        if p0 > 0:
            h -= p0 * math.log2(p0)
        if p1 > 0:
            h -= p1 * math.log2(p1)
        entropy[i] = h

    return {
        "entropy": entropy.tolist(),
        "p1": p1_arr.tolist(),
        "count": total_count.astype(int).tolist(),
        "num_prefixes": len(prefixes),
        "prefix_len_distribution": dict(plen_dist),
    }


def compute_per_as_entropy(
    prefixes: List[str],
    meta_data: Dict[str, Dict],
    max_bit_position: int = 128,
) -> Dict[int, Dict]:
    """
    Compute per-bit entropy grouped by AS.

    Args:
        prefixes: list of prefix strings
        meta_data: {prefix_str: {as_id: int, ...}}

    Returns:
        {as_id: entropy_result_dict}
    """
    # Group prefixes by AS
    as_to_prefixes = defaultdict(list)
    for p in prefixes:
        p_norm = normalize_prefix(p)
        meta = meta_data.get(p_norm, {})
        as_id = meta.get("as_id", 0)
        as_to_prefixes[as_id].append(p_norm)

    results = {}
    for as_id, as_prefixes in as_to_prefixes.items():
        if len(as_prefixes) < 2:
            # Need at least 2 prefixes to compute meaningful entropy
            continue
        results[as_id] = compute_bit_entropy(as_prefixes, max_bit_position)
        results[as_id]["num_seeds_in_as"] = len(as_prefixes)

    return results


# ============================================================================
# Entropy → Per-bit Temperature Mapping
# ============================================================================

def entropy_to_temperature(
    entropy: List[float],
    base_temperature: float = 1.0,
    min_temperature: float = 0.01,
    max_temperature: float = 2.0,
    low_entropy_threshold: float = 0.1,
    high_entropy_threshold: float = 0.8,
    lock_below: float = 0.05,
) -> List[float]:
    """
    Convert per-bit entropy values to per-bit temperature weights.

    Strategy:
      - Entropy ≈ 0 (all seeds agree): temperature → min_temperature (nearly deterministic)
      - Entropy < low_entropy_threshold: temperature = min_temperature to base * 0.3
      - Entropy in [low, high]: linear interpolation
      - Entropy > high_entropy_threshold: temperature = base to max_temperature

    Special: if entropy < lock_below, the bit is effectively "locked" (temp → 0.01)
    meaning the model's prediction will be nearly deterministic (argmax-like).

    Args:
        entropy: list of 128 entropy values (0.0 to 1.0)
        base_temperature: the "normal" temperature for medium-entropy bits
        min_temperature: minimum temperature for very low entropy bits
        max_temperature: maximum temperature for high entropy bits
        low_entropy_threshold: below this, bits are considered "conservative"
        high_entropy_threshold: above this, bits are considered "free"
        lock_below: entropy below this → effectively locked

    Returns:
        list of 128 temperature values
    """
    temperatures = []

    for h in entropy:
        if h < lock_below:
            # Nearly zero entropy: lock this bit (argmax-like behavior)
            t = min_temperature
        elif h < low_entropy_threshold:
            # Low entropy: conservative, interpolate between min and 0.3*base
            ratio = (h - lock_below) / max(low_entropy_threshold - lock_below, 1e-9)
            t = min_temperature + ratio * (base_temperature * 0.3 - min_temperature)
        elif h < high_entropy_threshold:
            # Medium entropy: interpolate between 0.3*base and base
            ratio = (h - low_entropy_threshold) / max(high_entropy_threshold - low_entropy_threshold, 1e-9)
            t = base_temperature * 0.3 + ratio * (base_temperature * 0.7)
        else:
            # High entropy: allow more variation
            ratio = min((h - high_entropy_threshold) / max(1.0 - high_entropy_threshold, 1e-9), 1.0)
            t = base_temperature + ratio * (max_temperature - base_temperature)

        temperatures.append(round(t, 4))

    return temperatures


def compute_generation_weights(
    seeds_file: str,
    meta_file: Optional[str] = None,
    per_as: bool = False,
    base_temperature: float = 1.0,
    seq_len: int = 128,
) -> Dict:
    """
    High-level function: read seeds, compute entropy, derive per-bit temperatures.

    Returns:
        {
            "global": {
                "entropy": [...],
                "temperature": [...],
                "p1": [...],
                ...
            },
            "per_as": {  # only if per_as=True and meta_file provided
                as_id: {"entropy": [...], "temperature": [...], ...},
                ...
            }
        }
    """
    # Read seeds
    with open(seeds_file) as f:
        prefixes = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    if not prefixes:
        raise ValueError(f"No prefixes found in {seeds_file}")

    # Global entropy
    global_result = compute_bit_entropy(prefixes, max_bit_position=seq_len)
    global_result["temperature"] = entropy_to_temperature(
        global_result["entropy"], base_temperature=base_temperature)

    result = {"global": global_result}

    # Per-AS entropy
    if per_as and meta_file and os.path.exists(meta_file):
        with open(meta_file) as f:
            meta_data = json.load(f)

        per_as_results = compute_per_as_entropy(prefixes, meta_data, seq_len)
        per_as_with_temp = {}
        for as_id, as_result in per_as_results.items():
            as_result["temperature"] = entropy_to_temperature(
                as_result["entropy"], base_temperature=base_temperature)
            per_as_with_temp[as_id] = as_result

        result["per_as"] = per_as_with_temp

    return result


# ============================================================================
# Visualization helpers
# ============================================================================

def print_entropy_summary(result: Dict, top_n: int = 20):
    """Print a human-readable summary of the entropy analysis."""
    entropy = result["entropy"]
    temperature = result["temperature"]
    p1 = result["p1"]
    count = result["count"]

    print(f"\n{'═' * 80}")
    print(f"  Bit-level Entropy Analysis Summary")
    print(f"  Prefixes analyzed: {result['num_prefixes']}")
    print(f"{'─' * 80}")

    # Overall stats
    ent_arr = np.array(entropy)
    print(f"\n  Entropy statistics (128 bit positions):")
    print(f"    Mean:   {ent_arr.mean():.4f}")
    print(f"    Median: {np.median(ent_arr):.4f}")
    print(f"    Min:    {ent_arr.min():.4f} (bit {ent_arr.argmin()})")
    print(f"    Max:    {ent_arr.max():.4f} (bit {ent_arr.argmax()})")

    # Nibble-level summary (every 4 bits)
    print(f"\n  Per-nibble (4-bit) entropy summary:")
    print(f"  {'Nibble':>6} {'Bits':>10} {'Avg H':>8} {'Avg Temp':>10} {'Interpretation'}")
    print(f"  {'─'*6} {'─'*10} {'─'*8} {'─'*10} {'─'*30}")

    for nib in range(32):  # 128/4 = 32 nibbles
        start = nib * 4
        end = min(start + 4, 128)
        avg_h = np.mean(entropy[start:end])
        avg_t = np.mean(temperature[start:end])

        if avg_h < 0.05:
            interp = "LOCKED (nearly constant)"
        elif avg_h < 0.2:
            interp = "Conservative (low variation)"
        elif avg_h < 0.6:
            interp = "Moderate variation"
        elif avg_h < 0.9:
            interp = "High variation"
        else:
            interp = "Maximum variation"

        print(f"  {nib:>6} [{start:>3}-{end-1:>3}] {avg_h:>8.4f} {avg_t:>10.4f} {interp}")

    # Top low-entropy positions (most constrained)
    print(f"\n  Top {top_n} lowest-entropy bit positions (most constrained):")
    sorted_idx = np.argsort(ent_arr)
    for rank, idx in enumerate(sorted_idx[:top_n]):
        print(f"    #{rank+1:>2} bit {idx:>3} (nibble {idx//4:>2}): "
              f"H={entropy[idx]:.4f}, p1={p1[idx]:.4f}, temp={temperature[idx]:.4f}, "
              f"n={count[idx]}")

    # Top high-entropy positions (most variable)
    print(f"\n  Top {top_n} highest-entropy bit positions (most variable):")
    for rank, idx in enumerate(sorted_idx[-top_n:][::-1]):
        print(f"    #{rank+1:>2} bit {idx:>3} (nibble {idx//4:>2}): "
              f"H={entropy[idx]:.4f}, p1={p1[idx]:.4f}, temp={temperature[idx]:.4f}, "
              f"n={count[idx]}")

    # Prefix length distribution
    plen_dist = result.get("prefix_len_distribution", {})
    if plen_dist:
        print(f"\n  Prefix length distribution:")
        for plen in sorted(plen_dist.keys()):
            bar = '█' * min(int(plen_dist[plen] / max(plen_dist.values()) * 30), 30)
            print(f"    /{plen:<3}: {plen_dist[plen]:>5} {bar}")

    print(f"{'═' * 80}\n")


def plot_entropy(result: Dict, output_path: str = None):
    """
    Generate a matplotlib plot of per-bit entropy and temperature.
    Saves to output_path if provided, otherwise shows interactively.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    entropy = result["entropy"]
    temperature = result["temperature"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    positions = list(range(128))

    # Plot 1: Entropy
    colors = []
    for h in entropy:
        if h < 0.05:
            colors.append('#2ecc71')  # green = locked
        elif h < 0.3:
            colors.append('#f39c12')  # orange = conservative
        elif h < 0.7:
            colors.append('#3498db')  # blue = moderate
        else:
            colors.append('#e74c3c')  # red = high variation

    ax1.bar(positions, entropy, color=colors, width=1.0, edgecolor='none')
    ax1.set_ylabel('Shannon Entropy H(bit_i)')
    ax1.set_title('Per-bit Information Entropy of Seed FRP Prefixes')
    ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, label='Low threshold')
    ax1.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, label='High threshold')

    # Add stage boundaries
    ax1.axvline(x=48, color='black', linestyle='-', alpha=0.3, linewidth=2)
    ax1.axvline(x=64, color='black', linestyle='-', alpha=0.3, linewidth=2)
    ax1.text(24, ax1.get_ylim()[1]*0.9, 'Coarse\n[0-47]', ha='center', fontsize=9)
    ax1.text(56, ax1.get_ylim()[1]*0.9, 'Medium\n[48-63]', ha='center', fontsize=9)
    ax1.text(96, ax1.get_ylim()[1]*0.9, 'Fine\n[64-127]', ha='center', fontsize=9)

    # Legend
    patches = [
        mpatches.Patch(color='#2ecc71', label='Locked (H<0.05)'),
        mpatches.Patch(color='#f39c12', label='Conservative (H<0.3)'),
        mpatches.Patch(color='#3498db', label='Moderate (0.3≤H<0.7)'),
        mpatches.Patch(color='#e74c3c', label='High variation (H≥0.7)'),
    ]
    ax1.legend(handles=patches, loc='upper right', fontsize=8)

    # Plot 2: Temperature
    ax2.bar(positions, temperature, color='#9b59b6', width=1.0, edgecolor='none', alpha=0.7)
    ax2.set_xlabel('Bit Position')
    ax2.set_ylabel('Generation Temperature')
    ax2.set_title('Per-bit Temperature for Mimic-mode Generation')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Base temp=1.0')
    ax2.axvline(x=48, color='black', linestyle='-', alpha=0.3, linewidth=2)
    ax2.axvline(x=64, color='black', linestyle='-', alpha=0.3, linewidth=2)

    # Highlight bits 0-11 (first 3 nibbles)
    ax2.axvspan(0, 12, alpha=0.1, color='red', label='Bits 0-11 (high-order)')

    ax2.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-bit information entropy of IPv6 FRP seed prefixes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python3 -m utils.entropy_analysis --seeds output/prepared_new/seeds_prepared.txt

  # With per-AS breakdown and plot
  python3 -m utils.entropy_analysis \\
      --seeds output/prepared_new/seeds_prepared.txt \\
      --meta output/prepared_new/meta.json \\
      --per-as --plot --output output/prepared_new/bit_entropy_weights.json

  # Custom temperature parameters
  python3 -m utils.entropy_analysis \\
      --seeds data/seeds.txt --base-temp 0.8 --lock-below 0.03
""")

    parser.add_argument('--seeds', required=True,
                        help='Path to seed prefixes file (one per line)')
    parser.add_argument('--meta', default=None,
                        help='Path to meta.json for per-AS analysis')
    parser.add_argument('--per-as', action='store_true',
                        help='Compute per-AS entropy (requires --meta)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON path for entropy weights')
    parser.add_argument('--plot', action='store_true',
                        help='Generate entropy/temperature plot')
    parser.add_argument('--plot-output', default=None,
                        help='Plot output path (default: <output_dir>/entropy_plot.png)')
    parser.add_argument('--base-temp', type=float, default=1.0,
                        help='Base temperature for medium-entropy bits (default: 1.0)')
    parser.add_argument('--lock-below', type=float, default=0.05,
                        help='Entropy threshold below which bits are locked (default: 0.05)')
    parser.add_argument('--summary', action='store_true', default=True,
                        help='Print summary to stdout (default: True)')
    parser.add_argument('--no-summary', action='store_true',
                        help='Suppress summary output')

    args = parser.parse_args()

    if not os.path.exists(args.seeds):
        print(f"Error: seeds file not found: {args.seeds}")
        sys.exit(1)

    # Compute
    print(f"Analyzing entropy for seeds: {args.seeds}")
    result = compute_generation_weights(
        seeds_file=args.seeds,
        meta_file=args.meta,
        per_as=args.per_as,
        base_temperature=args.base_temp,
        seq_len=128,
    )

    # Summary
    if not args.no_summary:
        print_entropy_summary(result["global"])

        if args.per_as and "per_as" in result:
            print(f"\n  Per-AS entropy computed for {len(result['per_as'])} AS groups")
            # Show top 5 AS by seed count
            sorted_as = sorted(result["per_as"].items(),
                               key=lambda x: x[1]["num_seeds_in_as"], reverse=True)
            for as_id, as_result in sorted_as[:5]:
                avg_h = np.mean(as_result["entropy"][:48])
                print(f"    AS {as_id}: {as_result['num_seeds_in_as']} seeds, "
                      f"avg coarse entropy={avg_h:.4f}")

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        # Convert numpy types for JSON serialization
        save_data = {
            "global": result["global"],
        }
        if "per_as" in result:
            save_data["per_as"] = {
                str(k): v for k, v in result["per_as"].items()
            }
        with open(args.output, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"\nEntropy weights saved to: {args.output}")

    # Plot
    if args.plot:
        plot_path = args.plot_output
        if not plot_path and args.output:
            plot_path = os.path.splitext(args.output)[0] + "_plot.png"
        elif not plot_path:
            plot_path = "entropy_plot.png"
        plot_entropy(result["global"], plot_path)


if __name__ == "__main__":
    main()
