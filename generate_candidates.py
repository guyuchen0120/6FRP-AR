#!/usr/bin/env python3
"""
Candidate FRP Generator (Design A + Design C)
=============================================
Combines two complementary no-GPU generators:
  - Design A: Statistical Nibble Generator (6Gen-style)
              Learns per-nibble conditional probability distributions from seeds,
              generates candidates by chain-sampling.
  - Design C: Prefix Tree Graph Gap Analyzer (6Graph-style)
              Builds a nibble-level prefix tree, finds unexplored sibling
              nibble values ("gaps"), generates candidates to fill them.

Usage:
    python3 generate_candidates.py --seeds seeds.txt --output candidates.txt
    python3 generate_candidates.py --seeds seeds.txt --output candidates.txt \\
        --num-stat 2000 --num-graph 2000 --lock-threshold 0.3 --gap-radius 4
    python3 generate_candidates.py --seeds seeds.txt --output candidates.txt \\
        --bgp-prefix 2001:db8::/32 --target-prefix-len 48

Input:
    seeds.txt  — one IPv6 prefix per line (e.g. "2001:610:9000::/40")
                 Lines starting with '#' are treated as comments.

Output:
    candidates.txt — one candidate IPv6 prefix per line, deduplicated,
                     seeds themselves excluded.
"""

import argparse
import ipaddress
import os
import sys
import time
from typing import List, Optional, Set

# Ensure project root is on path
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from generators.nibble_statistical import NibbleStatisticalGenerator
from generators.graph_generator import PrefixTreeGraphGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_seeds(path: str) -> List[str]:
    """Load seed prefixes from a text file (one per line, skip comments)."""
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                # Normalize: parse and re-serialize
                net = ipaddress.IPv6Network(line, strict=False)
                seeds.append(str(net))
            except ValueError:
                print(f"  [WARN] Skipping invalid prefix: {line}", file=sys.stderr)
    return seeds


def write_candidates(candidates: List[str], path: str):
    """Write candidate prefixes to a text file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        for c in candidates:
            f.write(c + '\n')


def filter_candidates(
    candidates: List[str],
    seeds_set: Set[str],
    bgp_prefix: Optional[str] = None,
) -> List[str]:
    """
    Deduplicate candidates and remove:
      - seeds themselves (exact match)
      - any candidate that is a subnet of an existing seed (sub-prefix)
    Optionally filter to only keep candidates within a BGP prefix.
    """
    seen: Set[str] = set(seeds_set)
    result = []

    # Pre-parse all seed networks for subnet checking
    seed_nets = []
    for s in seeds_set:
        try:
            seed_nets.append(ipaddress.IPv6Network(s, strict=False))
        except ValueError:
            pass

    bgp_net = None
    if bgp_prefix:
        try:
            bgp_net = ipaddress.IPv6Network(bgp_prefix, strict=False)
        except ValueError:
            print(f"  [WARN] Invalid BGP prefix: {bgp_prefix}", file=sys.stderr)

    filtered_exact = 0
    filtered_subnet = 0

    for c in candidates:
        # 1. Exact match (seed itself or already seen)
        if c in seen:
            filtered_exact += 1
            continue

        try:
            c_net = ipaddress.IPv6Network(c, strict=False)
        except ValueError:
            continue

        # 2. Sub-prefix of any existing seed → skip
        is_subnet = any(c_net.subnet_of(s_net) for s_net in seed_nets)
        if is_subnet:
            filtered_subnet += 1
            continue

        # 3. BGP prefix constraint (optional)
        if bgp_net is not None:
            if not (c_net.subnet_of(bgp_net) or c_net == bgp_net):
                continue

        seen.add(c)
        result.append(c)

    print(f"  Filtered exact/dup: {filtered_exact}, sub-prefix of seed: {filtered_subnet}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate candidate FRP prefixes using Design A + Design C",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 generate_candidates.py --seeds seeds.txt --output candidates.txt

  # Generate more candidates with custom settings
  python3 generate_candidates.py --seeds seeds.txt --output candidates.txt \\
      --num-stat 3000 --num-graph 3000 --lock-threshold 0.25 --gap-radius 5

  # Constrain to a specific BGP prefix
  python3 generate_candidates.py --seeds seeds.txt --output candidates.txt \\
      --bgp-prefix 2001:db8::/32 --target-prefix-len 48

  # Only run one generator
  python3 generate_candidates.py --seeds seeds.txt --output candidates.txt \\
      --no-graph
  python3 generate_candidates.py --seeds seeds.txt --output candidates.txt \\
      --no-stat
""")

    # I/O
    parser.add_argument('--seeds', required=True,
                        help='Input: text file with seed FRP prefixes (one per line)')
    parser.add_argument('--output', required=True,
                        help='Output: text file to write candidate FRP prefixes')

    # Design A (Statistical) settings
    parser.add_argument('--num-stat', type=int, default=2000,
                        help='Number of candidates from Design A (statistical). Default: 2000')
    parser.add_argument('--lock-threshold', type=float, default=0.3,
                        help='Entropy threshold for locking nibble positions in Design A. '
                             'Lower = lock more positions (more conservative). Default: 0.3')
    parser.add_argument('--no-bigram', action='store_true',
                        help='Disable bigram (conditional) sampling in Design A')
    parser.add_argument('--no-stat', action='store_true',
                        help='Skip Design A (statistical generator)')

    # Design C (Graph) settings
    parser.add_argument('--num-graph', type=int, default=2000,
                        help='Number of candidates from Design C (graph gap). Default: 2000')
    parser.add_argument('--gap-radius', type=int, default=3,
                        help='Nibble search radius for gap analysis in Design C. Default: 3')
    parser.add_argument('--max-nibble-depth', type=int, default=16,
                        help='Max nibble depth for prefix tree in Design C (16 = /64). Default: 16')
    parser.add_argument('--no-graph', action='store_true',
                        help='Skip Design C (graph gap generator)')

    # Shared settings
    parser.add_argument('--target-prefix-len', type=int, default=None,
                        help='Target output prefix length in bits (e.g. 48 for /48). '
                             'Default: inferred from seeds (median length)')
    parser.add_argument('--bgp-prefix', type=str, default=None,
                        help='Constrain all candidates to be within this BGP prefix')

    args = parser.parse_args()

    if args.no_stat and args.no_graph:
        parser.error("Cannot use both --no-stat and --no-graph at the same time.")

    # ── Load seeds ──
    print(f"Loading seeds from: {args.seeds}")
    seeds = load_seeds(args.seeds)
    if not seeds:
        print("ERROR: No valid seeds found. Aborting.", file=sys.stderr)
        sys.exit(1)
    seeds_set = set(seeds)
    print(f"  Loaded {len(seeds)} seeds")

    total_start = time.time()
    all_candidates: List[str] = []

    # ── Design A: Statistical Nibble Generator ──
    if not args.no_stat:
        print(f"\n[Design A] Statistical Nibble Generator")
        print(f"  lock_threshold={args.lock_threshold}, "
              f"bigram={not args.no_bigram}, "
              f"num_samples={args.num_stat}")
        t0 = time.time()

        gen_a = NibbleStatisticalGenerator(
            lock_entropy_threshold=args.lock_threshold,
            use_bigram=not args.no_bigram,
            smoothing=0.01,
        )
        gen_a.fit(seeds)

        stats_a = gen_a.get_entropy_summary()
        print(f"  Locked nibble positions ({len(stats_a['locked_positions'])}): "
              f"{stats_a['locked_positions']}")
        print(f"  Free nibble positions  ({len(stats_a['free_positions'])}): "
              f"{stats_a['free_positions']}")
        print(f"  Target prefix length: /{stats_a['target_nibble_len'] * 4}")

        candidates_a = gen_a.generate(
            num_samples=args.num_stat,
            target_prefix_len=args.target_prefix_len,
            bgp_prefix=args.bgp_prefix,
        )
        elapsed_a = time.time() - t0
        print(f"  Generated {len(candidates_a)} candidates in {elapsed_a:.2f}s")
        all_candidates.extend(candidates_a)

    # ── Design C: Prefix Tree Graph Gap Analyzer ──
    if not args.no_graph:
        print(f"\n[Design C] Prefix Tree Graph Gap Analyzer")
        print(f"  max_nibble_depth={args.max_nibble_depth}, "
              f"gap_radius={args.gap_radius}, "
              f"num_samples={args.num_graph}")
        t0 = time.time()

        gen_c = PrefixTreeGraphGenerator(
            max_nibble_depth=args.max_nibble_depth,
            gap_search_radius=args.gap_radius,
            prioritize_adjacent=True,
        )
        gen_c.fit(seeds)

        tree_stats = gen_c.get_tree_stats()
        print(f"  Tree: {tree_stats['tree_nodes']} nodes, "
              f"{tree_stats['total_gap_nibble_values']} gap nibble values")

        candidates_c = gen_c.generate(
            num_samples=args.num_graph,
            target_prefix_len=args.target_prefix_len,
            bgp_prefix=args.bgp_prefix,
            already_explored=seeds_set,
        )
        elapsed_c = time.time() - t0
        print(f"  Generated {len(candidates_c)} candidates in {elapsed_c:.2f}s")
        all_candidates.extend(candidates_c)

    # ── Merge, deduplicate, filter ──
    print(f"\n[Merge] Total raw candidates: {len(all_candidates)}")
    final_candidates = filter_candidates(all_candidates, seeds_set, args.bgp_prefix)
    print(f"  After dedup & seed removal: {len(final_candidates)}")

    # ── Write output ──
    write_candidates(final_candidates, args.output)
    total_elapsed = time.time() - total_start

    print(f"\n{'─' * 60}")
    print(f"  Seeds input       : {len(seeds)}")
    if not args.no_stat:
        print(f"  Design A output   : {len(candidates_a)}")
    if not args.no_graph:
        print(f"  Design C output   : {len(candidates_c)}")
    print(f"  Final candidates  : {len(final_candidates)}")
    print(f"  Output file       : {args.output}")
    print(f"  Total time        : {total_elapsed:.2f}s")
    print(f"{'─' * 60}")


if __name__ == '__main__':
    main()
