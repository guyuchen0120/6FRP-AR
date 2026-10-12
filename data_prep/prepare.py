#!/usr/bin/env python3
"""
Data preparation: link seed FRPs to AS metadata for training.

Input files:
    1. BGP prefix table:  tab-separated "prefix\tAS#" (e.g. 20250123.1600.dat).
       Lines starting with ';' are comments.
    2. AS metadata CSV:   columns "as,org_name,country,category,sub_category"
       (e.g. as_org_category.csv).
    3. Seeds file: one prefix per line, with optional RTT stats:
         "2a02:26f0::/32"
         "2a02:26f0::/32\t180.5\t20.3\t150.0"        (mean, std, min — all in ms)

Outputs (default to the same directory as the seed file):
    - meta.json           : per-seed metadata (for train.py --metadata)
    - as_seed_db.json     : mapping AS# -> list of seed prefixes (for Mode B mimic)
    - vocabs.json         : {org/country/cat1/cat2} name → integer id tables
    - seeds_prepared.txt  : deduped, normalized seed prefix list
    - stats.json          : statistics of the preparation (for sanity checking)

The script performs longest-prefix matching so a seed more specific than any
BGP entry still resolves to its covering BGP route's AS.
"""
import argparse
import csv
import ipaddress
import json
import logging
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =========================================================================
# Longest-prefix matching over IPv6 BGP table
# =========================================================================

class IPv6LPM:
    """
    Lightweight longest-prefix-match structure.

    Stores (prefix_len -> {masked_int: as}). Lookup iterates length 128→0
    (at most 128 steps per query).
    """
    def __init__(self):
        self.by_len: Dict[int, Dict[int, int]] = defaultdict(dict)
        self.sorted_lens: List[int] = []

    def add(self, prefix_str: str, as_num: int):
        try:
            net = ipaddress.IPv6Network(prefix_str, strict=False)
        except Exception:
            return
        key = int(net.network_address)
        self.by_len[net.prefixlen][key] = as_num

    def finalize(self):
        # Iterate from longest to shortest during lookup
        self.sorted_lens = sorted(self.by_len.keys(), reverse=True)

    def lookup(self, prefix_str: str) -> Optional[Tuple[int, int]]:
        """Return (as_num, matched_prefix_len) for the longest BGP prefix
        that covers prefix_str; None if no covering entry."""
        try:
            net = ipaddress.IPv6Network(prefix_str, strict=False)
        except Exception:
            return None
        addr = int(net.network_address)
        for plen in self.sorted_lens:
            if plen > net.prefixlen:
                # Can't be matched by a prefix more specific than ours
                continue
            mask = ((1 << plen) - 1) << (128 - plen) if plen > 0 else 0
            key = addr & mask
            tab = self.by_len[plen]
            if key in tab:
                return tab[key], plen
        return None


def load_bgp_table(path: str) -> IPv6LPM:
    lpm = IPv6LPM()
    n_total = 0
    n_bad = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';') or line.startswith('#'):
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) < 2:
                n_bad += 1
                continue
            prefix_str = parts[0]
            # The AS field may contain sets/paths e.g. "{1234,5678}" — just
            # pick the first integer we find.
            as_str = parts[1].strip()
            as_num: Optional[int] = None
            for tok in as_str.replace('{', ' ').replace('}', ' ').replace(',', ' ').split():
                try:
                    as_num = int(tok)
                    break
                except ValueError:
                    continue
            if as_num is None:
                n_bad += 1
                continue
            lpm.add(prefix_str, as_num)
            n_total += 1

    lpm.finalize()
    logger.info(f"BGP table loaded: {n_total} prefixes across "
                f"{len(lpm.by_len)} distinct lengths (skipped {n_bad})")
    return lpm


# =========================================================================
# AS metadata CSV
# =========================================================================

def load_as_metadata(path: str) -> Dict[int, Dict]:
    out = {}
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                as_num = int(row['as'])
            except (ValueError, KeyError):
                continue
            out[as_num] = {
                "org_name": (row.get("org_name") or "").strip(),
                "country": (row.get("country") or "").strip(),
                "category": (row.get("category") or "").strip(),
                "sub_category": (row.get("sub_category") or "").strip(),
            }
    logger.info(f"AS metadata loaded: {len(out)} ASes")
    return out


# =========================================================================
# Vocabulary construction
# =========================================================================

def build_vocab(items, reserve_zero_for_unknown: bool = True) -> Dict[str, int]:
    """Build a string -> int id map, stable in frequency-descending order."""
    counter = Counter(items)
    # Most common first; ties broken by lexicographic order for determinism
    ordered = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    start = 1 if reserve_zero_for_unknown else 0
    return {name: (start + i) for i, (name, _) in enumerate(ordered)}


def vocab_lookup(vocab: Dict[str, int], key: str) -> int:
    """Return vocab id; unknown → 0."""
    return vocab.get(key, 0)


# =========================================================================
# Seeds parsing
# =========================================================================

def load_seeds(path: str) -> List[Dict]:
    """
    Each seed can be:
        "2a02:26f0::/32"
        "2a02:26f0::/32\t180.5\t20.3\t150.0"
        "2a02:26f0::/32 180.5 20.3 150.0"
    RTT is (mean, std, min) in ms. Missing fields default to 0.
    """
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            prefix_str = parts[0]
            rtt = [0.0, 0.0, 0.0]
            for i in range(1, min(4, len(parts))):
                try:
                    rtt[i - 1] = float(parts[i])
                except ValueError:
                    pass
            try:
                net = ipaddress.IPv6Network(prefix_str, strict=False)
            except Exception:
                logger.warning(f"Skipping invalid prefix: {prefix_str}")
                continue
            out.append({"prefix": str(net), "rtt_stats": rtt})
    logger.info(f"Loaded {len(out)} seed prefixes")
    return out


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data by linking seed FRPs with AS metadata.")
    parser.add_argument('--bgp', default="data/ipasn_20260429.dat",
                        help='BGP prefix→AS file (e.g. 20250123.1600.dat)')
    parser.add_argument('--as-csv',default="data/as_org_category.csv",
                        help='AS metadata CSV (as,org_name,country,category,sub_category)')
    parser.add_argument('--seeds', default="data/seeds.txt",
                        help='Seed FRPs file (one prefix/line, optional RTT columns)')
    parser.add_argument('--out-dir', default="data_fixed",
                        help='Output directory (default: same as --seeds)')
    parser.add_argument('--proto-id', type=int, default=0,
                        help='Protocol id shared by all seeds (default 0 = ICMPv6)')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.seeds))
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load BGP table and AS metadata
    lpm = load_bgp_table(args.bgp)
    as_meta = load_as_metadata(args.as_csv)

    # 2) Load seeds
    seeds = load_seeds(args.seeds)

    # 3) Resolve each seed to an AS
    resolved: List[Dict] = []
    unresolved = 0
    missing_meta = 0
    for s in seeds:
        lookup = lpm.lookup(s["prefix"])
        if lookup is None:
            unresolved += 1
            continue
        as_num, matched_len = lookup
        meta = as_meta.get(as_num, {})
        if not meta:
            missing_meta += 1
        resolved.append({
            "prefix": s["prefix"],
            "rtt_stats": s["rtt_stats"],
            "as_num": as_num,
            "matched_bgp_len": matched_len,
            "org_name": meta.get("org_name", ""),
            "country": meta.get("country", ""),
            "category": meta.get("category", ""),
            "sub_category": meta.get("sub_category", ""),
        })

    logger.info(f"Seeds resolved: {len(resolved)}; "
                f"unresolved: {unresolved}; no_as_meta: {missing_meta}")

    if not resolved:
        logger.error("No seeds could be resolved to an AS. Aborting.")
        sys.exit(1)

    # 4) Build vocabularies from resolved data
    # Remap AS numbers to compact sequential ids (avoids huge embedding table)
    as_vocab = build_vocab([str(r["as_num"]) for r in resolved])
    org_vocab = build_vocab([r["org_name"] for r in resolved if r["org_name"]])
    country_vocab = build_vocab([r["country"] for r in resolved if r["country"]])
    cat1_vocab = build_vocab([r["category"] for r in resolved if r["category"]])
    cat2_vocab = build_vocab([r["sub_category"] for r in resolved if r["sub_category"]])

    logger.info(f"Vocabs built — AS: {len(as_vocab)}, orgs: {len(org_vocab)}, "
                f"countries: {len(country_vocab)}, "
                f"cat1: {len(cat1_vocab)}, cat2: {len(cat2_vocab)}")

    # 5) Construct meta.json and as_seed_db.json
    meta_out: Dict[str, Dict] = {}
    as_seed_db: Dict[str, List[str]] = defaultdict(list)

    for r in resolved:
        meta_out[r["prefix"]] = {
            "as_id": vocab_lookup(as_vocab, str(r["as_num"])),
            "as_num": int(r["as_num"]),   # raw AS number for traceability
            "org_id": vocab_lookup(org_vocab, r["org_name"]),
            "cat1_id": vocab_lookup(cat1_vocab, r["category"]),
            "cat2_id": vocab_lookup(cat2_vocab, r["sub_category"]),
            "proto_id": args.proto_id,
            "geo_id": vocab_lookup(country_vocab, r["country"]),
            "rtt_stats": r["rtt_stats"],
        }
        # as_seed_db keyed by compact as_id so Mode B can look up by the
        # same id the model uses internally
        as_seed_db[str(vocab_lookup(as_vocab, str(r["as_num"])))].append(r["prefix"])

    # 6) Write outputs
    meta_path = os.path.join(out_dir, "meta.json")
    as_db_path = os.path.join(out_dir, "as_seed_db.json")
    vocabs_path = os.path.join(out_dir, "vocabs.json")
    seeds_path = os.path.join(out_dir, "seeds_prepared.txt")
    stats_path = os.path.join(out_dir, "stats.json")

    with open(meta_path, 'w') as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)
    with open(as_db_path, 'w') as f:
        json.dump(as_seed_db, f, indent=2, ensure_ascii=False)
    with open(vocabs_path, 'w') as f:
        json.dump({
            "as_vocab": as_vocab,
            "org_vocab": org_vocab,
            "country_vocab": country_vocab,
            "cat1_vocab": cat1_vocab,
            "cat2_vocab": cat2_vocab,
        }, f, indent=2, ensure_ascii=False)
    with open(seeds_path, 'w') as f:
        for r in resolved:
            f.write(f"{r['prefix']}\n")

    # Stats
    as_counter = Counter(r["as_num"] for r in resolved)
    stats = {
        "total_seeds_input": len(seeds),
        "seeds_resolved": len(resolved),
        "seeds_unresolved": unresolved,
        "seeds_without_as_meta": missing_meta,
        "unique_as_count": len(as_counter),
        "top_10_AS_by_seed_count": as_counter.most_common(10),
        "vocab_sizes": {
            "as": len(as_vocab) + 1,            # +1 for "unknown" slot at id=0
            "org": len(org_vocab) + 1,
            "country (geo)": len(country_vocab) + 1,
            "cat1": len(cat1_vocab) + 1,
            "cat2": len(cat2_vocab) + 1,
        },
        "max_as_num_raw": max(as_counter.keys()) if as_counter else 0,
        "max_as_id_compact": max(as_vocab.values()) if as_vocab else 0,
    }
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Output files:")
    logger.info(f"  {meta_path}  ({len(meta_out)} entries)")
    logger.info(f"  {as_db_path}  ({len(as_seed_db)} ASes)")
    logger.info(f"  {vocabs_path}")
    logger.info(f"  {seeds_path}")
    logger.info(f"  {stats_path}")
    logger.info("=" * 60)
    logger.info(f"Statistics:")
    logger.info(f"  Seeds resolved:        {len(resolved)} / {len(seeds)}")
    logger.info(f"  Unique ASes covered:   {len(as_counter)}")
    # logger.info(f"  Max AS number (raw):   {stats['max_as_num_raw']}")
    logger.info(f"  Max compact AS id:     {stats['max_as_id_compact']}")
    logger.info(f"  Top 3 ASes by seed ct: {as_counter.most_common(3)}")
    # logger.info("")
    # logger.info("Next step — train with:")
    # logger.info(f"  python -m my_design_new train \\")
    # logger.info(f"      --frps {seeds_path} \\")
    # logger.info(f"      --metadata {meta_path} \\")
    # logger.info(f"      --entropy-file <path to exp1_result.json> \\")
    # logger.info(f"      --output-dir my_design_new/checkpoints_v2")
    # logger.info("")
    # logger.info("IMPORTANT: ensure ModelConfig vocab sizes are at least:")
    # logger.info(f"           num_as    >= {stats['vocab_sizes']['as']}")
    # logger.info(f"           num_org   >= {stats['vocab_sizes']['org']}")
    # logger.info(f"           num_cat1  >= {stats['vocab_sizes']['cat1']}")
    # logger.info(f"           num_cat2  >= {stats['vocab_sizes']['cat2']}")
    # logger.info(f"           num_geo   >= {stats['vocab_sizes']['country (geo)']}")


if __name__ == "__main__":
    main()
