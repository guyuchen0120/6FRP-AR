#!/usr/bin/env python3
"""
HMamba FRP candidate generator (test / inference).

Per-AS adaptive generation:
  Reads all seeds with metadata, groups them by AS, and for each AS
  dynamically decides:

    (1) num_samples — how many candidates to draw, scaled by group size.
        large AS  -> more samples (cap at --max-samples-per-as)
        small AS  -> fewer samples (>= --min-samples-per-as)

    (2) target_prefix_len — adaptive output prefix length.
        Use median seed prefix length, possibly clipped or shifted.

    (3) constrain_bits — how many leading bits are pinned from the AS's
        common ancestor (computed from intersection of all seed bits in
        the AS). This ensures candidates stay inside the AS's own BGP
        envelope, but still allows divergence below.

    (4) sampling temperature — AS-feature-aware:
          high-entropy AS  -> higher temperature (more diverse)
          low-entropy  AS  -> lower temperature  (more conservative)
        Combined with the per-bit entropy weight if provided.

    (5) num_paths — how many independent generation passes (different
        seeds-as-context) per candidate, scaled with group size.

After generation, candidates are filtered:
  - dedup
  - drop exact match with any seed
  - drop strict subnets of any seed

Usage:

  python3 test_hmamba.py \
      --ckpt    output/hmamba/best.pt \
      --seeds   output/prepared_new/seeds_prepared.txt \
      --meta    output/prepared_new/meta.json \
      --vocabs  output/prepared_new/vocabs.json \
      --entropy output/prepared_new/bit_entropy_weights.json \
      --output  output/hmamba/candidates.txt \
      --device  auto \
      --total-budget 5000 \
      --min-samples-per-as 2 \
      --max-samples-per-as 200
"""
import argparse
import ipaddress
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

try:
    from .config import Config
    from .generators.hmamba import HMambaFRPGenerator, MASK_ID, load_entropy_weights
    from .utils.ipv6_utils import (
        ipv6_prefix_to_bits, bits_to_ipv6_prefix, normalize_prefix,
        is_subnet_of,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config
    from generators.hmamba import HMambaFRPGenerator, MASK_ID, load_entropy_weights
    from utils.ipv6_utils import (
        ipv6_prefix_to_bits, bits_to_ipv6_prefix, normalize_prefix,
        is_subnet_of,
    )

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Loading
# ============================================================================

def load_seeds_with_meta(seeds_file: str,
                         meta_file: str) -> List[Dict]:
    with open(meta_file) as f:
        meta_map = json.load(f)
    items = []
    with open(seeds_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                p = normalize_prefix(line)
            except Exception:
                continue
            m = meta_map.get(p)
            if not m:
                continue
            items.append({"prefix": p, "meta": {
                "as_id":     int(m.get("as_id", 0)),
                "org_id":    int(m.get("org_id", 0)),
                "cat1_id":   int(m.get("cat1_id", 0)),
                "cat2_id":   int(m.get("cat2_id", 0)),
                "proto_id":  int(m.get("proto_id", 0)),
                "geo_id":    int(m.get("geo_id", 0)),
                "rtt_stats": m.get("rtt_stats", [0.0, 0.0, 0.0]),
            }})
    logger.info(f"Loaded {len(items)} seeds with metadata "
                f"({len(set(s['meta']['as_id'] for s in items))} unique AS)")
    return items


def group_by_as(seeds: List[Dict]) -> Dict[int, List[Dict]]:
    groups: Dict[int, List[Dict]] = defaultdict(list)
    for s in seeds:
        groups[s["meta"]["as_id"]].append(s)
    return groups


# ============================================================================
# AS-level features
# ============================================================================

def common_prefix_bits(seed_prefixes: List[str]) -> Tuple[List[int], int]:
    """
    Compute the longest common bit-prefix across all seeds in this AS.
    Returns (bits_list, length_in_bits).
    """
    if not seed_prefixes:
        return [], 0
    all_bits = []
    min_len = 128
    for p in seed_prefixes:
        bits, plen = ipv6_prefix_to_bits(p)
        all_bits.append(bits)
        min_len = min(min_len, plen)
    common: List[int] = []
    for i in range(min_len):
        v = all_bits[0][i]
        if all(b[i] == v for b in all_bits):
            common.append(v)
        else:
            break
    return common, len(common)


def median_prefix_len(seed_prefixes: List[str]) -> int:
    lens = []
    for p in seed_prefixes:
        try:
            lens.append(ipaddress.IPv6Network(p, strict=False).prefixlen)
        except Exception:
            pass
    if not lens:
        return 48
    lens.sort()
    return lens[len(lens) // 2]


def estimate_as_entropy(seed_prefixes: List[str]) -> float:
    """
    Cheap proxy for an AS's diversity:
      mean Hamming distance between random pairs of seed bit vectors,
      normalized to [0, 1]. Larger = more diverse AS.
    """
    if len(seed_prefixes) < 2:
        return 0.0
    all_bits = []
    for p in seed_prefixes:
        bits, plen = ipv6_prefix_to_bits(p)
        # Pad to 64 bits for fair comparison (use only the routing portion).
        bits = bits + [0] * (64 - len(bits))
        all_bits.append(bits[:64])

    n_pairs = min(20, len(all_bits) * (len(all_bits) - 1) // 2)
    if n_pairs <= 0:
        return 0.0
    total = 0.0
    cnt = 0
    for _ in range(n_pairs):
        i, j = random.sample(range(len(all_bits)), 2)
        d = sum(1 for a, b in zip(all_bits[i], all_bits[j]) if a != b)
        total += d / 64.0
        cnt += 1
    return total / max(cnt, 1)


# ============================================================================
# AS plan: dynamic per-AS generation parameters
# ============================================================================

def plan_for_as(as_id: int, seeds: List[Dict],
                args) -> Dict:
    """
    Decide generation parameters for one AS group based on its size and
    feature signal.
    """
    n_seeds = len(seeds)
    seed_prefs = [s["prefix"] for s in seeds]

    # (1) Number of candidates: log-scale w.r.t. group size, clipped.
    base = args.min_samples_per_as
    extra = int(round(args.samples_log_factor * math.log2(max(n_seeds, 1) + 1)))
    num_samples = min(args.max_samples_per_as,
                      max(args.min_samples_per_as, base + extra))

    # (2) Target prefix length: median of group, clipped to [args.min_target_len,
    #     args.max_target_len].
    med = median_prefix_len(seed_prefs)
    if args.fixed_target_len:
        target_len = args.fixed_target_len
    else:
        target_len = max(args.min_target_len,
                         min(args.max_target_len, med))

    # (3) Common ancestor bits as BGP-style hard constraint
    common_bits, common_len = common_prefix_bits(seed_prefs)
    # Don't pin too few or too many bits; clamp to [16, 32].
    constrain_bits = max(args.min_constrain_bits,
                         min(args.max_constrain_bits, common_len))
    if constrain_bits > target_len:
        constrain_bits = max(0, target_len - 4)

    # (4) Temperature: small AS or homogeneous AS -> low T (conservative);
    #     large AS or diverse AS -> higher T (explore more).
    diversity = estimate_as_entropy(seed_prefs)            # [0, 1]
    size_factor = min(1.0, math.log2(n_seeds + 1) / math.log2(50.0 + 1))
    temperature = (args.base_temperature
                   + args.temp_diversity_weight * diversity
                   + args.temp_size_weight * size_factor)
    temperature = max(0.5, min(args.max_temperature, temperature))

    # (5) Number of independent generation paths/passes (each pass varies the
    #     conditioning seed used to pick metadata sample within group).
    num_paths = min(args.max_paths_per_as,
                    max(1, int(round(math.log2(n_seeds + 1)))))

    return {
        "as_id": as_id,
        "n_seeds": n_seeds,
        "num_samples": num_samples,
        "target_len": target_len,
        "constrain_bits": constrain_bits,
        "common_len": common_len,
        "common_bits": common_bits,
        "temperature": round(temperature, 3),
        "diversity": round(diversity, 3),
        "median_seed_len": med,
        "num_paths": num_paths,
    }


# ============================================================================
# Generation per-AS
# ============================================================================

@torch.no_grad()
def generate_for_as(model: HMambaFRPGenerator,
                    plan: Dict,
                    seeds: List[Dict],
                    device: str,
                    seq_len: int,
                    per_bit_temp: Optional[torch.Tensor] = None,
                    batch_cap: int = 16) -> List[Dict]:
    """
    Run multiple independent generation passes for one AS.

    Each pass:
      - Picks one "anchor seed" from this AS to provide metadata + (optionally)
        seed-specific BGP constraint bits.
      - Runs num_samples / num_paths samples in this pass.
      - The final batch fed to the model contains ONLY samples of this AS
        (so the inner Mamba sees a coherent same-AS sequence).
    """
    as_id = plan["as_id"]
    target_len = plan["target_len"]
    constrain_bits = plan["constrain_bits"]
    common_bits = plan["common_bits"]
    num_samples = plan["num_samples"]
    num_paths = max(1, plan["num_paths"])
    temperature = plan["temperature"]

    samples_per_path = max(1, num_samples // num_paths)
    leftover = num_samples - samples_per_path * num_paths

    out: List[Dict] = []
    model.eval()

    # Use AS common bits as the BGP-style constraint by default
    cb_tensor = torch.zeros(seq_len, dtype=torch.long, device=device)
    cb_len = min(constrain_bits, len(common_bits))
    if cb_len > 0:
        cb_tensor[:cb_len] = torch.tensor(common_bits[:cb_len],
                                          dtype=torch.long, device=device)

    for path_i in range(num_paths):
        n_this = samples_per_path + (1 if path_i < leftover else 0)
        if n_this <= 0:
            continue

        # Anchor seed (metadata + RTT) — rotate through seeds of the AS
        anchor = seeds[path_i % len(seeds)]
        m = anchor["meta"]
        base_meta = {
            "as_id":     torch.tensor([m["as_id"]],     dtype=torch.long, device=device),
            "org_id":    torch.tensor([m["org_id"]],    dtype=torch.long, device=device),
            "cat1_id":   torch.tensor([m["cat1_id"]],   dtype=torch.long, device=device),
            "cat2_id":   torch.tensor([m["cat2_id"]],   dtype=torch.long, device=device),
            "proto_id":  torch.tensor([m["proto_id"]],  dtype=torch.long, device=device),
            "geo_id":    torch.tensor([m["geo_id"]],    dtype=torch.long, device=device),
            "rtt_stats": torch.tensor([m["rtt_stats"]], dtype=torch.float32, device=device),
        }
        plens = torch.tensor([target_len], dtype=torch.long, device=device)
        bgp_bits = cb_tensor.unsqueeze(0)              # (1, seq_len)
        bgp_len = torch.tensor([cb_len], dtype=torch.long, device=device)

        # Generate in chunks to keep memory bounded
        remaining = n_this
        while remaining > 0:
            chunk = min(remaining, batch_cap)
            bits = model.generate_staged(
                meta=base_meta,
                target_prefix_lens=plens,
                bgp_constraint_bits=bgp_bits,
                bgp_constraint_len=bgp_len,
                temperature=temperature,
                num_samples=chunk,
                per_bit_temperature=per_bit_temp,
            )  # (chunk, seq_len)

            for i in range(chunk):
                bits_i = bits[i, :target_len].tolist()
                try:
                    pref = bits_to_ipv6_prefix(bits_i)
                except Exception:
                    continue
                out.append({
                    "as_id": as_id,
                    "anchor_seed": anchor["prefix"],
                    "candidate": pref,
                    "target_len": target_len,
                    "temperature": temperature,
                })
            remaining -= chunk
    return out


# ============================================================================
# Filtering
# ============================================================================

def filter_candidates(candidates: List[Dict],
                      seed_prefixes: List[str]) -> List[Dict]:
    """
    Remove duplicates, exact-match seeds, and any candidate that is a strict
    subnet of any seed.
    """
    # Pre-parse seed networks for subnet checking
    seed_nets = []
    seed_set = set(seed_prefixes)
    for s in seed_prefixes:
        try:
            seed_nets.append(ipaddress.IPv6Network(s, strict=False))
        except Exception:
            pass

    seen = set()
    kept = []
    drop_dup = drop_eq = drop_sub = 0
    for c in candidates:
        cand = c["candidate"]
        if cand in seen:
            drop_dup += 1
            continue
        seen.add(cand)
        if cand in seed_set:
            drop_eq += 1
            continue
        try:
            cn = ipaddress.IPv6Network(cand, strict=False)
        except Exception:
            continue
        if any(cn.subnet_of(sn) for sn in seed_nets):
            drop_sub += 1
            continue
        kept.append(c)
    logger.info(f"Filter: dropped dup={drop_dup}, exact-seed={drop_eq}, "
                f"subnet-of-seed={drop_sub}; kept={len(kept)}")
    return kept


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser("HMamba per-AS adaptive candidate generator")
    p.add_argument("--ckpt", required=True, help="trained HMamba ckpt (best.pt)")
    p.add_argument("--seeds", required=True, help="seeds_prepared.txt")
    p.add_argument("--meta", required=True, help="meta.json")
    p.add_argument("--vocabs", default=None, help="vocabs.json (optional)")
    p.add_argument("--entropy", default=None, help="bit_entropy_weights.json")
    p.add_argument("--output", default="output/hmamba/candidates.txt")
    p.add_argument("--report", default=None,
                   help="optional JSON report file with per-AS plan & counts")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)

    # Per-AS adaptive knobs
    p.add_argument("--min-samples-per-as", type=int, default=2,
                   help="min candidates to draw per AS")
    p.add_argument("--max-samples-per-as", type=int, default=200,
                   help="cap on candidates per AS")
    p.add_argument("--samples-log-factor", type=float, default=15.0,
                   help="extra samples = round(factor * log2(n+1))")
    p.add_argument("--max-paths-per-as", type=int, default=5,
                   help="max independent generation passes per AS")
    p.add_argument("--total-budget", type=int, default=0,
                   help="if > 0, scale per-AS num_samples to fit this budget")
    p.add_argument("--max-as", type=int, default=0,
                   help="if > 0, only process top-N largest AS groups")

    # Length controls
    p.add_argument("--min-target-len", type=int, default=32)
    p.add_argument("--max-target-len", type=int, default=64)
    p.add_argument("--fixed-target-len", type=int, default=0,
                   help="if > 0, override per-AS adaptive target length")
    p.add_argument("--min-constrain-bits", type=int, default=16)
    p.add_argument("--max-constrain-bits", type=int, default=32)

    # Temperature controls
    p.add_argument("--base-temperature", type=float, default=0.8)
    p.add_argument("--temp-diversity-weight", type=float, default=0.4)
    p.add_argument("--temp-size-weight", type=float, default=0.3)
    p.add_argument("--max-temperature", type=float, default=1.5)

    # Inference
    p.add_argument("--batch-cap", type=int, default=16,
                   help="max samples per single forward call (memory)")
    return p.parse_args()


def resolve_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; using CPU.")
        return "cpu"
    return arg


def apply_total_budget(plans: List[Dict], total_budget: int,
                       min_per_as: int) -> List[Dict]:
    """Re-scale num_samples across all AS plans so total ~= total_budget."""
    if total_budget <= 0 or not plans:
        return plans
    cur_total = sum(p["num_samples"] for p in plans)
    if cur_total == 0:
        return plans
    scale = total_budget / cur_total
    for p in plans:
        p["num_samples"] = max(min_per_as,
                               int(round(p["num_samples"] * scale)))
    return plans


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    logger.info(f"Device: {device}")

    # ---- Load ckpt ----
    ckpt = torch.load(args.ckpt, map_location=device)
    saved_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    cfg = Config()
    cfg.device = device
    if saved_cfg:
        for k in ["d_model", "num_layers", "seq_len", "vocab_size",
                  "coarse_end", "medium_end",
                  "num_as", "num_org", "num_cat1", "num_cat2",
                  "num_proto", "num_geo",
                  "as_embed_dim", "org_embed_dim", "cat1_embed_dim",
                  "cat2_embed_dim", "proto_embed_dim", "geo_embed_dim",
                  "rtt_feat_dim"]:
            if k in saved_cfg:
                setattr(cfg.model, k, saved_cfg[k])
    elif args.vocabs and os.path.exists(args.vocabs):
        with open(args.vocabs) as f:
            v = json.load(f)
        def _pad(n): return max(16, int(n * 1.25) + 16)
        cfg.model.num_as   = _pad(len(v.get("as_vocab", {})) + 1)
        cfg.model.num_org  = _pad(len(v.get("org_vocab", {})) + 1)
        cfg.model.num_cat1 = _pad(len(v.get("cat1_vocab", {})) + 1)
        cfg.model.num_cat2 = _pad(len(v.get("cat2_vocab", {})) + 1)
        cfg.model.num_geo  = _pad(len(v.get("country_vocab", {})) + 1)

    model = HMambaFRPGenerator(cfg.model).to(device)
    state = (ckpt["model_state_dict"]
             if isinstance(ckpt, dict) and "model_state_dict" in ckpt
             else ckpt)
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded HMamba ckpt: {args.ckpt} ({n_params/1e6:.2f} M params)")

    # ---- Per-bit temperature from entropy weights ----
    pbt = None
    if args.entropy and os.path.exists(args.entropy):
        ent = load_entropy_weights(args.entropy, cfg.model.seq_len)
        if ent is not None:
            # Map normalized entropy (mean=1) to a per-bit T multiplier in
            # [0.3, 1.5]. Low entropy bit -> tight (low T), high entropy bit
            # -> spread (high T).
            pbt = (0.3 + ent.clamp(0, 1.5) * 0.8).to(device)
            logger.info("Using entropy-derived per-bit temperature.")

    # ---- Seeds & AS grouping ----
    seeds = load_seeds_with_meta(args.seeds, args.meta)
    if not seeds:
        logger.error("No seeds with metadata found.")
        return
    groups = group_by_as(seeds)
    # Sort AS groups by size (descending). Optionally truncate to top-N.
    sorted_as = sorted(groups.items(), key=lambda x: -len(x[1]))
    if args.max_as > 0:
        sorted_as = sorted_as[:args.max_as]
        logger.info(f"Limiting to top-{args.max_as} largest AS groups.")
    logger.info(f"Processing {len(sorted_as)} AS groups; "
                f"top-5 sizes: "
                f"{[(a, len(s)) for a, s in sorted_as[:5]]}")

    # ---- Build per-AS plans ----
    plans: List[Dict] = []
    for as_id, grp in sorted_as:
        plans.append(plan_for_as(as_id, grp, args))
    plans = apply_total_budget(plans, args.total_budget, args.min_samples_per_as)

    total_target = sum(p["num_samples"] for p in plans)
    logger.info(f"Total target samples across all AS: {total_target}")

    # ---- Run generation per AS ----
    all_candidates: List[Dict] = []
    for i, plan in enumerate(plans):
        as_id = plan["as_id"]
        grp = groups[as_id]
        cands = generate_for_as(
            model, plan, grp,
            device=device,
            seq_len=cfg.model.seq_len,
            per_bit_temp=pbt,
            batch_cap=args.batch_cap,
        )
        all_candidates.extend(cands)
        if (i + 1) % 20 == 0 or (i + 1) == len(plans):
            logger.info(f"  AS {i+1}/{len(plans)} done "
                        f"(as_id={as_id}, n_seeds={plan['n_seeds']}, "
                        f"target_len={plan['target_len']}, "
                        f"T={plan['temperature']}, "
                        f"got={len(cands)}; cumulative={len(all_candidates)})")

    logger.info(f"Generated {len(all_candidates)} raw candidates")

    # ---- Filter ----
    kept = filter_candidates(all_candidates, [s["prefix"] for s in seeds])

    # ---- Write output ----
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        for c in kept:
            f.write(f"{c['candidate']}\t# as={c['as_id']} "
                    f"len={c['target_len']} T={c['temperature']} "
                    f"anchor={c['anchor_seed']}\n")
    logger.info(f"Wrote {len(kept)} candidates -> {args.output}")

    # ---- Optional report ----
    if args.report:
        report = {
            "total_seeds": len(seeds),
            "as_groups_processed": len(plans),
            "raw_candidates": len(all_candidates),
            "final_candidates": len(kept),
            "per_as_plan": [
                {**{k: p[k] for k in (
                    "as_id", "n_seeds", "num_samples", "target_len",
                    "constrain_bits", "common_len", "temperature",
                    "diversity", "median_seed_len", "num_paths")}}
                for p in plans
            ],
        }
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Wrote per-AS report -> {args.report}")


if __name__ == "__main__":
    main()
