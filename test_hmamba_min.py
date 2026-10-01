#!/usr/bin/env python3
"""
Minimal runnable testing/inference script for HMamba FRP generator.

Two modes:
  1) seed-expand: for each seed prefix, run staged generation conditioned
     on that seed's metadata + BGP-prefix constraint of the seed's first L bits.
  2) sanity:      run a forward + a tiny generation batch to validate model.

Usage:

  # Generate candidates around given seed prefixes
  python test_hmamba_min.py \
      --ckpt   output/hmamba_min/hmamba_generator.pt \
      --meta   output/prepared_new/meta.json \
      --vocabs output/prepared_new/vocabs.json \
      --entropy output/prepared_new/bit_entropy_weights.json \
      --seeds  output/prepared_new/seeds_prepared.txt \
      --max-seeds 50 \
      --num-samples 4 \
      --target-len 64 \
      --output output/hmamba_min/candidates.txt
"""
import argparse
import ipaddress
import json
import logging
import os
import random
from typing import Dict, List

import torch

try:
    from .config import Config
    from .generators.hmamba import HMambaFRPGenerator, MASK_ID, load_entropy_weights
    from .utils.ipv6_utils import (
        ipv6_prefix_to_bits, bits_to_ipv6_prefix, normalize_prefix,
        is_subnet_of,
    )
except ImportError:
    import sys
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


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def build_meta_tensors(metas: List[Dict], device: str) -> Dict[str, torch.Tensor]:
    return {
        "as_id":     torch.tensor([m["as_id"]     for m in metas], dtype=torch.long, device=device),
        "org_id":    torch.tensor([m["org_id"]    for m in metas], dtype=torch.long, device=device),
        "cat1_id":   torch.tensor([m["cat1_id"]   for m in metas], dtype=torch.long, device=device),
        "cat2_id":   torch.tensor([m["cat2_id"]   for m in metas], dtype=torch.long, device=device),
        "proto_id":  torch.tensor([m["proto_id"]  for m in metas], dtype=torch.long, device=device),
        "geo_id":    torch.tensor([m["geo_id"]    for m in metas], dtype=torch.long, device=device),
        "rtt_stats": torch.tensor([m["rtt_stats"] for m in metas], dtype=torch.float32, device=device),
    }


def load_seeds(seeds_file: str, meta_file: str,
               max_seeds: int = 0) -> List[Dict]:
    with open(meta_file) as f:
        meta_map = json.load(f)
    items = []
    with open(seeds_file) as f:
        for line in f:
            line = line.strip()
            if not line:
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
            if max_seeds and len(items) >= max_seeds:
                break
    logger.info(f"Loaded {len(items)} seeds with metadata")
    return items


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

@torch.no_grad()
def generate_for_seeds(model: HMambaFRPGenerator,
                       seeds: List[Dict],
                       device: str,
                       num_samples: int = 4,
                       target_len: int = 64,
                       constrain_bits: int = 16,
                       temperature: float = 1.0,
                       per_bit_temp: torch.Tensor = None,
                       batch_size: int = 8) -> List[Dict]:
    """
    For each seed:
      - Take its first `constrain_bits` bits as a hard BGP-style constraint.
      - Sample `num_samples` completions up to `target_len` bits.
    """
    results = []
    model.eval()
    seq_len = model.cfg.seq_len

    for start in range(0, len(seeds), batch_size):
        chunk = seeds[start:start + batch_size]
        metas = [s["meta"] for s in chunk]
        meta_t = build_meta_tensors(metas, device)

        # BGP constraint = first `constrain_bits` of each seed
        bgp_bits = torch.zeros(len(chunk), seq_len, dtype=torch.long, device=device)
        bgp_lens = torch.zeros(len(chunk), dtype=torch.long, device=device)
        plens = torch.full((len(chunk),), target_len, dtype=torch.long, device=device)

        for i, s in enumerate(chunk):
            bits, plen = ipv6_prefix_to_bits(s["prefix"])
            cb = min(constrain_bits, plen)
            bgp_bits[i, :cb] = torch.tensor(bits[:cb], dtype=torch.long, device=device)
            bgp_lens[i] = cb

        out_bits = model.generate_staged(
            meta=meta_t,
            target_prefix_lens=plens,
            bgp_constraint_bits=bgp_bits,
            bgp_constraint_len=bgp_lens,
            temperature=temperature,
            num_samples=num_samples,
            per_bit_temperature=per_bit_temp,
        )  # (B*num_samples, 128)

        for i, s in enumerate(chunk):
            for j in range(num_samples):
                idx = i * num_samples + j
                bits = out_bits[idx, :target_len].tolist()
                try:
                    pref = bits_to_ipv6_prefix(bits)
                except Exception:
                    continue
                results.append({
                    "seed": s["prefix"],
                    "candidate": pref,
                    "as_id": s["meta"]["as_id"],
                })
    return results


# ----------------------------------------------------------------------------
# Filtering: drop candidates that are subnets of (or equal to) any existing seed
# ----------------------------------------------------------------------------

def filter_against_seeds(candidates: List[Dict],
                         seed_prefixes: List[str]) -> List[Dict]:
    seen = set()
    kept = []
    for c in candidates:
        cand = c["candidate"]
        if cand in seen:
            continue
        seen.add(cand)
        # Drop if cand is identical to or strictly inside any seed
        skip = False
        for s in seed_prefixes:
            if cand == s:
                skip = True
                break
            try:
                if is_subnet_of(cand, s):
                    skip = True
                    break
            except Exception:
                pass
        if not skip:
            kept.append(c)
    return kept


# ----------------------------------------------------------------------------
# Sanity check
# ----------------------------------------------------------------------------

@torch.no_grad()
def sanity_check(model: HMambaFRPGenerator, device: str):
    """Run a tiny forward and a 1-sample generation to validate model."""
    model.eval()
    B = 4
    bit_seq = torch.full((B, 128), MASK_ID, dtype=torch.long, device=device)
    meta = {
        "as_id":     torch.tensor([1, 1, 2, 2], dtype=torch.long, device=device),
        "org_id":    torch.zeros(B, dtype=torch.long, device=device),
        "cat1_id":   torch.zeros(B, dtype=torch.long, device=device),
        "cat2_id":   torch.zeros(B, dtype=torch.long, device=device),
        "proto_id":  torch.zeros(B, dtype=torch.long, device=device),
        "geo_id":    torch.zeros(B, dtype=torch.long, device=device),
        "rtt_stats": torch.zeros(B, 3, dtype=torch.float32, device=device),
    }
    logits = model(bit_seq, meta)
    logger.info(f"Sanity forward OK: logits shape = {tuple(logits.shape)}")

    plens = torch.full((B,), 32, dtype=torch.long, device=device)
    bits = model.generate_staged(meta=meta, target_prefix_lens=plens,
                                 num_samples=1, temperature=1.0)
    logger.info(f"Sanity generate OK: bits shape = {tuple(bits.shape)}")
    for i in range(B):
        try:
            p = bits_to_ipv6_prefix(bits[i, :32].tolist())
            logger.info(f"  sample {i}: {p}")
        except Exception as e:
            logger.info(f"  sample {i}: <decode error {e}>")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser("HMamba minimal tester / candidate generator")
    p.add_argument("--ckpt", required=True, help="trained hmamba_generator.pt")
    p.add_argument("--meta", default=None, help="meta.json")
    p.add_argument("--seeds", default=None, help="seeds_prepared.txt")
    p.add_argument("--vocabs", default=None, help="vocabs.json")
    p.add_argument("--entropy", default=None, help="bit_entropy_weights.json")
    p.add_argument("--output", default="output/hmamba_min/candidates.txt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-seeds", type=int, default=50,
                   help="limit number of seeds (0 = all)")
    p.add_argument("--num-samples", type=int, default=4,
                   help="candidates per seed")
    p.add_argument("--target-len", type=int, default=64,
                   help="generated prefix length")
    p.add_argument("--constrain-bits", type=int, default=16,
                   help="how many leading bits of seed are kept fixed")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sanity", action="store_true",
                   help="only run a sanity forward+generate, no seeds needed")
    args = p.parse_args()

    cfg = Config()
    cfg.device = args.device

    # Apply vocab sizes from saved ckpt or vocabs.json
    ckpt = torch.load(args.ckpt, map_location=args.device)
    saved_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    if saved_cfg:
        cfg.model.d_model    = saved_cfg.get("d_model",    cfg.model.d_model)
        cfg.model.num_layers = saved_cfg.get("num_layers", cfg.model.num_layers)
        cfg.model.num_as     = saved_cfg.get("num_as",     cfg.model.num_as)
        cfg.model.num_org    = saved_cfg.get("num_org",    cfg.model.num_org)
        cfg.model.num_cat1   = saved_cfg.get("num_cat1",   cfg.model.num_cat1)
        cfg.model.num_cat2   = saved_cfg.get("num_cat2",   cfg.model.num_cat2)
        cfg.model.num_geo    = saved_cfg.get("num_geo",    cfg.model.num_geo)
    elif args.vocabs and os.path.exists(args.vocabs):
        with open(args.vocabs) as f:
            v = json.load(f)
        def _pad(n): return max(16, int(n * 1.25) + 16)
        cfg.model.num_as   = _pad(len(v.get("as_vocab", {})) + 1)
        cfg.model.num_org  = _pad(len(v.get("org_vocab", {})) + 1)
        cfg.model.num_cat1 = _pad(len(v.get("cat1_vocab", {})) + 1)
        cfg.model.num_cat2 = _pad(len(v.get("cat2_vocab", {})) + 1)
        cfg.model.num_geo  = _pad(len(v.get("country_vocab", {})) + 1)

    model = HMambaFRPGenerator(cfg.model).to(args.device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and \
        "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded HMamba ckpt: {args.ckpt} ({n_params/1e6:.2f} M params)")

    # Per-bit temperature from entropy weights (optional)
    pbt = None
    if args.entropy and os.path.exists(args.entropy):
        ent = load_entropy_weights(args.entropy, cfg.model.seq_len)
        if ent is not None:
            # Map entropy [0, 1+] -> temperature [0.1, 1.5]
            pbt = (0.1 + ent.clamp(0, 1.5) * 0.9).to(args.device)
            logger.info("Using entropy-derived per-bit temperature")

    if args.sanity:
        sanity_check(model, args.device)
        return

    if not (args.seeds and args.meta):
        logger.warning("--seeds and --meta required (or use --sanity)")
        return

    seeds = load_seeds(args.seeds, args.meta, args.max_seeds)
    if not seeds:
        logger.error("No seeds with metadata found")
        return

    candidates = generate_for_seeds(
        model, seeds,
        device=args.device,
        num_samples=args.num_samples,
        target_len=args.target_len,
        constrain_bits=args.constrain_bits,
        temperature=args.temperature,
        per_bit_temp=pbt,
        batch_size=args.batch_size,
    )
    logger.info(f"Generated {len(candidates)} raw candidates")

    seed_set = [s["prefix"] for s in seeds]
    kept = filter_against_seeds(candidates, seed_set)
    logger.info(f"After filtering against seeds: {len(kept)} candidates "
                f"(dropped {len(candidates) - len(kept)})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for c in kept:
            f.write(f"{c['candidate']}\t<- seed={c['seed']} as_id={c['as_id']}\n")
    logger.info(f"Wrote {len(kept)} candidates to {args.output}")


if __name__ == "__main__":
    main()
