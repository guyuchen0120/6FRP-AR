#!/usr/bin/env python3
"""
6FRP-AR v3 online inference — single-server, local probing.

Two modes:
    --mode seed_expand  : expand from known seed FRP cohort(s)
    --mode target       : search for FRPs inside given BGP prefix(es)

Probing is done locally via LocalAPDProber (requires root + IPv6).
For offline testing, use --simulate with a known-frps file.

Output is structured into separate files under --output-dir:
    candidates.json  — generated candidates with verifier scores
    probed.json      — probe results with classification details
    final_frps.txt   — confirmed FRP list (one per line)
    summary.json     — run statistics
"""
import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import Config
from .generators.unified import UnifiedFRPGenerator
from .verifier.verifier import EnsembleVerifier
from .alignment.tree_align import GlobalFRPFilter
from .online.feedback import OnlineFeedback
from .modes.seed_expand import run_seed_expand
from .modes.target import run_target, run_target_batch
from .utils.apd import APDProber, LocalAPDProber, SimulatedAPDProber
from .utils.ipv6_utils import normalize_prefix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Model loading
# ============================================================================

def load_models(cfg: Config, checkpoint_dir: str):
    """Load generator and verifier from checkpoint directory."""
    gen_path = os.path.join(checkpoint_dir, "generator.pt")
    ver_path = os.path.join(checkpoint_dir, "verifier.pt")

    generator = UnifiedFRPGenerator(cfg.model).to(cfg.device)
    verifier = EnsembleVerifier(
        ensemble_size=cfg.model.verifier_ensemble_size,
        max_seq_len=cfg.model.verifier_max_seq_len,
        bit_embed_dim=32,
        hidden_dim=cfg.model.verifier_d_model,
        num_layers=cfg.model.verifier_num_layers,
        num_heads=cfg.model.verifier_num_heads,
        condition_dim=cfg.model.d_model,
        dropout=0.1,
    ).to(cfg.device)

    if os.path.exists(gen_path):
        ckpt = torch.load(gen_path, map_location=cfg.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            generator.load_state_dict(ckpt["model_state_dict"])
        else:
            generator.load_state_dict(ckpt)
        logger.info(f"Loaded generator from {gen_path}")
    else:
        logger.warning(f"Generator checkpoint not found: {gen_path}")

    if os.path.exists(ver_path):
        verifier.load_state_dict(torch.load(ver_path, map_location=cfg.device))
        logger.info(f"Loaded verifier from {ver_path}")
    else:
        logger.warning(f"Verifier checkpoint not found: {ver_path}")

    generator.eval()
    verifier.eval()
    return generator, verifier


# ============================================================================
# Prober factory
# ============================================================================

def make_prober(cfg: Config, simulate_file: Optional[str] = None) -> APDProber:
    """
    Create the appropriate prober:
      - If --simulate FILE is given: SimulatedAPDProber
      - Otherwise: LocalAPDProber (requires root + IPv6)
    """
    if simulate_file and os.path.exists(simulate_file):
        with open(simulate_file) as f:
            known = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        logger.info(f"Using SimulatedAPDProber with {len(known)} known FRPs")
        return SimulatedAPDProber(
            known_frps=known,
            probe_count=cfg.apd.probe_count,
            timeout=cfg.apd.timeout,
            success_rate_threshold=cfg.apd.success_rate_threshold,
        )

    # Local probing — check root first
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        logger.error("LocalAPDProber requires root. Use sudo or --simulate for testing.")
        sys.exit(1)

    logger.info(f"Using LocalAPDProber (concurrency={cfg.apd.concurrency})")
    return LocalAPDProber(
        probe_count=cfg.apd.probe_count,
        timeout=cfg.apd.timeout,
        success_rate_threshold=cfg.apd.success_rate_threshold,
        concurrency=cfg.apd.concurrency,
        inter_probe_delay=cfg.apd.inter_probe_delay,
    )


# ============================================================================
# Metadata parsing
# ============================================================================

def parse_meta(meta_str: Optional[str]) -> Dict:
    """Parse metadata from CLI: JSON string or key=val,key=val."""
    if not meta_str:
        return {}
    meta_str = meta_str.strip()
    if meta_str.startswith('{'):
        return json.loads(meta_str)
    result = {}
    for pair in meta_str.split(','):
        if '=' in pair:
            k, v = pair.split('=', 1)
            try:
                result[k.strip()] = int(v.strip())
            except ValueError:
                try:
                    result[k.strip()] = float(v.strip())
                except ValueError:
                    result[k.strip()] = v.strip()
    return result


# ============================================================================
# Output helpers
# ============================================================================

def save_results(output_dir: str, candidates: List, probed: List,
                 frps: List[str], summary: Dict):
    """Save structured results to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Candidates
    cand_path = os.path.join(output_dir, "candidates.json")
    with open(cand_path, "w") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)

    # Probed results
    probed_path = os.path.join(output_dir, "probed.json")
    with open(probed_path, "w") as f:
        json.dump(probed, f, indent=2, ensure_ascii=False)

    # Final FRPs
    frp_path = os.path.join(output_dir, "final_frps.txt")
    with open(frp_path, "w") as f:
        for p in frps:
            f.write(f"{p}\n")

    # Summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to {output_dir}/")
    logger.info(f"  candidates.json: {len(candidates)} entries")
    logger.info(f"  probed.json: {len(probed)} entries")
    logger.info(f"  final_frps.txt: {len(frps)} FRPs")
    logger.info(f"  summary.json: run statistics")


# ============================================================================
# Mode A: Seed Expand
# ============================================================================

def cmd_seed_expand(args, cfg, generator, verifier, prober):
    """Execute Mode A: Seed Expansion."""
    with open(args.seeds) as f:
        seeds = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    meta = parse_meta(args.meta)
    logger.info(f"Mode A: Seed Expand with {len(seeds)} seeds")

    feedback = OnlineFeedback(generator, verifier, cfg, device=cfg.device)
    global_filter = GlobalFRPFilter()
    for s in seeds:
        global_filter.add(normalize_prefix(s))

    start_time = time.time()
    result = run_seed_expand(
        seed_prefixes=seeds,
        meta=meta,
        generator=generator,
        verifier=verifier,
        prober=prober,
        cfg=cfg,
        feedback=feedback,
        global_filter=global_filter,
    )
    elapsed = time.time() - start_time

    # Build structured output
    summary = {
        "mode": "seed_expand",
        "seeds_count": len(seeds),
        "candidates_generated": result["candidates_generated"],
        "candidates_probed": result["candidates_probed"],
        "newly_found_frps": len(result["newly_found_frps"]),
        "total_frps": len(result["all_frps_after_run"]),
        "alignment_stats": result["alignment_stats"],
        "feedback_stats": feedback.get_stats(),
        "elapsed_seconds": round(elapsed, 1),
    }

    save_results(
        args.output_dir,
        candidates=[],  # TODO: capture from mode internals
        probed=[],
        frps=result["all_frps_after_run"],
        summary=summary,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("SEED EXPAND RESULTS")
    print(f"{'='*60}")
    print(f"Seeds:              {len(seeds)}")
    print(f"Candidates:         {result['candidates_generated']}")
    print(f"Probed:             {result['candidates_probed']}")
    print(f"New FRPs found:     {len(result['newly_found_frps'])}")
    print(f"Total FRPs:         {len(result['all_frps_after_run'])}")
    print(f"Time:               {elapsed:.1f}s")
    print(f"{'='*60}")


# ============================================================================
# Mode B: Target Discovery
# ============================================================================

def cmd_target(args, cfg, generator, verifier, prober):
    """Execute Mode B: Target Discovery."""
    with open(args.targets) as f:
        targets = [l.strip() for l in f if l.strip() and not l.startswith('#')]

    meta = parse_meta(args.meta)

    # Load per-BGP metadata if provided
    per_bgp_meta = None
    if args.per_bgp_meta and os.path.exists(args.per_bgp_meta):
        with open(args.per_bgp_meta) as f:
            per_bgp_meta = json.load(f)

    # Load AS seed DB if provided
    as_seed_db = None
    if args.as_seed_db and os.path.exists(args.as_seed_db):
        with open(args.as_seed_db) as f:
            raw = json.load(f)
            as_seed_db = {int(k): v for k, v in raw.items()}

    logger.info(f"Mode B: Target Discovery on {len(targets)} BGP prefixes")

    feedback = OnlineFeedback(generator, verifier, cfg, device=cfg.device)
    global_filter = GlobalFRPFilter()

    start_time = time.time()
    result = run_target_batch(
        bgp_prefixes=targets,
        default_meta=meta,
        generator=generator,
        verifier=verifier,
        prober=prober,
        cfg=cfg,
        per_bgp_meta=per_bgp_meta,
        as_seed_db=as_seed_db,
        feedback=feedback,
        global_filter=global_filter,
        target_plen=args.target_plen,
    )
    elapsed = time.time() - start_time

    # Build structured output
    per_bgp_summary = []
    for r in result["per_bgp_results"]:
        per_bgp_summary.append({
            "bgp_prefix": r["bgp_prefix"],
            "mimic_mode": r["mimic_mode"],
            "candidates_generated": r["candidates_generated"],
            "candidates_probed": r["candidates_probed"],
            "newly_found": len(r["newly_found_frps"]),
            "newly_found_frps": r["newly_found_frps"],
        })

    summary = {
        "mode": "target",
        "bgp_prefixes_count": len(targets),
        "total_frps_found": result["num_frps_found"],
        "total_frps_final": len(result["all_frps"]),
        "per_bgp_results": per_bgp_summary,
        "feedback_stats": feedback.get_stats(),
        "elapsed_seconds": round(elapsed, 1),
    }

    save_results(
        args.output_dir,
        candidates=[],
        probed=[],
        frps=result["all_frps"],
        summary=summary,
    )

    # Print summary
    print(f"\n{'='*60}")
    print("TARGET DISCOVERY RESULTS")
    print(f"{'='*60}")
    print(f"BGP prefixes:       {len(targets)}")
    print(f"Total FRPs found:   {result['num_frps_found']}")
    print(f"Final FRP set:      {len(result['all_frps'])}")
    print(f"Time:               {elapsed:.1f}s")
    print(f"\nPer-BGP breakdown:")
    for r in result["per_bgp_results"]:
        mode_str = "mimic" if r["mimic_mode"] else "innovate"
        print(f"  {r['bgp_prefix']} [{mode_str}]: "
              f"+{len(r['newly_found_frps'])} FRPs "
              f"(gen={r['candidates_generated']}, probed={r['candidates_probed']})")
    print(f"{'='*60}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="6FRP-AR v3 inference — single-server local probing")
    parser.add_argument('--mode', required=True, choices=['seed_expand', 'target'])
    parser.add_argument('--checkpoint-dir', default='output',
                        help='Directory containing generator.pt and verifier.pt')
    parser.add_argument('--output-dir', default='output/results',
                        help='Directory for structured output files')
    parser.add_argument('--budget', type=int, default=200,
                        help='Active learning budget per round')
    parser.add_argument('--candidates', type=int, default=1000,
                        help='Candidates to generate per round')
    parser.add_argument('--device', default='cpu')

    # Prober selection
    parser.add_argument('--simulate', default=None, metavar='FILE',
                        help='Use SimulatedAPDProber with this known-FRP file '
                             '(for offline testing without root/IPv6)')

    # Mode A args
    parser.add_argument('--seeds', default=None, help='[Mode A] Seed FRP file')

    # Mode B args
    parser.add_argument('--targets', default=None, help='[Mode B] BGP prefix file')
    parser.add_argument('--per-bgp-meta', default=None,
                        help='[Mode B] JSON: BGP prefix -> metadata')
    parser.add_argument('--as-seed-db', default=None,
                        help='[Mode B] JSON: AS ID -> seed FRP list')
    parser.add_argument('--target-plen', type=int, default=64,
                        help='[Mode B] Target prefix length (default: 64)')

    # Shared
    parser.add_argument('--meta', default=None,
                        help='Metadata: JSON string or key=val,key=val')
    parser.add_argument('--vocabs', default=None,
                        help='vocabs.json for embedding size auto-fit')

    args = parser.parse_args()

    cfg = Config()
    cfg.device = args.device
    cfg.train.al_budget_per_round = args.budget
    cfg.train.al_candidates_per_round = args.candidates

    # Vocab tightening
    if args.vocabs and os.path.exists(args.vocabs):
        with open(args.vocabs) as f:
            vocabs = json.load(f)
        def _pad(n):
            return max(16, int(n * 1.25) + 16)
        cfg.model.num_as = _pad(len(vocabs.get("as_vocab", {})) + 1)
        cfg.model.num_org = _pad(len(vocabs.get("org_vocab", {})) + 1)
        cfg.model.num_cat1 = _pad(len(vocabs.get("cat1_vocab", {})) + 1)
        cfg.model.num_cat2 = _pad(len(vocabs.get("cat2_vocab", {})) + 1)
        cfg.model.num_geo = _pad(len(vocabs.get("country_vocab", {})) + 1)

    # Load models
    generator, verifier = load_models(cfg, args.checkpoint_dir)

    # Build prober
    prober = make_prober(cfg, simulate_file=args.simulate)

    # Dispatch
    if args.mode == 'seed_expand':
        if not args.seeds:
            parser.error("--mode seed_expand requires --seeds")
        cmd_seed_expand(args, cfg, generator, verifier, prober)
    else:
        if not args.targets:
            parser.error("--mode target requires --targets")
        cmd_target(args, cfg, generator, verifier, prober)


if __name__ == "__main__":
    main()
