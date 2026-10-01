#!/usr/bin/env python3
"""
6FRP-AR v3 Standalone-APD Inference Pipeline.

This version uses the standalone_apd probing approach (ping-style with
multi-attempt retry + downward-recursive tree alignment) instead of the
original TreeAligner (upward + lateral).

Key differences from infer.py:
  - Probing: ping-style (5 attempts/address, 1s interval) via apd_core
  - Tree alignment: downward recursive (find sub-FRPs from partial prefixes)
  - Refinement: upward contraction to find shortest FRP prefix
  - Speed: batch generation, concurrent probing, probe caching
  - Output: detailed per-candidate status with timing

Usage:
    # Seed expansion (simulated, no root needed)
    python3 infer_standalone.py --mode seed_expand \\
        --seeds data/seeds.txt --simulate data/known_frps.txt \\
        --checkpoint-dir output --output-dir output/run_001

    # Target discovery (live probing, requires root)
    sudo python3 infer_standalone.py --mode target \\
        --targets data/bgp_prefixes.txt --as-seed-db data/as_seeds.json \\
        --checkpoint-dir output --output-dir output/run_002

    # Fast mode (reduced attempts, higher concurrency)
    sudo python3 infer_standalone.py --mode seed_expand \\
        --seeds data/seeds.txt --max-attempts 3 --concurrency 32 \\
        --fast-refine --output-dir output/run_fast
"""

import argparse
import ipaddress
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

# ---------------------------------------------------------------------------
# Path setup: ensure both the package parent and the package itself are importable
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
_pkg_name = os.path.basename(_here)  # "my_design_final"

# Add parent so that `import my_design_final.xxx` works (for relative imports)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
# Also add _here so that `import config` etc. works directly
if _here not in sys.path:
    sys.path.insert(0, _here)

# Direct imports (no relative imports in these modules)
from config import Config
from generators.unified import UnifiedFRPGenerator
from verifier.verifier import EnsembleVerifier, prepare_verifier_input
from utils.ipv6_utils import (
    ipv6_prefix_to_bits, normalize_prefix, bits_to_ipv6_prefix,
    is_subnet_of,
)
from utils.apd_core import (
    diagnose_prefix,
    determine_frp_status,
    refine_frp_prefix,
    tree_align_prefix,
    find_candidate_sub_frps,
    is_covered_by,
    has_raw_socket_capability,
)

# These modules use relative imports (from ..xxx), import via package
import importlib
_feedback_mod = importlib.import_module(f"{_pkg_name}.online.feedback")
OnlineFeedback = _feedback_mod.OnlineFeedback

_sampler_mod = importlib.import_module(f"{_pkg_name}.active_learning.sampler")
ActiveLearningSampler = _sampler_mod.ActiveLearningSampler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants & Defaults
# ============================================================================

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_TIMEOUT = 2.0
DEFAULT_CONCURRENCY = 16
DEFAULT_MIN_PREFIX_LEN = 16
DEFAULT_MAX_TA_DEPTH = 4


# ============================================================================
# Probe Cache (speed optimization)
# ============================================================================

class ProbeCache:
    """Cache probe results to avoid redundant probing."""

    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self.hits = 0
        self.misses = 0

    def get(self, prefix: str) -> Optional[Dict]:
        key = normalize_prefix(prefix)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, prefix: str, result: Dict):
        key = normalize_prefix(prefix)
        self._cache[key] = result

    def has(self, prefix: str) -> bool:
        return normalize_prefix(prefix) in self._cache

    def stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_hit_rate": round(self.hits / max(total, 1), 3),
            "cache_size": len(self._cache),
        }


# ============================================================================
# FRP Tracker (global state)
# ============================================================================

class FRPTracker:
    """Track confirmed FRPs, handle coverage dedup."""

    def __init__(self):
        self.confirmed: List[Dict] = []  # {prefix, source, original, ...}
        self.total_probes_sent = 0
        self.detection_probes = 0
        self.refinement_probes = 0
        self.ta_probes = 0
        self.ta_new_frps = 0

    def add_frp(self, prefix: str, source: str, original: str = None,
                extra: Dict = None) -> bool:
        """Add a confirmed FRP. Returns True if actually new."""
        prefix = normalize_prefix(prefix)
        prefix_net = ipaddress.IPv6Network(prefix, strict=False)

        # Check if already covered
        for f in self.confirmed:
            f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
            if is_covered_by(prefix_net, f_net):
                return False

        # Remove existing FRPs covered by new broader one
        kept = []
        for f in self.confirmed:
            f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
            if not is_covered_by(f_net, prefix_net):
                kept.append(f)
        self.confirmed = kept

        entry = {
            "prefix": prefix,
            "source": source,
            "original": original or prefix,
            "timestamp": time.time(),
        }
        if extra:
            entry.update(extra)
        self.confirmed.append(entry)
        return True

    def is_covered(self, prefix: str) -> bool:
        try:
            p_net = ipaddress.IPv6Network(prefix, strict=False)
        except Exception:
            return False
        for f in self.confirmed:
            f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
            if is_covered_by(p_net, f_net):
                return True
        return False

    def get_prefixes(self) -> List[str]:
        return [f["prefix"] for f in self.confirmed]

    def count(self) -> int:
        return len(self.confirmed)


# ============================================================================
# Detailed Output Formatter
# ============================================================================

class OutputFormatter:
    """Real-time formatted output for the pipeline."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._idx = 0

    def header(self, mode: str, config: Dict):
        print(f"\n{'═' * 78}")
        print(f"  6FRP-AR Standalone-APD Inference")
        print(f"  Mode: {mode.upper().replace('_', ' ')}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─' * 78}")
        for k, v in config.items():
            print(f"  {k:20s}: {v}")
        print(f"{'═' * 78}")
        print()
        print(f"{'#':>4} {'Candidate':<44} {'Status':<12} {'Detail'}")
        print(f"{'─'*4} {'─'*44} {'─'*12} {'─'*40}")
        sys.stdout.flush()

    def candidate_start(self, prefix: str, verifier_prob: float = None,
                        verifier_std: float = None):
        self._idx += 1
        prob_str = f"p={verifier_prob:.3f}" if verifier_prob is not None else ""
        std_str = f"±{verifier_std:.3f}" if verifier_std is not None else ""
        score_str = f" [{prob_str}{std_str}]" if prob_str else ""
        print(f"{self._idx:>4} {prefix:<44} {'PROBING':<12}{score_str}",
              end='', flush=True)

    def candidate_result(self, prefix: str, is_frp: bool, reach_count: int,
                         refined_prefix: str = None, time_ms: float = None):
        time_str = f" ({time_ms:.0f}ms)" if time_ms else ""
        if is_frp:
            refined_str = f" → {refined_prefix}" if refined_prefix and refined_prefix != prefix else ""
            print(f"\r{self._idx:>4} {prefix:<44} {'✓ FRP':<12} "
                  f"{reach_count}/16{refined_str}{time_str}")
        else:
            print(f"\r{self._idx:>4} {prefix:<44} {'✗ NOT FRP':<12} "
                  f"{reach_count}/16{time_str}")
        sys.stdout.flush()

    def ta_found(self, parent_prefix: str, new_frp: str, depth: int):
        print(f"{'':>4}  ↳ TA-L{depth}: {new_frp:<40} ✓ FRP (from {parent_prefix})")
        sys.stdout.flush()

    def refine_step(self, prefix: str, new_len: int, is_frp: bool):
        if self.verbose:
            mark = "✓" if is_frp else "✗"
            print(f"{'':>4}    refine /{new_len} {mark}")
            sys.stdout.flush()

    def feedback_update(self, update_type: str, detail: str = ""):
        if self.verbose:
            print(f"{'':>4}  ⟳ {update_type}: {detail}")
            sys.stdout.flush()

    def stage_summary(self, stage: str, stats: Dict):
        print(f"\n{'─' * 78}")
        print(f"  Stage: {stage}")
        for k, v in stats.items():
            print(f"    {k}: {v}")
        print(f"{'─' * 78}")
        sys.stdout.flush()

    def final_summary(self, stats: Dict):
        print(f"\n{'═' * 78}")
        print(f"  FINAL RESULTS")
        print(f"{'─' * 78}")
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k:<30}: {v}")
        print(f"{'═' * 78}\n")
        sys.stdout.flush()


# ============================================================================
# Core: Standalone APD Probe + Align for a single candidate
# ============================================================================

def probe_and_align_candidate(
    prefix: str,
    tracker: FRPTracker,
    cache: ProbeCache,
    formatter: OutputFormatter,
    timeout: float = DEFAULT_TIMEOUT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    concurrency: int = DEFAULT_CONCURRENCY,
    min_prefix_len: int = DEFAULT_MIN_PREFIX_LEN,
    do_refine: bool = True,
    do_ta: bool = True,
    max_ta_depth: int = DEFAULT_MAX_TA_DEPTH,
    fast_refine: bool = False,
    verifier_prob: float = None,
    verifier_std: float = None,
) -> Dict:
    """
    Probe a single candidate prefix using standalone_apd logic:
      1. Ping-style diagnose (16 sub-prefixes × max_attempts)
      2. If FRP → refine (find shortest FRP)
      3. If partial → tree_align (find sub-FRPs recursively)

    Returns detailed result dict.
    """
    prefix = normalize_prefix(prefix)
    result = {
        "prefix": prefix,
        "is_frp": False,
        "reach_count": 0,
        "refined_prefix": None,
        "ta_frps": [],
        "probes_used": 0,
        "time_ms": 0,
        "verifier_prob": verifier_prob,
        "verifier_std": verifier_std,
    }

    # Skip if already covered
    if tracker.is_covered(prefix):
        result["skipped"] = "covered"
        return result

    start = time.time()
    formatter.candidate_start(prefix, verifier_prob, verifier_std)

    # --- Step 1: Diagnose ---
    cached = cache.get(prefix)
    if cached:
        diag = cached
    else:
        diag = diagnose_prefix(
            prefix,
            timeout=timeout,
            max_attempts=max_attempts,
            concurrency=concurrency,
        )
        cache.put(prefix, diag)

    tracker.detection_probes += diag["total_probes_sent"]
    tracker.total_probes_sent += diag["total_probes_sent"]
    result["reach_count"] = diag["reachable_count"]
    result["probes_used"] += diag["total_probes_sent"]

    # --- Step 2: Determine FRP status ---
    is_frp, reason = determine_frp_status(diag)
    result["is_frp"] = is_frp

    if is_frp:
        # --- Step 3a: Refine (find shortest FRP) ---
        refined_prefix = prefix
        if do_refine:
            ref = refine_frp_prefix(
                prefix,
                timeout=timeout,
                max_attempts=max_attempts,
                min_prefix_len=min_prefix_len,
                concurrency=concurrency,
                fast_refine=fast_refine,
            )
            refined_prefix = ref["refined"]
            tracker.refinement_probes += ref["total_refinement_probes"]
            tracker.total_probes_sent += ref["total_refinement_probes"]
            result["probes_used"] += ref["total_refinement_probes"]
            result["refined_prefix"] = refined_prefix
            result["refinement_detail"] = {
                "original_len": ref["original_len"],
                "refined_len": ref["refined_len"],
                "reduction_bits": ref["reduction_bits"],
                "steps": len(ref["steps"]),
            }

            # Each intermediate FRP found during refinement also counts
            for step in ref["steps"]:
                if step["is_frp"]:
                    tracker.add_frp(
                        step["prefix"], source="refine_intermediate",
                        original=prefix,
                    )

        # Add the final refined FRP
        tracker.add_frp(refined_prefix, source="direct", original=prefix)

        elapsed_ms = (time.time() - start) * 1000
        result["time_ms"] = elapsed_ms
        formatter.candidate_result(prefix, True, diag["reachable_count"],
                                   refined_prefix, elapsed_ms)

    else:
        # --- Step 3b: Tree alignment for partial reachability ---
        if do_ta and diag["reachable_count"] > 0:
            known_frp_set = set(tracker.get_prefixes())
            ta_result = tree_align_prefix(
                prefix, diag, known_frp_set,
                timeout=timeout,
                max_attempts=max_attempts,
                min_prefix_len=min_prefix_len,
                concurrency=concurrency,
                refine=do_refine,
                max_depth=max_ta_depth,
                fast_refine=fast_refine,
            )
            tracker.ta_probes += ta_result["total_probes"]
            tracker.total_probes_sent += ta_result["total_probes"]
            result["probes_used"] += ta_result["total_probes"]

            for new_frp in ta_result["new_frps"]:
                new_p = new_frp["prefix"]
                if tracker.add_frp(new_p, source="tree_align", original=prefix):
                    tracker.ta_new_frps += 1
                    result["ta_frps"].append(new_p)
                    formatter.ta_found(prefix, new_p, new_frp.get("depth", 0))

        elapsed_ms = (time.time() - start) * 1000
        result["time_ms"] = elapsed_ms
        formatter.candidate_result(prefix, False, diag["reachable_count"],
                                   time_ms=elapsed_ms)

    return result


# ============================================================================
# Batch probing with parallelism (speed optimization)
# ============================================================================

def probe_batch_parallel(
    prefixes: List[str],
    tracker: FRPTracker,
    cache: ProbeCache,
    formatter: OutputFormatter,
    max_workers: int = 4,
    **probe_kwargs,
) -> List[Dict]:
    """
    Probe multiple candidates with limited parallelism.

    Note: We use limited parallelism (max_workers=4) because each individual
    probe already uses concurrency=16 internally. Too much parallelism would
    overwhelm the network stack.
    """
    results = []

    if max_workers <= 1:
        # Sequential (safer, clearer output)
        for p in prefixes:
            r = probe_and_align_candidate(
                p, tracker, cache, formatter, **probe_kwargs)
            results.append(r)
    else:
        # Parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for p in prefixes:
                fut = pool.submit(
                    probe_and_align_candidate,
                    p, tracker, cache, formatter, **probe_kwargs,
                )
                futures[fut] = p

            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    results.append(r)
                except Exception as e:
                    logger.warning(f"Probe failed for {futures[fut]}: {e}")
                    results.append({
                        "prefix": futures[fut],
                        "error": str(e),
                        "is_frp": False,
                    })

    return results


# ============================================================================
# Model Loading
# ============================================================================

def load_models(cfg: Config, checkpoint_dir: str, device: str = "cpu"):
    """Load generator and verifier."""
    logger.info(f"Loading models from: {checkpoint_dir}")

    gen_path = os.path.join(checkpoint_dir, "generator.pt")
    ver_path = os.path.join(checkpoint_dir, "verifier.pt")

    generator = UnifiedFRPGenerator(cfg.model).to(device)
    verifier = EnsembleVerifier(
        ensemble_size=cfg.model.verifier_ensemble_size,
        max_seq_len=cfg.model.verifier_max_seq_len,
        bit_embed_dim=32,
        hidden_dim=cfg.model.verifier_d_model,
        num_layers=cfg.model.verifier_num_layers,
        num_heads=cfg.model.verifier_num_heads,
        condition_dim=cfg.model.d_model,
        dropout=0.1,
    ).to(device)

    if os.path.exists(gen_path):
        # 兼容旧版 PyTorch，移除 weights_only 参数
        ckpt = torch.load(gen_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            generator.load_state_dict(ckpt["model_state_dict"])
        else:
            generator.load_state_dict(ckpt)
        logger.info(f"  ✓ Generator loaded ({os.path.getsize(gen_path)/1e6:.1f}MB)")
    else:
        logger.warning(f"  ✗ Generator not found: {gen_path}")

    if os.path.exists(ver_path):
        # 兼容旧版 PyTorch，移除 weights_only 参数
        verifier.load_state_dict(
            torch.load(ver_path, map_location=device))
        logger.info(f"  ✓ Verifier loaded ({os.path.getsize(ver_path)/1e6:.1f}MB)")
    else:
        logger.warning(f"  ✗ Verifier not found: {ver_path}")
    # if os.path.exists(gen_path):
    #     ckpt = torch.load(gen_path, map_location=device, weights_only=False)
    #     if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    #         generator.load_state_dict(ckpt["model_state_dict"])
    #     else:
    #         generator.load_state_dict(ckpt)
    #     logger.info(f"  ✓ Generator loaded ({os.path.getsize(gen_path)/1e6:.1f}MB)")
    # else:
    #     logger.warning(f"  ✗ Generator not found: {gen_path}")

    # if os.path.exists(ver_path):
    #     verifier.load_state_dict(
    #         torch.load(ver_path, map_location=device, weights_only=False))
    #     logger.info(f"  ✓ Verifier loaded ({os.path.getsize(ver_path)/1e6:.1f}MB)")
    # else:
    #     logger.warning(f"  ✗ Verifier not found: {ver_path}")

    generator.eval()
    verifier.eval()

    gen_params = sum(p.numel() for p in generator.parameters())
    ver_params = sum(p.numel() for p in verifier.parameters())
    logger.info(f"  Params: generator={gen_params:,} verifier={ver_params:,}")

    return generator, verifier


# ============================================================================
# Candidate Generation (batch, fast)
# ============================================================================

def generate_candidates_batch(
    generator: UnifiedFRPGenerator,
    cfg: Config,
    meta: Dict,
    seed_prefixes: List[str] = None,
    bgp_prefix: str = None,
    target_plen: int = 64,
    num_samples: int = 1000,
    temperature: float = 1.0,
    device: str = "cpu",
) -> List[str]:
    """
    Generate candidate prefixes using the generator.

    Speed optimization: single batched forward pass.
    """
    # Build meta batch
    meta_batch = {
        "as_id": torch.tensor([meta.get("as_id", 0)], dtype=torch.long, device=device),
        "org_id": torch.tensor([meta.get("org_id", 0)], dtype=torch.long, device=device),
        "cat1_id": torch.tensor([meta.get("cat1_id", 0)], dtype=torch.long, device=device),
        "cat2_id": torch.tensor([meta.get("cat2_id", 0)], dtype=torch.long, device=device),
        "proto_id": torch.tensor([meta.get("proto_id", 0)], dtype=torch.long, device=device),
        "geo_id": torch.tensor([meta.get("geo_id", 0)], dtype=torch.long, device=device),
        "rtt_stats": torch.tensor(
            [meta.get("rtt_stats", [0.0, 0.0, 0.0])],
            dtype=torch.float32, device=device),
    }

    # Seed encoding
    seed_bits = None
    seed_mask = None
    if seed_prefixes:
        max_k = cfg.model.max_seeds_per_batch
        seq_len = cfg.model.seq_len
        s_bits = torch.zeros((max_k, seq_len), dtype=torch.long)
        s_mask = torch.zeros(max_k, dtype=torch.float32)
        for i, p in enumerate(seed_prefixes[:max_k]):
            b, plen = ipv6_prefix_to_bits(p)
            L = min(len(b), seq_len)
            s_bits[i, :L] = torch.tensor(b[:L], dtype=torch.long)
            s_mask[i] = 1.0
        seed_bits = s_bits.unsqueeze(0).to(device)
        seed_mask = s_mask.unsqueeze(0).to(device)

    # BGP constraint
    bgp_bits_t = None
    bgp_len_t = None
    if bgp_prefix:
        bgp_bits_list, bgp_len = ipv6_prefix_to_bits(bgp_prefix)
        bgp_bits_t = torch.zeros((1, cfg.model.seq_len), dtype=torch.long, device=device)
        for i, b in enumerate(bgp_bits_list[:cfg.model.seq_len]):
            bgp_bits_t[0, i] = b
        bgp_len_t = torch.tensor([bgp_len], dtype=torch.long, device=device)

    target_plen_t = torch.tensor([target_plen], dtype=torch.long, device=device)

    # Generate
    generator.eval()
    with torch.no_grad():
        bits = generator.generate_staged(
            meta=meta_batch,
            seed_bits=seed_bits,
            seed_mask=seed_mask,
            target_prefix_lens=target_plen_t,
            bgp_constraint_bits=bgp_bits_t,
            bgp_constraint_len=bgp_len_t,
            temperature=temperature,
            num_samples=num_samples,
        )

    # Convert to prefix strings + dedup
    candidates = []
    seen = set()
    for i in range(bits.size(0)):
        b = bits[i].tolist()[:target_plen]
        try:
            p = bits_to_ipv6_prefix(b)
        except Exception:
            continue
        p = normalize_prefix(p)
        if p in seen:
            continue
        # BGP coverage check
        if bgp_prefix and not is_subnet_of(p, bgp_prefix):
            continue
        seen.add(p)
        candidates.append(p)

    return candidates


# ============================================================================
# Verifier Scoring (batch, fast)
# ============================================================================

def score_candidates_batch(
    candidates: List[str],
    generator: UnifiedFRPGenerator,
    verifier: EnsembleVerifier,
    meta: Dict,
    cfg: Config,
    device: str = "cpu",
) -> List[Dict]:
    """
    Score candidates with the ensemble verifier.

    Speed optimization: single batched forward pass for all candidates.
    """
    if not candidates:
        return []

    # Build meta for condition encoder
    meta_batch = {
        "as_id": torch.tensor([meta.get("as_id", 0)], dtype=torch.long, device=device),
        "org_id": torch.tensor([meta.get("org_id", 0)], dtype=torch.long, device=device),
        "cat1_id": torch.tensor([meta.get("cat1_id", 0)], dtype=torch.long, device=device),
        "cat2_id": torch.tensor([meta.get("cat2_id", 0)], dtype=torch.long, device=device),
        "proto_id": torch.tensor([meta.get("proto_id", 0)], dtype=torch.long, device=device),
        "geo_id": torch.tensor([meta.get("geo_id", 0)], dtype=torch.long, device=device),
        "rtt_stats": torch.tensor(
            [meta.get("rtt_stats", [0.0, 0.0, 0.0])],
            dtype=torch.float32, device=device),
    }

    bits_list = [ipv6_prefix_to_bits(p)[0] for p in candidates]
    bit_seq, seq_lens = prepare_verifier_input(
        bits_list, cfg.model.verifier_max_seq_len)
    bit_seq = bit_seq.to(device)
    seq_lens = seq_lens.to(device)

    verifier.eval()
    generator.eval()
    with torch.no_grad():
        condition = generator.condition_encoder(meta_batch)  # (1, D)
        cond_expand = condition.expand(len(candidates), -1)
        mean_probs, std_probs = verifier(bit_seq, seq_lens, cond_expand)

    scored = []
    for i, p in enumerate(candidates):
        scored.append({
            "prefix": p,
            "prob": mean_probs[i].item(),
            "std": std_probs[i].item(),
        })

    return scored


# ============================================================================
# Active Learning Selection
# ============================================================================

def select_candidates(
    scored: List[Dict],
    budget: int,
    already_known: List[str] = None,
    cfg: Config = None,
) -> List[Dict]:
    """Select top candidates using active learning sampler."""
    if not scored:
        return []

    sampler = ActiveLearningSampler(
        uncertainty_weight=cfg.train.al_uncertainty_weight if cfg else 0.5,
        diversity_weight=cfg.train.al_diversity_weight if cfg else 0.3,
        representativeness_weight=cfg.train.al_representativeness_weight if cfg else 0.2,
    )

    budget = min(budget, len(scored))
    selected = sampler.select(scored, budget, already_probed=already_known or [])
    return selected


# ============================================================================
# Online Feedback (lightweight wrapper)
# ============================================================================

def do_online_feedback(
    feedback: OnlineFeedback,
    probe_results: List[Dict],
    condition: torch.Tensor,
    meta: Dict,
    formatter: OutputFormatter,
):
    """Report probe results to online feedback for model updates."""
    for r in probe_results:
        if r.get("skipped") or r.get("error"):
            continue
        is_frp = r["is_frp"] or len(r.get("ta_frps", [])) > 0
        feedback.report_probe(
            prefix=r["prefix"],
            is_frp=is_frp,
            condition=condition,
            meta=meta,
        )

    stats = feedback.get_stats()
    if stats["verifier_updates"] > 0 or stats["reptile_updates"] > 0:
        formatter.feedback_update(
            "model_update",
            f"ver={stats['verifier_updates']} rep={stats['reptile_updates']} "
            f"ft={stats['full_finetunes']}"
        )


# ============================================================================
# Mode A: Seed Expansion
# ============================================================================

def run_mode_seed_expand(args, cfg, generator, verifier):
    """
    Seed expansion mode:
      1. Generate candidates conditioned on seed FRPs
      2. Score with verifier
      3. Active learning selection
      4. Probe each selected candidate (ping-style + refine + TA)
      5. Online feedback to update models
    """
    device = cfg.device
    formatter = OutputFormatter(verbose=args.verbose)

    # Load seeds
    with open(args.seeds) as f:
        seeds = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    if not seeds:
        logger.error("No seeds found. Aborting.")
        sys.exit(1)

    meta = parse_meta(args.meta)
    logger.info(f"Seeds: {len(seeds)}, Meta: {meta}")

    # Determine target prefix length from seeds
    seed_plens = [ipv6_prefix_to_bits(s)[1] for s in seeds]
    tgt_plen = int(sorted(seed_plens)[len(seed_plens) // 2]) if seed_plens else 64

    # Initialize components
    tracker = FRPTracker()
    cache = ProbeCache()
    feedback = OnlineFeedback(generator, verifier, cfg, device=device)

    # Add seeds to tracker
    for s in seeds:
        tracker.add_frp(normalize_prefix(s), source="seed")

    config_display = {
        "seeds": len(seeds),
        "target_plen": f"/{tgt_plen}",
        "candidates": args.candidates,
        "budget": args.budget,
        "max_attempts": args.max_attempts,
        "concurrency": args.concurrency,
        "refine": not args.no_refine,
        "tree_align": not args.no_ta,
        "fast_refine": args.fast_refine,
        "device": device,
    }
    formatter.header("seed_expand", config_display)

    total_start = time.time()

    # --- Stage 1: Generate ---
    t0 = time.time()
    logger.info(f"[Stage 1] Generating {args.candidates} candidates...")
    candidates = generate_candidates_batch(
        generator, cfg, meta,
        seed_prefixes=seeds,
        target_plen=tgt_plen,
        num_samples=args.candidates,
        temperature=1.0,
        device=device,
    )
    # Filter already covered
    candidates = [c for c in candidates if not tracker.is_covered(c)]
    gen_time = time.time() - t0
    logger.info(f"  Generated {len(candidates)} unique candidates in {gen_time:.1f}s")

    if not candidates:
        logger.warning("No new candidates generated. Done.")
        return

    # --- Stage 2: Verifier scoring ---
    t0 = time.time()
    logger.info(f"[Stage 2] Scoring {len(candidates)} candidates with verifier...")
    scored = score_candidates_batch(
        candidates, generator, verifier, meta, cfg, device)
    score_time = time.time() - t0
    logger.info(f"  Scored in {score_time:.1f}s "
                f"(avg prob={sum(s['prob'] for s in scored)/len(scored):.3f})")

    # --- Stage 3: Active learning selection ---
    t0 = time.time()
    selected = select_candidates(
        scored, args.budget, already_known=tracker.get_prefixes(), cfg=cfg)
    al_time = time.time() - t0
    logger.info(f"[Stage 3] Selected {len(selected)} candidates "
                f"(AL took {al_time:.1f}s)")

    # --- Stage 4: Probe + Align ---
    logger.info(f"[Stage 4] Probing {len(selected)} candidates "
                f"(ping-style, max_attempts={args.max_attempts})...")
    print()  # newline before table

    probe_kwargs = {
        "timeout": args.timeout,
        "max_attempts": args.max_attempts,
        "concurrency": args.concurrency,
        "min_prefix_len": args.min_prefix_len,
        "do_refine": not args.no_refine,
        "do_ta": not args.no_ta,
        "max_ta_depth": args.max_ta_depth,
        "fast_refine": args.fast_refine,
    }

    probe_results = []
    for s in selected:
        r = probe_and_align_candidate(
            s["prefix"], tracker, cache, formatter,
            verifier_prob=s["prob"], verifier_std=s["std"],
            **probe_kwargs,
        )
        probe_results.append(r)

    probe_time = time.time() - total_start - gen_time - score_time - al_time

    # --- Stage 5: Online feedback ---
    t0 = time.time()
    with torch.no_grad():
        meta_batch = {
            "as_id": torch.tensor([meta.get("as_id", 0)], dtype=torch.long, device=device),
            "org_id": torch.tensor([meta.get("org_id", 0)], dtype=torch.long, device=device),
            "cat1_id": torch.tensor([meta.get("cat1_id", 0)], dtype=torch.long, device=device),
            "cat2_id": torch.tensor([meta.get("cat2_id", 0)], dtype=torch.long, device=device),
            "proto_id": torch.tensor([meta.get("proto_id", 0)], dtype=torch.long, device=device),
            "geo_id": torch.tensor([meta.get("geo_id", 0)], dtype=torch.long, device=device),
            "rtt_stats": torch.tensor(
                [meta.get("rtt_stats", [0.0, 0.0, 0.0])],
                dtype=torch.float32, device=device),
        }
        condition = generator.condition_encoder(meta_batch)[0]

    do_online_feedback(feedback, probe_results, condition, meta, formatter)
    feedback_time = time.time() - t0

    total_elapsed = time.time() - total_start

    # --- Summary ---
    newly_found = tracker.count() - len(seeds)
    hit_rate = sum(1 for r in probe_results if r["is_frp"]) / max(len(probe_results), 1)

    formatter.final_summary({
        "Seeds input": len(seeds),
        "Candidates generated": len(candidates),
        "Candidates scored": len(scored),
        "Candidates probed": len(selected),
        "Direct FRPs found": sum(1 for r in probe_results if r["is_frp"]),
        "TA FRPs found": tracker.ta_new_frps,
        "Total new FRPs": newly_found,
        "Total FRPs (incl seeds)": tracker.count(),
        "Hit rate (direct)": f"{hit_rate:.1%}",
        "Probes": {
            "detection": tracker.detection_probes,
            "refinement": tracker.refinement_probes,
            "tree_align": tracker.ta_probes,
            "total": tracker.total_probes_sent,
        },
        "Cache": cache.stats(),
        "Timing": {
            "generation": f"{gen_time:.1f}s",
            "scoring": f"{score_time:.1f}s",
            "active_learning": f"{al_time:.1f}s",
            "probing": f"{probe_time:.1f}s",
            "feedback": f"{feedback_time:.1f}s",
            "total": f"{total_elapsed:.1f}s",
        },
        "Feedback": feedback.get_stats(),
    })

    # Save results
    save_output(args.output_dir, tracker, probe_results, scored, {
        "mode": "seed_expand",
        "seeds": len(seeds),
        "config": config_display,
        "timing": {
            "generation_s": round(gen_time, 2),
            "scoring_s": round(score_time, 2),
            "al_s": round(al_time, 2),
            "probing_s": round(probe_time, 2),
            "feedback_s": round(feedback_time, 2),
            "total_s": round(total_elapsed, 2),
        },
        "results": {
            "candidates_generated": len(candidates),
            "candidates_probed": len(selected),
            "direct_frps": sum(1 for r in probe_results if r["is_frp"]),
            "ta_frps": tracker.ta_new_frps,
            "total_new_frps": newly_found,
            "total_frps": tracker.count(),
            "hit_rate": round(hit_rate, 4),
        },
        "probes": {
            "detection": tracker.detection_probes,
            "refinement": tracker.refinement_probes,
            "tree_align": tracker.ta_probes,
            "total": tracker.total_probes_sent,
        },
        "cache": cache.stats(),
        "feedback": feedback.get_stats(),
    })


# ============================================================================
# Mode B: Target Discovery
# ============================================================================

# def run_mode_target(args, cfg, generator, verifier):
#     """
#     Target discovery mode:
#       For each BGP prefix:
#         1. Look up AS seeds (mimic vs innovate)
#         2. Generate candidates within BGP prefix
#         3. Score + select
#         4. Probe + align
#         5. Online feedback
#     """
#     device = cfg.device
#     formatter = OutputFormatter(verbose=args.verbose)

#     # Load targets
#     with open(args.targets) as f:
#         targets = [l.strip() for l in f if l.strip() and not l.startswith('#')]
#     if not targets:
#         logger.error("No targets found. Aborting.")
#         sys.exit(1)

#     meta = parse_meta(args.meta)

#     # Load AS seed DB
#     as_seed_db = None
#     if args.as_seed_db and os.path.exists(args.as_seed_db):
#         with open(args.as_seed_db) as f:
#             raw = json.load(f)
#             as_seed_db = {int(k): v for k, v in raw.items()}
#         logger.info(f"Loaded AS seed DB: {len(as_seed_db)} entries")

#     # Initialize
#     tracker = FRPTracker()
#     cache = ProbeCache()
#     feedback = OnlineFeedback(generator, verifier, cfg, device=device)

#     config_display = {
#         "targets": len(targets),
#         "target_plen": f"/{args.target_plen}",
#         "candidates/BGP": args.candidates,
#         "budget/BGP": args.budget,
#         "max_attempts": args.max_attempts,
#         "concurrency": args.concurrency,
#         "refine": not args.no_refine,
#         "tree_align": not args.no_ta,
#         "fast_refine": args.fast_refine,
#         "device": device,
#     }
#     formatter.header("target_discovery", config_display)

#     total_start = time.time()
#     all_probe_results = []
#     all_scored = []
#     per_bgp_stats = []

#     for bgp_idx, bgp_prefix in enumerate(targets):
#         bgp_prefix = normalize_prefix(bgp_prefix)
#         bgp_bits, bgp_len = ipv6_prefix_to_bits(bgp_prefix)
#         as_id = int(meta.get("as_id", 0))

#         # Determine mimic vs innovate
#         seeds = []
#         if as_seed_db and as_id in as_seed_db:
#             seeds = as_seed_db[as_id][:cfg.model.max_seeds_per_batch]
#         mimic_mode = len(seeds) > 0
#         temp = 1.0 if mimic_mode else 1.3

#         mode_str = "mimic" if mimic_mode else "innovate"
#         print(f"\n{'─' * 78}")
#         print(f"  BGP [{bgp_idx+1}/{len(targets)}]: {bgp_prefix} "
#               f"(AS={as_id}, mode={mode_str}, seeds={len(seeds)})")
#         print(f"{'─' * 78}")

#         bgp_start = time.time()

#         # Generate
#         candidates = generate_candidates_batch(
#             generator, cfg, meta,
#             seed_prefixes=seeds if mimic_mode else None,
#             bgp_prefix=bgp_prefix,
#             target_plen=args.target_plen,
#             num_samples=args.candidates,
#             temperature=temp,
#             device=device,
#         )
#         candidates = [c for c in candidates if not tracker.is_covered(c)]

#         if not candidates:
#             logger.info(f"  No new candidates for {bgp_prefix}")
#             per_bgp_stats.append({
#                 "bgp": bgp_prefix, "mode": mode_str,
#                 "generated": 0, "probed": 0, "frps": 0,
#             })
#             continue

#         # Score
#         scored = score_candidates_batch(
#             candidates, generator, verifier, meta, cfg, device)
#         all_scored.extend(scored)

#         # Select
#         selected = select_candidates(
#             scored, args.budget, already_known=tracker.get_prefixes(), cfg=cfg)

#         logger.info(f"  Generated {len(candidates)}, scored, "
#                     f"selected {len(selected)} for probing")

#         # Probe
#         probe_kwargs = {
#             "timeout": args.timeout,
#             "max_attempts": args.max_attempts,
#             "concurrency": args.concurrency,
#             "min_prefix_len": max(bgp_len, args.min_prefix_len),
#             "do_refine": not args.no_refine,
#             "do_ta": not args.no_ta,
#             "max_ta_depth": args.max_ta_depth,
#             "fast_refine": args.fast_refine,
#         }

#         bgp_results = []
#         for s in selected:
#             r = probe_and_align_candidate(
#                 s["prefix"], tracker, cache, formatter,
#                 verifier_prob=s["prob"], verifier_std=s["std"],
#                 **probe_kwargs,
#             )
#             bgp_results.append(r)
#             all_probe_results.append(r)

#         # Online feedback
#         with torch.no_grad():
#             meta_batch = {
#                 "as_id": torch.tensor([as_id], dtype=torch.long, device=device),
#                 "org_id": torch.tensor([meta.get("org_id", 0)], dtype=torch.long, device=device),
#                 "cat1_id": torch.tensor([meta.get("cat1_id", 0)], dtype=torch.long, device=device),
#                 "cat2_id": torch.tensor([meta.get("cat2_id", 0)], dtype=torch.long, device=device),
#                 "proto_id": torch.tensor([meta.get("proto_id", 0)], dtype=torch.long, device=device),
#                 "geo_id": torch.tensor([meta.get("geo_id", 0)], dtype=torch.long, device=device),
#                 "rtt_stats": torch.tensor(
#                     [meta.get("rtt_stats", [0.0, 0.0, 0.0])],
#                     dtype=torch.float32, device=device),
#             }
#             condition = generator.condition_encoder(meta_batch)[0]
#         do_online_feedback(feedback, bgp_results, condition, meta, formatter)

#         bgp_elapsed = time.time() - bgp_start
#         bgp_frps = sum(1 for r in bgp_results if r["is_frp"])
#         bgp_ta = sum(len(r.get("ta_frps", [])) for r in bgp_results)

#         per_bgp_stats.append({
#             "bgp": bgp_prefix, "mode": mode_str,
#             "generated": len(candidates), "probed": len(selected),
#             "frps": bgp_frps + bgp_ta, "time_s": round(bgp_elapsed, 1),
#         })

#         print(f"  → {bgp_prefix}: +{bgp_frps} direct, +{bgp_ta} TA "
#               f"({bgp_elapsed:.1f}s)")

#     total_elapsed = time.time() - total_start

#     # Final summary
#     total_frps = tracker.count()
#     total_probed = len(all_probe_results)
#     direct_frps = sum(1 for r in all_probe_results if r["is_frp"])

#     formatter.final_summary({
#         "BGP prefixes": f"{len(targets)} input",
#         "Total candidates generated": sum(s["generated"] for s in per_bgp_stats),
#         "Total candidates probed": total_probed,
#         "Direct FRPs": direct_frps,
#         "TA FRPs": tracker.ta_new_frps,
#         "Total FRPs": total_frps,
#         "Hit rate": f"{direct_frps / max(total_probed, 1):.1%}",
#         "Probes": {
#             "detection": tracker.detection_probes,
#             "refinement": tracker.refinement_probes,
#             "tree_align": tracker.ta_probes,
#             "total": tracker.total_probes_sent,
#         },
#         "Cache": cache.stats(),
#         "Timing": {
#             "total": f"{total_elapsed:.1f}s",
#             "avg_per_bgp": f"{total_elapsed/max(len(targets),1):.1f}s",
#         },
#         "Feedback": feedback.get_stats(),
#     })

#     # Save
#     save_output(args.output_dir, tracker, all_probe_results, all_scored, {
#         "mode": "target",
#         "targets": len(targets),
#         "config": config_display,
#         "per_bgp": per_bgp_stats,
#         "timing": {"total_s": round(total_elapsed, 2)},
#         "results": {
#             "total_probed": total_probed,
#             "direct_frps": direct_frps,
#             "ta_frps": tracker.ta_new_frps,
#             "total_frps": total_frps,
#         },
#         "probes": {
#             "detection": tracker.detection_probes,
#             "refinement": tracker.refinement_probes,
#             "tree_align": tracker.ta_probes,
#             "total": tracker.total_probes_sent,
#         },
#         "cache": cache.stats(),
#         "feedback": feedback.get_stats(),
#     })
# 在 run_mode_target 函数中增加调试信息和新逻辑
def run_mode_target(args, cfg, generator, verifier):
    """
    Target discovery mode:
      For each BGP prefix:
        1. Look up AS seeds (mimic vs innovate)
        2. Generate candidates within BGP prefix
        3. Score + select
        4. Probe + align
        5. Online feedback
    """
    device = cfg.device
    formatter = OutputFormatter(verbose=args.verbose)

    # Load targets
    with open(args.targets) as f:
        targets = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    if not targets:
        logger.error("No targets found. Aborting.")
        sys.exit(1)

    meta = parse_meta(args.meta)

    # Load AS seed DB
    as_seed_db = None
    if args.as_seed_db and os.path.exists(args.as_seed_db):
        with open(args.as_seed_db) as f:
            raw = json.load(f)
            as_seed_db = {int(k): v for k, v in raw.items()}
        logger.info(f"Loaded AS seed DB: {len(as_seed_db)} entries")

    # 加载已有种子前缀
    existing_seeds = set()
    if args.seeds and os.path.exists(args.seeds):
        with open(args.seeds) as f:
            existing_seeds = {l.strip() for l in f if l.strip() and not l.startswith('#')}
        logger.info(f"Loaded {len(existing_seeds)} existing seeds from {args.seeds}")
    
    # 加载 meta.json 以获取 AS 信息
    meta_json_path = args.meta_json if hasattr(args, 'meta_json') and args.meta_json else 'output/prepared_new/meta.json'
    meta_data = {}
    if os.path.exists(meta_json_path):
        with open(meta_json_path) as f:
            meta_data = json.load(f)
        logger.info(f"Loaded meta data with {len(meta_data)} prefixes from {meta_json_path}")

    # 初始化跟踪器并添加现有种子作为已知FRP
    tracker = FRPTracker()
    cache = ProbeCache()
    feedback = OnlineFeedback(generator, verifier, cfg, device=device)

    # 添加现有种子到跟踪器
    for seed in existing_seeds:
        tracker.add_frp(normalize_prefix(seed), source="existing_seed")

    config_display = {
        "targets": len(targets),
        "target_plen": f"/{args.target_plen}",
        "candidates/BGP": args.candidates,
        "budget/BGP": args.budget,
        "max_attempts": args.max_attempts,
        "concurrency": args.concurrency,
        "refine": not args.no_refine,
        "tree_align": not args.no_ta,
        "fast_refine": args.fast_refine,
        "device": device,
    }
    formatter.header("target_discovery", config_display)

    total_start = time.time()
    all_probe_results = []
    all_scored = []
    per_bgp_stats = []

    for bgp_idx, bgp_prefix in enumerate(targets):
        bgp_prefix = normalize_prefix(bgp_prefix)
        bgp_bits, bgp_len = ipv6_prefix_to_bits(bgp_prefix)
        
        # 调试：查找BGP前缀在已有种子中的父子关系
        logger.debug(f"[DEBUG] Processing BGP prefix: {bgp_prefix}")
        
        # 检查是否为现有种子的父前缀
        parent_seeds = []
        for seed in existing_seeds:
            if is_supernet_of(seed, bgp_prefix):
                parent_seeds.append(seed)
        
        # 检查是否为现有种子的子前缀
        child_seeds = []
        for seed in existing_seeds:
            if is_subnet_of(seed, bgp_prefix):
                child_seeds.append(seed)
        
        # 检查BGP前缀本身是否有meta信息
        bgp_meta_info = meta_data.get(bgp_prefix, {})
        if bgp_meta_info:
            logger.debug(f"[DEBUG] Found direct meta info for {bgp_prefix}")
        else:
            logger.warning(f"[WARNING] No direct meta info found for {bgp_prefix}")
            
            # 尝试从子前缀获取AS信息
            child_as_info = find_as_from_children(bgp_prefix, meta_data)
            if child_as_info:
                logger.info(f"[INFO] Found AS info from children for {bgp_prefix}: {child_as_info}")
                bgp_meta_info = child_as_info
            else:
                # 尝试从父前缀获取AS信息
                parent_as_info = find_as_from_parents(bgp_prefix, meta_data)
                if parent_as_info:
                    logger.info(f"[INFO] Found AS info from parents for {bgp_prefix}: {parent_as_info}")
                    bgp_meta_info = parent_as_info
                else:
                    logger.warning(f"[WARNING] No AS info found for {bgp_prefix} or its relatives, using default")
                    bgp_meta_info = meta  # 使用默认meta
        
        if parent_seeds:
            logger.info(f"[INFO] Found {len(parent_seeds)} parent seeds for {bgp_prefix}: {parent_seeds}")
        if child_seeds:
            logger.info(f"[INFO] Found {len(child_seeds)} child seeds in {bgp_prefix}: {child_seeds}")
        
        # 将匹配的种子添加到结果中
        for seed in parent_seeds + child_seeds:
            if tracker.add_frp(seed, source="overlap_seed", original=bgp_prefix):
                logger.info(f"[INFO] Added overlapping seed {seed} from {bgp_prefix}")
        
        # 获取对应BGP前缀的AS ID
        as_id = bgp_meta_info.get("as_id", meta.get("as_id", 0))
        logger.debug(f"[DEBUG] Using AS ID {as_id} for {bgp_prefix}")

        # Determine mimic vs innovate
        seeds = []
        if as_seed_db and as_id in as_seed_db:
            seeds = as_seed_db[as_id][:cfg.model.max_seeds_per_batch]
        mimic_mode = len(seeds) > 0
        temp = 1.0 if mimic_mode else 1.3

        mode_str = "mimic" if mimic_mode else "innovate"
        print(f"\n{'─' * 78}")
        print(f"  BGP [{bgp_idx+1}/{len(targets)}]: {bgp_prefix} "
              f"(AS={as_id}, mode={mode_str}, seeds={len(seeds)})")
        print(f"{'─' * 78}")

        bgp_start = time.time()

        # Generate
        candidates = generate_candidates_batch(
            generator, cfg, bgp_meta_info,  # 使用BGP特定的元数据
            seed_prefixes=seeds if mimic_mode else None,
            bgp_prefix=bgp_prefix,
            target_plen=args.target_plen,
            num_samples=args.candidates,
            temperature=temp,
            device=device,
        )
        candidates = [c for c in candidates if not tracker.is_covered(c)]

        if not candidates:
            logger.info(f"  No new candidates for {bgp_prefix}")
            per_bgp_stats.append({
                "bgp": bgp_prefix, "mode": mode_str,
                "generated": 0, "probed": 0, "frps": 0,
                "parent_seeds": len(parent_seeds),
                "child_seeds": len(child_seeds)
            })
            continue

        # Score
        scored = score_candidates_batch(
            candidates, generator, verifier, bgp_meta_info, cfg, device)
        all_scored.extend(scored)

        # Select
        selected = select_candidates(
            scored, args.budget, already_known=tracker.get_prefixes(), cfg=cfg)

        logger.info(f"  Generated {len(candidates)}, scored, "
                    f"selected {len(selected)} for probing")

        # Probe
        probe_kwargs = {
            "timeout": args.timeout,
            "max_attempts": args.max_attempts,
            "concurrency": args.concurrency,
            "min_prefix_len": max(bgp_len, args.min_prefix_len),
            "do_refine": not args.no_refine,
            "do_ta": not args.no_ta,
            "max_ta_depth": args.max_ta_depth,
            "fast_refine": args.fast_refine,
        }

        bgp_results = []
        for s in selected:
            r = probe_and_align_candidate(
                s["prefix"], tracker, cache, formatter,
                verifier_prob=s["prob"], verifier_std=s["std"],
                **probe_kwargs,
            )
            bgp_results.append(r)
            all_probe_results.append(r)

        # Online feedback
        with torch.no_grad():
            # 使用BGP特定的元数据
            meta_batch = {
                "as_id": torch.tensor([bgp_meta_info.get("as_id", 0)], dtype=torch.long, device=device),
                "org_id": torch.tensor([bgp_meta_info.get("org_id", 0)], dtype=torch.long, device=device),
                "cat1_id": torch.tensor([bgp_meta_info.get("cat1_id", 0)], dtype=torch.long, device=device),
                "cat2_id": torch.tensor([bgp_meta_info.get("cat2_id", 0)], dtype=torch.long, device=device),
                "proto_id": torch.tensor([bgp_meta_info.get("proto_id", 0)], dtype=torch.long, device=device),
                "geo_id": torch.tensor([bgp_meta_info.get("geo_id", 0)], dtype=torch.long, device=device),
                "rtt_stats": torch.tensor(
                    [bgp_meta_info.get("rtt_stats", [0.0, 0.0, 0.0])],
                    dtype=torch.float32, device=device),
            }
            condition = generator.condition_encoder(meta_batch)[0]
        do_online_feedback(feedback, bgp_results, condition, bgp_meta_info, formatter)

        bgp_elapsed = time.time() - bgp_start
        bgp_frps = sum(1 for r in bgp_results if r["is_frp"])
        bgp_ta = sum(len(r.get("ta_frps", [])) for r in bgp_results)

        per_bgp_stats.append({
            "bgp": bgp_prefix, "mode": mode_str,
            "generated": len(candidates), "probed": len(selected),
            "frps": bgp_frps + bgp_ta, "time_s": round(bgp_elapsed, 1),
            "parent_seeds": len(parent_seeds),
            "child_seeds": len(child_seeds)
        })

        print(f"  → {bgp_prefix}: +{bgp_frps} direct, +{bgp_ta} TA "
              f"({bgp_elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    # Final summary
    total_frps = tracker.count()
    total_probed = len(all_probe_results)
    direct_frps = sum(1 for r in all_probe_results if r["is_frp"])

    formatter.final_summary({
        "BGP prefixes": f"{len(targets)} input",
        "Total candidates generated": sum(s["generated"] for s in per_bgp_stats),
        "Total candidates probed": total_probed,
        "Direct FRPs": direct_frps,
        "TA FRPs": tracker.ta_new_frps,
        "Total FRPs": total_frps,
        "Hit rate": f"{direct_frps / max(total_probed, 1):.1%}",
        "Probes": {
            "detection": tracker.detection_probes,
            "refinement": tracker.refinement_probes,
            "tree_align": tracker.ta_probes,
            "total": tracker.total_probes_sent,
        },
        "Cache": cache.stats(),
        "Timing": {
            "total": f"{total_elapsed:.1f}s",
            "avg_per_bgp": f"{total_elapsed/max(len(targets),1):.1f}s",
        },
        "Feedback": feedback.get_stats(),
    })

    # Save
    save_output(args.output_dir, tracker, all_probe_results, all_scored, {
        "mode": "target",
        "targets": len(targets),
        "config": config_display,
        "per_bgp": per_bgp_stats,
        "timing": {"total_s": round(total_elapsed, 2)},
        "results": {
            "total_probed": total_probed,
            "direct_frps": direct_frps,
            "ta_frps": tracker.ta_new_frps,
            "total_frps": total_frps,
        },
        "probes": {
            "detection": tracker.detection_probes,
            "refinement": tracker.refinement_probes,
            "tree_align": tracker.ta_probes,
            "total": tracker.total_probes_sent,
        },
        "cache": cache.stats(),
        "feedback": feedback.get_stats(),
    })


def find_as_from_children(parent_prefix, meta_data):
    """
    从子前缀中查找AS信息
    """
    parent_net = ipaddress.IPv6Network(parent_prefix, strict=False)
    parent_len = parent_net.prefixlen
    
    # 收集所有子前缀的AS信息
    child_as_infos = []
    for prefix_str, meta_info in meta_data.items():
        try:
            child_net = ipaddress.IPv6Network(prefix_str, strict=False)
            # 检查child_net是否是parent_net的子网
            if child_net.prefixlen > parent_len and is_subnet_of(prefix_str, parent_prefix):
                child_as_infos.append(meta_info)
        except ValueError:
            continue  # 忽略无效的前缀
    
    if not child_as_infos:
        return None
    
    # 返回最常见的AS信息或第一个
    from collections import Counter
    as_ids = [info.get('as_id') for info in child_as_infos if info.get('as_id')]
    if as_ids:
        most_common_as_id = Counter(as_ids).most_common(1)[0][0]
        # 找到第一个具有最常见AS ID的记录作为代表
        for info in child_as_infos:
            if info.get('as_id') == most_common_as_id:
                return info
    
    # 如果没有找到有效的AS ID，返回第一个可用的记录
    return child_as_infos[0] if child_as_infos else None


def find_as_from_parents(child_prefix, meta_data):
    """
    从父前缀中查找AS信息
    """
    child_net = ipaddress.IPv6Network(child_prefix, strict=False)
    child_len = child_net.prefixlen
    
    # 收集所有父前缀的AS信息
    parent_as_infos = []
    for prefix_str, meta_info in meta_data.items():
        try:
            parent_net = ipaddress.IPv6Network(prefix_str, strict=False)
            # 检查parent_net是否是child_net的父网
            if parent_net.prefixlen < child_len and is_supernet_of(child_prefix, prefix_str):
                parent_as_infos.append(meta_info)
        except ValueError:
            continue  # 忽略无效的前缀
    
    if not parent_as_infos:
        return None
    
    # 返回最常见的AS信息或第一个
    from collections import Counter
    as_ids = [info.get('as_id') for info in parent_as_infos if info.get('as_id')]
    if as_ids:
        most_common_as_id = Counter(as_ids).most_common(1)[0][0]
        # 找到第一个具有最常见AS ID的记录作为代表
        for info in parent_as_infos:
            if info.get('as_id') == most_common_as_id:
                return info
    
    # 如果没有找到有效的AS ID，返回第一个可用的记录
    return parent_as_infos[0] if parent_as_infos else None


def is_subnet_of(child_str, parent_str):
    """判断 child 是否为 parent 的子前缀（或相同）"""
    try:
        child = ipaddress.IPv6Network(child_str, strict=False)
        parent = ipaddress.IPv6Network(parent_str, strict=False)
        return child.subnet_of(parent)
    except ValueError:
        return False


def is_supernet_of(child_str, parent_str):
    """判断 parent 是否为 child 的父前缀（或相同）"""
    try:
        child = ipaddress.IPv6Network(child_str, strict=False)
        parent = ipaddress.IPv6Network(parent_str, strict=False)
        return child.supernet_of(parent)
    except ValueError:
        return False


# ============================================================================
# Output Saving
# ============================================================================

def save_output(output_dir: str, tracker: FRPTracker,
                probe_results: List[Dict], scored: List[Dict],
                summary: Dict):
    """Save all results to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # FRP list
    frp_path = os.path.join(output_dir, "final_frps.txt")
    with open(frp_path, 'w') as f:
        for p in tracker.get_prefixes():
            f.write(f"{p}\n")

    # Probe results (compact)
    probed_path = os.path.join(output_dir, "probed.json")
    compact_results = []
    for r in probe_results:
        compact = {k: v for k, v in r.items()
                   if k not in ("refinement_detail",)}
        compact_results.append(compact)
    with open(probed_path, 'w') as f:
        json.dump(compact_results, f, indent=2, ensure_ascii=False, default=str)

    # Scored candidates
    scored_path = os.path.join(output_dir, "candidates_scored.json")
    with open(scored_path, 'w') as f:
        json.dump(scored, f, indent=2, ensure_ascii=False)

    # Summary
    summary["timestamp"] = datetime.now().isoformat()
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    # FRP detail
    detail_path = os.path.join(output_dir, "frps_detail.json")
    with open(detail_path, 'w') as f:
        json.dump(tracker.confirmed, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nResults saved to {output_dir}/")
    logger.info(f"  final_frps.txt       : {tracker.count()} FRPs")
    logger.info(f"  probed.json          : {len(probe_results)} probe records")
    logger.info(f"  candidates_scored.json: {len(scored)} scored candidates")
    logger.info(f"  summary.json         : run statistics")
    logger.info(f"  frps_detail.json     : FRP provenance details")


# ============================================================================
# Helpers
# ============================================================================

def parse_meta(meta_str: Optional[str]) -> Dict:
    """Parse metadata from CLI."""
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
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="6FRP-AR Standalone-APD Inference (ping-style + downward TA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Seed expansion (simulated)
  python3 infer_standalone.py --mode seed_expand \\
      --seeds data/seeds.txt --simulate data/known_frps.txt

  # Target discovery (live, requires root)
  sudo python3 infer_standalone.py --mode target \\
      --targets data/bgp.txt --as-seed-db data/as_seeds.json

  # Fast mode (fewer attempts, fast refine)
  sudo python3 infer_standalone.py --mode seed_expand \\
      --seeds data/seeds.txt --max-attempts 3 --fast-refine

  # No tree alignment (detection + refine only)
  sudo python3 infer_standalone.py --mode seed_expand \\
      --seeds data/seeds.txt --no-ta
""")

    # Mode
    parser.add_argument('--mode', required=True,
                        choices=['seed_expand', 'target'],
                        help='Inference mode')

    # Model
    parser.add_argument('--checkpoint-dir', default='output/models',
                        help='Directory with generator.pt and verifier.pt')
    parser.add_argument('--device', default='cpu',
                        help='Compute device (cpu/cuda/mps)')
    parser.add_argument('--vocabs', default=None,
                        help='vocabs.json for embedding size auto-fit')

    # Output
    parser.add_argument('--output-dir', default='output/standalone_results',
                        help='Output directory')

    # Generation
    parser.add_argument('--candidates', type=int, default=1000,
                        help='Candidates to generate per round')
    parser.add_argument('--budget', type=int, default=200,
                        help='Active learning budget (probes per round)')

    # Probing (standalone_apd style)
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help=f'Per-attempt timeout (default: {DEFAULT_TIMEOUT}s)')
    parser.add_argument('--max-attempts', type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help=f'Max ping attempts per address (default: {DEFAULT_MAX_ATTEMPTS})')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Parallel probes within prefix (default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('--min-prefix-len', type=int, default=DEFAULT_MIN_PREFIX_LEN,
                        help=f'Min prefix length for refinement (default: {DEFAULT_MIN_PREFIX_LEN})')

    # Alignment
    parser.add_argument('--no-refine', action='store_true',
                        help='Disable refinement (skip finding shortest FRP)')
    parser.add_argument('--no-ta', action='store_true',
                        help='Disable tree alignment (skip sub-FRP discovery)')
    parser.add_argument('--fast-refine', action='store_true',
                        help='Use fast 4-bit-then-1-bit refinement')
    parser.add_argument('--max-ta-depth', type=int, default=DEFAULT_MAX_TA_DEPTH,
                        help=f'Max tree alignment recursion depth (default: {DEFAULT_MAX_TA_DEPTH})')

    # Mode A
    parser.add_argument('--seeds', default=None,
                        help='[seed_expand] File with seed FRP prefixes')

    # Mode B
    parser.add_argument('--targets', default=None,
                        help='[target] File with BGP prefixes')
    parser.add_argument('--as-seed-db', default=None,
                        help='[target] JSON: AS ID -> seed FRP list')
    parser.add_argument('--target-plen', type=int, default=64,
                        help='[target] Target prefix length (default: 64)')
    # 在 main() 函数的参数解析部分添加
    parser.add_argument('--meta-file', default='output/prepared_new/meta.json',
                        help='Path to meta.json file containing AS information')
    parser.add_argument('--seeds-target', default='output/prepared_new/seeds_prepared.txt',
                        help='[target] File with seed FRP prefixes (needed for target mode)')

    # Shared
    parser.add_argument('--meta', default=None,
                        help='Metadata: JSON or key=val,key=val')
    parser.add_argument('--simulate', default=None,
                        help='Simulate probing (not implemented in standalone mode, '
                             'use for model-only testing)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output (show refine steps, feedback updates)')

    args = parser.parse_args()

    # Verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Root check for live probing
    if not args.simulate:
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            logger.error("Live probing requires root. Use: sudo python3 infer_standalone.py ...")
            logger.error("Or use --simulate for model-only testing (no actual probing).")
            sys.exit(1)
        if not has_raw_socket_capability():
            logger.error("Cannot create raw ICMPv6 socket. Check permissions.")
            sys.exit(1)

    # Build config
    cfg = Config()
    cfg.device = args.device
    cfg.train.al_budget_per_round = args.budget
    cfg.train.al_candidates_per_round = args.candidates
    cfg.apd.timeout = args.timeout
    cfg.apd.concurrency = args.concurrency

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
        logger.info(f"Vocab sizes adjusted from {args.vocabs}")

    # Load models
    generator, verifier = load_models(cfg, args.checkpoint_dir, args.device)

    # Dispatch
    if args.mode == 'seed_expand':
        if not args.seeds:
            parser.error("--mode seed_expand requires --seeds")
        run_mode_seed_expand(args, cfg, generator, verifier)
    elif args.mode == 'target':
        if not args.targets:
            parser.error("--mode target requires --targets")
        run_mode_target(args, cfg, generator, verifier)


if __name__ == "__main__":
    main()
