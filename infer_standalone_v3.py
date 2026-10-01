#!/usr/bin/env python3
"""
6FRP-AR v3 Standalone-APD Inference Pipeline (Enhanced v3).

Enhancements over v2:
  1. Seed Preprocessing: APD + refine + upward expansion + TA for each seed
  2. Metadata-guided generation with adaptive candidate counts per AS
  3. Target mode: seed discovery/reuse + cross-domain transfer
  4. Global FRP management: parent/child dedup, real-time sync
  5. AS-level performance tracking and adaptive budget allocation

Usage:
    # Seed expansion with preprocessing
    sudo python3 infer_standalone.py --mode seed_expand \\
        --seeds data/seeds.txt --checkpoint-dir output/models \\
        --output-dir output/run_v3 --vocabs output/vocabs.json

    # Target discovery with seed reuse
    sudo python3 infer_standalone.py --mode target \\
        --targets data/bgp_prefixes.txt --seeds data/seeds.txt \\
        --as-seed-db data/as_seeds.json --meta-file output/prepared_new/meta.json \\
        --checkpoint-dir output/models --output-dir output/run_target_v3

    # Fast mode
    sudo python3 infer_standalone.py --mode seed_expand \\
        --seeds data/seeds.txt --max-attempts 3 --fast-refine --concurrency 32
"""

import argparse
import ipaddress
import json
import logging
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
_pkg_name = os.path.basename(_here)  # "my_design_final"

if _parent not in sys.path:
    sys.path.insert(0, _parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from config import Config
from generators.unified import UnifiedFRPGenerator
from verifier.verifier import EnsembleVerifier, prepare_verifier_input
from utils.ipv6_utils import (
    ipv6_prefix_to_bits, normalize_prefix, bits_to_ipv6_prefix,
    is_subnet_of, parent_prefix, siblings_at_ancestor,
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
# Constants
# ============================================================================

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_TIMEOUT = 2.0
DEFAULT_CONCURRENCY = 16
DEFAULT_MIN_PREFIX_LEN = 16
DEFAULT_MAX_TA_DEPTH = 4
DEFAULT_UPWARD_EXPAND_BITS = 4  # Expand k bits upward from refined seed


# ============================================================================
# Probe Cache
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

    def stats(self) -> Dict:
        total = self.hits + self.misses
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_hit_rate": round(self.hits / max(total, 1), 3),
            "cache_size": len(self._cache),
        }


# ============================================================================
# Global FRP Manager
# ============================================================================

class GlobalFRPManager:
    """
    Global FRP database with parent/child dedup logic:
      - New prefix is child of existing FRP → skip (already covered)
      - New prefix is parent of existing FRP → replace (more general)
      - New prefix is unrelated → add
    """

    def __init__(self):
        self.confirmed: List[Dict] = []
        self.total_probes_sent = 0
        self.detection_probes = 0
        self.refinement_probes = 0
        self.ta_probes = 0
        self.ta_new_frps = 0
        self._prefix_set: Set[str] = set()

    def add_frp(self, prefix: str, source: str, original: str = None,
                extra: Dict = None) -> bool:
        """
        Add a confirmed FRP with parent/child dedup.
        Returns True if actually new (not already covered).
        """
        prefix = normalize_prefix(prefix)
        try:
            prefix_net = ipaddress.IPv6Network(prefix, strict=False)
        except Exception:
            return False

        # Check if already covered by an existing broader FRP
        for f in self.confirmed:
            f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
            if is_covered_by(prefix_net, f_net):
                return False  # Already covered

        # Remove existing FRPs that are covered by this new broader one
        kept = []
        removed = []
        for f in self.confirmed:
            f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
            if is_covered_by(f_net, prefix_net):
                removed.append(f["prefix"])
            else:
                kept.append(f)
        self.confirmed = kept
        self._prefix_set = {f["prefix"] for f in self.confirmed}

        if removed:
            logger.info(f"    [FRP-DB] New /{prefix_net.prefixlen} replaces "
                        f"{len(removed)} narrower FRPs: {removed[:3]}...")

        entry = {
            "prefix": prefix,
            "source": source,
            "original": original or prefix,
            "timestamp": time.time(),
        }
        if extra:
            entry.update(extra)
        self.confirmed.append(entry)
        self._prefix_set.add(prefix)
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

    def get_prefixes_by_source(self, source: str) -> List[str]:
        return [f["prefix"] for f in self.confirmed if f["source"] == source]

    def count(self) -> int:
        return len(self.confirmed)

    def get_frps_in_bgp(self, bgp_prefix: str) -> List[str]:
        """Find all confirmed FRPs that fall within a BGP prefix."""
        results = []
        try:
            bgp_net = ipaddress.IPv6Network(bgp_prefix, strict=False)
        except Exception:
            return results
        for f in self.confirmed:
            try:
                f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
                if is_covered_by(f_net, bgp_net):
                    results.append(f["prefix"])
            except Exception:
                continue
        return results

    def sync_to_file(self, filepath: str):
        """Real-time sync to disk."""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            for p in self.get_prefixes():
                f.write(f"{p}\n")


# ============================================================================
# AS Performance Tracker (adaptive budget)
# ============================================================================

class ASPerformanceTracker:
    """
    Track per-AS probing performance for adaptive budget allocation.

    High-yield AS → more candidates next round
    Low-yield AS → fewer candidates
    """

    def __init__(self, base_candidates: int = 1000, base_budget: int = 200):
        self.base_candidates = base_candidates
        self.base_budget = base_budget
        self._stats: Dict[int, Dict] = defaultdict(lambda: {
            "total_probed": 0,
            "total_frps": 0,
            "hit_rate": 0.0,
            "rounds": 0,
        })

    def report(self, as_id: int, probed: int, frps_found: int):
        """Report results for an AS."""
        s = self._stats[as_id]
        s["total_probed"] += probed
        s["total_frps"] += frps_found
        s["rounds"] += 1
        s["hit_rate"] = s["total_frps"] / max(s["total_probed"], 1)

    def get_adaptive_candidates(self, as_id: int, known_prefix_count: int = 0) -> int:
        """
        Get adaptive candidate count for an AS.

        Rules:
          - Large AS (>1000 known prefixes): 2x base
          - High hit rate (>30%): 1.5x base
          - Low hit rate (<5%): 0.5x base
          - New AS (no history): 1x base
        """
        s = self._stats[as_id]

        # Size factor
        if known_prefix_count > 1000:
            size_factor = 2.0
        elif known_prefix_count > 100:
            size_factor = 1.5
        else:
            size_factor = 1.0

        # Performance factor
        if s["rounds"] == 0:
            perf_factor = 1.0  # No history
        elif s["hit_rate"] > 0.3:
            perf_factor = 1.5
        elif s["hit_rate"] > 0.1:
            perf_factor = 1.0
        elif s["hit_rate"] > 0.05:
            perf_factor = 0.75
        else:
            perf_factor = 0.5

        return int(self.base_candidates * size_factor * perf_factor)

    def get_adaptive_budget(self, as_id: int) -> int:
        """Get adaptive budget for an AS."""
        s = self._stats[as_id]
        if s["rounds"] == 0:
            return self.base_budget
        if s["hit_rate"] > 0.3:
            return int(self.base_budget * 1.5)
        elif s["hit_rate"] < 0.05:
            return int(self.base_budget * 0.5)
        return self.base_budget

    def get_stats(self) -> Dict:
        return dict(self._stats)


# ============================================================================
# Output Formatter
# ============================================================================

class OutputFormatter:
    """Real-time formatted output."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._idx = 0

    def header(self, mode: str, config: Dict):
        print(f"\n{'═' * 78}")
        print(f"  6FRP-AR v3 Standalone-APD Inference (Enhanced)")
        print(f"  Mode: {mode.upper().replace('_', ' ')}")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─' * 78}")
        for k, v in config.items():
            print(f"  {k:24s}: {v}")
        print(f"{'═' * 78}\n")
        sys.stdout.flush()

    def section(self, title: str):
        print(f"\n{'─' * 78}")
        print(f"  {title}")
        print(f"{'─' * 78}")
        sys.stdout.flush()

    def probe_table_header(self):
        print(f"\n{'#':>4} {'Candidate':<44} {'Status':<12} {'Detail'}")
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

    def seed_preprocess(self, seed: str, refined: str, expanded: str,
                        ta_frps: int):
        print(f"  {seed:<44} → refined: {refined}")
        if expanded:
            print(f"{'':>4}  ↑ expanded to {expanded}, TA found {ta_frps} new FRPs")
        sys.stdout.flush()

    def feedback_update(self, update_type: str, detail: str = ""):
        if self.verbose:
            print(f"{'':>4}  ⟳ {update_type}: {detail}")
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
# Core: Probe + Align single candidate
# ============================================================================

def probe_and_align_candidate(
    prefix: str,
    tracker: GlobalFRPManager,
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
    source: str = "generated",
) -> Dict:
    """
    Probe a single candidate:
      1. Ping-style diagnose (16 sub-prefixes × max_attempts)
      2. If FRP → refine (find shortest FRP)
      3. If partial → tree_align (find sub-FRPs recursively)
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

    if tracker.is_covered(prefix):
        result["skipped"] = "covered"
        return result

    start = time.time()
    formatter.candidate_start(prefix, verifier_prob, verifier_std)

    # Step 1: Diagnose
    cached = cache.get(prefix)
    if cached:
        diag = cached
    else:
        diag = diagnose_prefix(
            prefix, timeout=timeout, max_attempts=max_attempts,
            concurrency=concurrency,
        )
        cache.put(prefix, diag)

    tracker.detection_probes += diag["total_probes_sent"]
    tracker.total_probes_sent += diag["total_probes_sent"]
    result["reach_count"] = diag["reachable_count"]
    result["probes_used"] += diag["total_probes_sent"]

    # Step 2: Determine FRP status
    is_frp, reason = determine_frp_status(diag)
    result["is_frp"] = is_frp

    if is_frp:
        # Step 3a: Refine
        refined_prefix = prefix
        if do_refine:
            ref = refine_frp_prefix(
                prefix, timeout=timeout, max_attempts=max_attempts,
                min_prefix_len=min_prefix_len, concurrency=concurrency,
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
            # Intermediate FRPs from refinement
            for step in ref["steps"]:
                if step["is_frp"]:
                    tracker.add_frp(step["prefix"], source="refine_intermediate",
                                    original=prefix)

        tracker.add_frp(refined_prefix, source=source, original=prefix)

        elapsed_ms = (time.time() - start) * 1000
        result["time_ms"] = elapsed_ms
        formatter.candidate_result(prefix, True, diag["reachable_count"],
                                   refined_prefix, elapsed_ms)
    else:
        # Step 3b: Tree alignment for partial reachability
        if do_ta and diag["reachable_count"] > 0:
            known_frp_set = set(tracker.get_prefixes())
            ta_result = tree_align_prefix(
                prefix, diag, known_frp_set,
                timeout=timeout, max_attempts=max_attempts,
                min_prefix_len=min_prefix_len, concurrency=concurrency,
                refine=do_refine, max_depth=max_ta_depth,
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
# Seed Preprocessing: APD + Refine + Upward Expansion + TA
# ============================================================================

def preprocess_seeds(
    seeds: List[str],
    tracker: GlobalFRPManager,
    cache: ProbeCache,
    formatter: OutputFormatter,
    timeout: float = DEFAULT_TIMEOUT,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    concurrency: int = DEFAULT_CONCURRENCY,
    min_prefix_len: int = DEFAULT_MIN_PREFIX_LEN,
    fast_refine: bool = False,
    upward_expand_bits: int = DEFAULT_UPWARD_EXPAND_BITS,
    do_ta: bool = True,
    max_ta_depth: int = DEFAULT_MAX_TA_DEPTH,
) -> Dict:
    """
    Seed preprocessing pipeline:
      1. For each seed: APD diagnose + refine to find shortest FRP
      2. Expand upward by k bits from refined prefix
      3. Diagnose expanded prefix for other reachable nibbles
      4. If partial reachability found outside seed → run tree alignment

    Returns:
        dict with preprocessing stats and newly discovered FRPs
    """
    formatter.section("SEED PREPROCESSING (APD + Refine + Upward Expand + TA)")
    logger.info(f"Preprocessing {len(seeds)} seeds "
                f"(upward_expand={upward_expand_bits} bits)...")

    stats = {
        "seeds_input": len(seeds),
        "seeds_confirmed_frp": 0,
        "seeds_not_frp": 0,
        "refinements_done": 0,
        "upward_expansions": 0,
        "ta_triggered": 0,
        "new_frps_from_preprocessing": 0,
        "total_probes": 0,
    }

    for i, seed in enumerate(seeds):
        seed = normalize_prefix(seed)
        logger.info(f"  [{i+1}/{len(seeds)}] Preprocessing seed: {seed}")

        # Step 1: Diagnose the seed itself
        diag = diagnose_prefix(
            seed, timeout=timeout, max_attempts=max_attempts,
            concurrency=concurrency,
        )
        cache.put(seed, diag)
        stats["total_probes"] += diag["total_probes_sent"]
        tracker.detection_probes += diag["total_probes_sent"]
        tracker.total_probes_sent += diag["total_probes_sent"]

        is_frp, _ = determine_frp_status(diag)

        if not is_frp:
            # Seed is not actually FRP — still add it but mark
            logger.warning(f"    Seed {seed} is NOT FRP ({diag['reachable_count']}/16)")
            stats["seeds_not_frp"] += 1
            # Try TA on it anyway
            if do_ta and diag["reachable_count"] > 0:
                known_set = set(tracker.get_prefixes())
                ta_res = tree_align_prefix(
                    seed, diag, known_set,
                    timeout=timeout, max_attempts=max_attempts,
                    min_prefix_len=min_prefix_len, concurrency=concurrency,
                    refine=True, max_depth=max_ta_depth, fast_refine=fast_refine,
                )
                stats["total_probes"] += ta_res["total_probes"]
                tracker.ta_probes += ta_res["total_probes"]
                tracker.total_probes_sent += ta_res["total_probes"]
                for nf in ta_res["new_frps"]:
                    if tracker.add_frp(nf["prefix"], source="seed_preprocess_ta",
                                       original=seed):
                        stats["new_frps_from_preprocessing"] += 1
                        tracker.ta_new_frps += 1
            continue

        stats["seeds_confirmed_frp"] += 1

        # Step 2: Refine to find shortest FRP
        ref = refine_frp_prefix(
            seed, timeout=timeout, max_attempts=max_attempts,
            min_prefix_len=min_prefix_len, concurrency=concurrency,
            fast_refine=fast_refine,
        )
        refined = ref["refined"]
        stats["total_probes"] += ref["total_refinement_probes"]
        tracker.refinement_probes += ref["total_refinement_probes"]
        tracker.total_probes_sent += ref["total_refinement_probes"]
        stats["refinements_done"] += 1

        # Add refined FRP
        tracker.add_frp(refined, source="seed_refined", original=seed)
        # Also add intermediate FRPs
        for step in ref["steps"]:
            if step["is_frp"]:
                tracker.add_frp(step["prefix"], source="seed_refine_intermediate",
                                original=seed)

        # Step 3: Expand upward by k bits
        refined_net = ipaddress.IPv6Network(refined, strict=False)
        refined_len = refined_net.prefixlen

        if refined_len - upward_expand_bits >= min_prefix_len:
            expanded_len = refined_len - upward_expand_bits
            expanded_prefix = f"{str(refined_net.network_address)}/{expanded_len}"
            expanded_prefix = normalize_prefix(expanded_prefix)

            logger.info(f"    Expanding upward: {refined} → {expanded_prefix}")
            stats["upward_expansions"] += 1

            # Diagnose the expanded prefix
            exp_diag = diagnose_prefix(
                expanded_prefix, timeout=timeout, max_attempts=max_attempts,
                concurrency=concurrency,
            )
            cache.put(expanded_prefix, exp_diag)
            stats["total_probes"] += exp_diag["total_probes_sent"]
            tracker.detection_probes += exp_diag["total_probes_sent"]
            tracker.total_probes_sent += exp_diag["total_probes_sent"]

            exp_is_frp, _ = determine_frp_status(exp_diag)

            if exp_is_frp:
                # The expanded prefix is also FRP! Even broader coverage
                tracker.add_frp(expanded_prefix, source="seed_upward_expand",
                                original=seed)
                stats["new_frps_from_preprocessing"] += 1
                logger.info(f"    ✓ Expanded {expanded_prefix} is also FRP!")
            elif exp_diag["reachable_count"] > 0:
                # Partial reachability — other nibbles outside the seed
                # Run tree alignment to find sibling FRPs
                logger.info(f"    Expanded {expanded_prefix}: "
                            f"{exp_diag['reachable_count']}/16 reachable, "
                            f"running TA...")
                stats["ta_triggered"] += 1

                if do_ta:
                    known_set = set(tracker.get_prefixes())
                    ta_res = tree_align_prefix(
                        expanded_prefix, exp_diag, known_set,
                        timeout=timeout, max_attempts=max_attempts,
                        min_prefix_len=min_prefix_len, concurrency=concurrency,
                        refine=True, max_depth=max_ta_depth,
                        fast_refine=fast_refine,
                    )
                    stats["total_probes"] += ta_res["total_probes"]
                    tracker.ta_probes += ta_res["total_probes"]
                    tracker.total_probes_sent += ta_res["total_probes"]

                    for nf in ta_res["new_frps"]:
                        if tracker.add_frp(nf["prefix"],
                                           source="seed_expand_ta",
                                           original=seed):
                            stats["new_frps_from_preprocessing"] += 1
                            tracker.ta_new_frps += 1
                            logger.info(f"      ↳ TA found: {nf['prefix']}")

            formatter.seed_preprocess(
                seed, refined, expanded_prefix,
                ta_frps=stats["new_frps_from_preprocessing"])
        else:
            formatter.seed_preprocess(seed, refined, None, ta_frps=0)

    logger.info(f"  Preprocessing complete: "
                f"{stats['seeds_confirmed_frp']}/{stats['seeds_input']} confirmed, "
                f"+{stats['new_frps_from_preprocessing']} new FRPs from expansion/TA")

    return stats


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
        ckpt = torch.load(gen_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            generator.load_state_dict(ckpt["model_state_dict"])
        else:
            generator.load_state_dict(ckpt)
        logger.info(f"  ✓ Generator loaded ({os.path.getsize(gen_path)/1e6:.1f}MB)")
    else:
        logger.warning(f"  ✗ Generator not found: {gen_path}")

    if os.path.exists(ver_path):
        verifier.load_state_dict(torch.load(ver_path, map_location=device))
        logger.info(f"  ✓ Verifier loaded ({os.path.getsize(ver_path)/1e6:.1f}MB)")
    else:
        logger.warning(f"  ✗ Verifier not found: {ver_path}")

    generator.eval()
    verifier.eval()

    gen_params = sum(p.numel() for p in generator.parameters())
    ver_params = sum(p.numel() for p in verifier.parameters())
    logger.info(f"  Params: generator={gen_params:,} verifier={ver_params:,}")

    return generator, verifier


# ============================================================================
# Candidate Generation (batch)
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
    per_bit_temperature: Optional[torch.Tensor] = None,
) -> List[str]:
    """
    Generate candidate prefixes using the generator (single batched pass).

    Args:
        per_bit_temperature: (128,) tensor of per-position temperature derived
            from entropy analysis. In mimic mode, low-entropy bits (e.g., bits 0-11)
            get very low temperature to preserve the network prefix structure,
            while high-entropy bits get normal/elevated temperature for variation.
    """
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

    bgp_bits_t = None
    bgp_len_t = None
    if bgp_prefix:
        bgp_bits_list, bgp_len = ipv6_prefix_to_bits(bgp_prefix)
        bgp_bits_t = torch.zeros((1, cfg.model.seq_len), dtype=torch.long, device=device)
        for i, b in enumerate(bgp_bits_list[:cfg.model.seq_len]):
            bgp_bits_t[0, i] = b
        bgp_len_t = torch.tensor([bgp_len], dtype=torch.long, device=device)

    target_plen_t = torch.tensor([target_plen], dtype=torch.long, device=device)

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
            per_bit_temperature=per_bit_temperature,
        )

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
        if bgp_prefix and not is_subnet_of(p, bgp_prefix):
            continue
        seen.add(p)
        candidates.append(p)

    return candidates


# ============================================================================
# Verifier Scoring (batch)
# ============================================================================

def score_candidates_batch(
    candidates: List[str],
    generator: UnifiedFRPGenerator,
    verifier: EnsembleVerifier,
    meta: Dict,
    cfg: Config,
    device: str = "cpu",
) -> List[Dict]:
    """Score candidates with ensemble verifier (single batched pass)."""
    if not candidates:
        return []

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
        condition = generator.condition_encoder(meta_batch)
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
    return sampler.select(scored, budget, already_probed=already_known or [])


# ============================================================================
# Online Feedback
# ============================================================================

def do_online_feedback(
    feedback: OnlineFeedback,
    probe_results: List[Dict],
    generator: UnifiedFRPGenerator,
    meta: Dict,
    device: str,
    formatter: OutputFormatter,
):
    """Report probe results to online feedback for model updates."""
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

    for r in probe_results:
        if r.get("skipped") or r.get("error"):
            continue
        is_frp = r["is_frp"] or len(r.get("ta_frps", [])) > 0
        feedback.report_probe(
            prefix=r["prefix"], is_frp=is_frp,
            condition=condition, meta=meta,
        )

    stats = feedback.get_stats()
    if stats["verifier_updates"] > 0 or stats["reptile_updates"] > 0:
        formatter.feedback_update(
            "model_update",
            f"ver={stats['verifier_updates']} rep={stats['reptile_updates']} "
            f"ft={stats['full_finetunes']}")


# ============================================================================
# Metadata helpers
# ============================================================================

def find_meta_for_bgp(bgp_prefix: str, meta_data: Dict,
                      default_meta: Dict) -> Dict:
    """
    Find metadata for a BGP prefix:
      1. Direct match
      2. From child prefixes (most common AS)
      3. From parent prefixes
      4. Default
    """
    bgp_prefix = normalize_prefix(bgp_prefix)

    # Direct match
    if bgp_prefix in meta_data:
        return meta_data[bgp_prefix]

    # From children
    bgp_net = ipaddress.IPv6Network(bgp_prefix, strict=False)
    bgp_len = bgp_net.prefixlen
    child_as_ids = []
    child_metas = []
    for prefix_str, meta_info in meta_data.items():
        try:
            p_net = ipaddress.IPv6Network(prefix_str, strict=False)
            if p_net.prefixlen > bgp_len and is_subnet_of(prefix_str, bgp_prefix):
                child_as_ids.append(meta_info.get("as_id", 0))
                child_metas.append(meta_info)
        except Exception:
            continue

    if child_as_ids:
        most_common_as = Counter(child_as_ids).most_common(1)[0][0]
        for m in child_metas:
            if m.get("as_id") == most_common_as:
                return m

    # From parents
    for prefix_str, meta_info in meta_data.items():
        try:
            p_net = ipaddress.IPv6Network(prefix_str, strict=False)
            if p_net.prefixlen < bgp_len and is_subnet_of(bgp_prefix, prefix_str):
                return meta_info
        except Exception:
            continue

    return default_meta


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
# Mode A: Seed Expansion (Enhanced)
# ============================================================================

def run_mode_seed_expand(args, cfg, generator, verifier):
    """
    Enhanced seed expansion:
      Phase 0: Seed preprocessing (APD + refine + upward expand + TA)
      Phase 1: Generate candidates conditioned on preprocessed seeds
      Phase 2: Score with verifier
      Phase 3: Active learning selection (adaptive budget)
      Phase 4: Probe each selected candidate
      Phase 5: Online feedback
    """
    device = cfg.device
    formatter = OutputFormatter(verbose=args.verbose)

    # Load seeds
    with open(args.seeds) as f:
        seeds = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    if not seeds:
        logger.error("No seeds found. Aborting.")
        sys.exit(1)

    # ─── Load metadata from prepared_new/meta.json ───
    meta_data = {}  # prefix -> {as_id, org_id, cat1_id, ...}
    if args.meta_file and os.path.exists(args.meta_file):
        with open(args.meta_file) as f:
            meta_data = json.load(f)
        logger.info(f"Loaded meta.json: {len(meta_data)} prefix entries")
    else:
        logger.warning(f"meta-file not found: {args.meta_file}")

    # Load AS seed DB for cross-AS reference
    as_seed_db = {}  # as_id(compact) -> [prefix, ...]
    if args.as_seed_db and os.path.exists(args.as_seed_db):
        with open(args.as_seed_db) as f:
            raw = json.load(f)
            as_seed_db = {int(k): v for k, v in raw.items()}
        logger.info(f"Loaded as_seed_db.json: {len(as_seed_db)} AS entries")

    # Build per-seed metadata lookup
    # Each seed gets its meta from meta.json (exact match or parent match)
    default_meta = parse_meta(args.meta)  # CLI fallback
    seed_meta_map = {}  # seed_prefix -> meta_dict
    for seed in seeds:
        seed_norm = normalize_prefix(seed)
        if seed_norm in meta_data:
            seed_meta_map[seed_norm] = meta_data[seed_norm]
        else:
            # Try to find a parent prefix in meta_data that covers this seed
            found = False
            for prefix_str, m_info in meta_data.items():
                try:
                    if is_subnet_of(seed_norm, prefix_str):
                        seed_meta_map[seed_norm] = m_info
                        found = True
                        break
                except Exception:
                    continue
            if not found:
                seed_meta_map[seed_norm] = default_meta

    # Group seeds by AS for metadata-guided generation
    as_to_seeds = defaultdict(list)  # as_id -> [seed_prefix, ...]
    for seed_norm, m in seed_meta_map.items():
        as_id = m.get("as_id", 0)
        as_to_seeds[as_id].append(seed_norm)

    # Use the most common AS's meta as the primary meta (for single-pass generation)
    # or iterate per-AS if multiple AS groups exist
    if as_to_seeds:
        primary_as_id = max(as_to_seeds, key=lambda k: len(as_to_seeds[k]))
        primary_meta = seed_meta_map.get(
            as_to_seeds[primary_as_id][0], default_meta)
    else:
        primary_as_id = 0
        primary_meta = default_meta

    logger.info(f"Seeds: {len(seeds)}, AS groups: {len(as_to_seeds)}, "
                f"Primary AS: {primary_as_id} ({len(as_to_seeds.get(primary_as_id, []))} seeds)")
    logger.info(f"Primary meta: as_id={primary_meta.get('as_id')}, "
                f"org_id={primary_meta.get('org_id')}, "
                f"geo_id={primary_meta.get('geo_id')}")

    # Determine target prefix length from seeds
    seed_plens = [ipv6_prefix_to_bits(s)[1] for s in seeds]
    # tgt_plen = int(sorted(seed_plens)[len(seed_plens) // 2]) if seed_plens else 64
    tgt_plen = args.target_plen if args.target_plen else int(sorted(seed_plens)[len(seed_plens) // 2])

    # Initialize components
    tracker = GlobalFRPManager()
    cache = ProbeCache()
    feedback = OnlineFeedback(generator, verifier, cfg, device=device)
    as_tracker = ASPerformanceTracker(
        base_candidates=args.candidates, base_budget=args.budget)

    config_display = {
        "seeds": len(seeds),
        "AS_groups": len(as_to_seeds),
        "primary_AS": f"{primary_as_id} ({len(as_to_seeds.get(primary_as_id, []))} seeds)",
        "meta_file": args.meta_file or "(none)",
        "as_seed_db": f"{len(as_seed_db)} AS" if as_seed_db else "(none)",
        "target_plen": f"/{tgt_plen}",
        "base_candidates": args.candidates,
        "base_budget": args.budget,
        "max_attempts": args.max_attempts,
        "concurrency": args.concurrency,
        "refine": not args.no_refine,
        "tree_align": not args.no_ta,
        "fast_refine": args.fast_refine,
        "upward_expand_bits": args.upward_expand,
        "seed_preprocessing": not args.no_preprocess,
        "device": device,
    }
    formatter.header("seed_expand (enhanced v3)", config_display)

    total_start = time.time()

    # ─── Phase 0: Seed Preprocessing ───
    preprocess_stats = {}
    if not args.no_preprocess:
        preprocess_stats = preprocess_seeds(
            seeds, tracker, cache, formatter,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            concurrency=args.concurrency,
            min_prefix_len=args.min_prefix_len,
            fast_refine=args.fast_refine,
            upward_expand_bits=args.upward_expand,
            do_ta=not args.no_ta,
            max_ta_depth=args.max_ta_depth,
        )
    else:
        # Just add seeds directly
        for s in seeds:
            tracker.add_frp(normalize_prefix(s), source="seed")

    preprocess_time = time.time() - total_start
    frps_after_preprocess = tracker.count()
    logger.info(f"After preprocessing: {frps_after_preprocess} FRPs "
                f"({preprocess_time:.1f}s)")

    # Use preprocessed FRPs as enhanced seeds for generation
    enhanced_seeds = tracker.get_prefixes()

    # ─── Entropy Analysis: compute per-bit temperature for mimic mode ───
    # Analyze the bit-level entropy of seeds to guide generation:
    # - Low entropy bits (e.g., bits 0-11) → low temperature (preserve structure)
    # - High entropy bits → normal/elevated temperature (allow variation)
    from utils.entropy_analysis import compute_bit_entropy, entropy_to_temperature

    entropy_result = compute_bit_entropy(seeds, max_bit_position=cfg.model.seq_len)
    per_bit_temp_list = entropy_to_temperature(
        entropy_result["entropy"],
        base_temperature=1.0,
        min_temperature=0.01,
        max_temperature=2.0,
        low_entropy_threshold=0.1,
        high_entropy_threshold=0.8,
        lock_below=0.05,
    )
    per_bit_temp_tensor = torch.tensor(per_bit_temp_list, dtype=torch.float32, device=device)

    # Log entropy summary for high-order bits (0-11)
    high_order_entropy = entropy_result["entropy"][:12]
    high_order_temps = per_bit_temp_list[:12]
    logger.info(f"[Entropy] Bits 0-11 avg entropy: {sum(high_order_entropy)/12:.4f}, "
                f"avg temperature: {sum(high_order_temps)/12:.4f}")
    logger.info(f"[Entropy] Bits 0-47 (coarse) avg entropy: "
                f"{sum(entropy_result['entropy'][:48])/48:.4f}")
    logger.info(f"[Entropy] Bits 48-63 (medium) avg entropy: "
                f"{sum(entropy_result['entropy'][48:64])/16:.4f}")
    logger.info(f"[Entropy] Bits 64-127 (fine) avg entropy: "
                f"{sum(entropy_result['entropy'][64:128])/64:.4f}")

    # Also compute per-AS entropy if multiple AS groups
    per_as_bit_temps = {}  # as_id -> tensor(128,)
    if len(as_to_seeds) > 1:
        for as_id_key, as_seeds_list in as_to_seeds.items():
            if len(as_seeds_list) >= 3:
                as_entropy = compute_bit_entropy(as_seeds_list, max_bit_position=cfg.model.seq_len)
                as_temp_list = entropy_to_temperature(
                    as_entropy["entropy"], base_temperature=1.0)
                per_as_bit_temps[as_id_key] = torch.tensor(
                    as_temp_list, dtype=torch.float32, device=device)
                logger.info(f"  AS {as_id_key}: {len(as_seeds_list)} seeds, "
                            f"bits 0-11 avg H={sum(as_entropy['entropy'][:12])/12:.4f}")

    # ─── Phase 1: Generate (per-AS metadata-guided) ───
    t0 = time.time()
    candidates = []

    if len(as_to_seeds) <= 1:
        # Single AS group: one-shot generation with primary meta
        as_id = primary_as_id
        adaptive_candidates = as_tracker.get_adaptive_candidates(
            as_id, known_prefix_count=len(enhanced_seeds))
        logger.info(f"[Phase 1] Generating {adaptive_candidates} candidates "
                    f"(single AS={as_id}, meta: org={primary_meta.get('org_id')}, "
                    f"geo={primary_meta.get('geo_id')})...")

        candidates = generate_candidates_batch(
            generator, cfg, primary_meta,
            seed_prefixes=enhanced_seeds[:cfg.model.max_seeds_per_batch],
            target_plen=tgt_plen,
            num_samples=adaptive_candidates,
            temperature=1.0,
            device=device,
            per_bit_temperature=per_bit_temp_tensor,
        )
    else:
        # Multiple AS groups: generate per-AS with respective metadata
        logger.info(f"[Phase 1] Generating candidates for {len(as_to_seeds)} AS groups...")
        for as_id, as_seeds_list in as_to_seeds.items():
            as_meta = seed_meta_map.get(as_seeds_list[0], default_meta)
            # Adaptive count based on AS size
            adaptive_n = as_tracker.get_adaptive_candidates(
                as_id, known_prefix_count=len(as_seeds_list))
            # Scale by proportion of seeds in this AS
            proportion = len(as_seeds_list) / max(len(seeds), 1)
            n_for_as = max(50, int(adaptive_n * proportion))

            # Use per-AS entropy temperature if available, else global
            as_pbt = per_as_bit_temps.get(as_id, per_bit_temp_tensor)

            logger.info(f"  AS {as_id}: {len(as_seeds_list)} seeds, "
                        f"generating {n_for_as} candidates "
                        f"(org={as_meta.get('org_id')}, geo={as_meta.get('geo_id')})")

            as_candidates = generate_candidates_batch(
                generator, cfg, as_meta,
                seed_prefixes=as_seeds_list[:cfg.model.max_seeds_per_batch],
                target_plen=tgt_plen,
                num_samples=n_for_as,
                temperature=1.0,
                device=device,
                per_bit_temperature=as_pbt,
            )
            candidates.extend(as_candidates)

    # Dedup and filter covered
    seen = set()
    unique_candidates = []
    for c in candidates:
        c_norm = normalize_prefix(c)
        if c_norm not in seen and not tracker.is_covered(c_norm):
            seen.add(c_norm)
            unique_candidates.append(c_norm)
    candidates = unique_candidates
    gen_time = time.time() - t0
    logger.info(f"  Generated {len(candidates)} unique candidates in {gen_time:.1f}s")

    if not candidates:
        logger.warning("No new candidates generated. Done.")
        save_output(args.output_dir, tracker, [], [], {
            "mode": "seed_expand", "note": "no candidates generated",
            "preprocess": preprocess_stats,
        })
        return

    # ─── Phase 2: Verifier scoring ───
    t0 = time.time()
    logger.info(f"[Phase 2] Scoring {len(candidates)} candidates...")
    scored = score_candidates_batch(
        candidates, generator, verifier, primary_meta, cfg, device)
    score_time = time.time() - t0
    avg_prob = sum(s['prob'] for s in scored) / len(scored)
    logger.info(f"  Scored in {score_time:.1f}s (avg prob={avg_prob:.3f})")

    # ─── Phase 3: Active learning selection ───
    t0 = time.time()
    adaptive_budget = as_tracker.get_adaptive_budget(primary_as_id)
    selected = select_candidates(
        scored, adaptive_budget, already_known=tracker.get_prefixes(), cfg=cfg)
    al_time = time.time() - t0
    logger.info(f"[Phase 3] Selected {len(selected)} candidates "
                f"(adaptive budget={adaptive_budget})")

    # ─── Phase 4: Probe + Align ───
    formatter.section(f"PROBING ({len(selected)} candidates)")
    formatter.probe_table_header()

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
            source="generated", **probe_kwargs,
        )
        probe_results.append(r)

    probe_time = time.time() - total_start - preprocess_time - gen_time - score_time - al_time

    # ─── Phase 5: Online feedback ───
    t0 = time.time()
    do_online_feedback(feedback, probe_results, generator, primary_meta, device, formatter)
    feedback_time = time.time() - t0

    # Update AS performance tracker
    direct_frps = sum(1 for r in probe_results if r["is_frp"])
    ta_frps_count = sum(len(r.get("ta_frps", [])) for r in probe_results)
    as_tracker.report(primary_as_id, len(selected), direct_frps + ta_frps_count)

    total_elapsed = time.time() - total_start

    # Sync FRP database to disk
    frp_file = os.path.join(args.output_dir, "final_frps.txt")
    tracker.sync_to_file(frp_file)

    # ─── Summary ───
    newly_found = tracker.count() - len(seeds)
    hit_rate = direct_frps / max(len(probe_results), 1)

    formatter.final_summary({
        "Seeds input": len(seeds),
        "After preprocessing": frps_after_preprocess,
        "Candidates generated": len(candidates),
        "Candidates probed": len(selected),
        "Direct FRPs (probing)": direct_frps,
        "TA FRPs (probing)": ta_frps_count,
        "Total new FRPs": newly_found,
        "Total FRPs (final)": tracker.count(),
        "Hit rate (direct)": f"{hit_rate:.1%}",
        "Entropy-guided generation": {
            "bits_0-11 avg entropy": f"{sum(entropy_result['entropy'][:12])/12:.4f}",
            "bits_0-11 avg temperature": f"{sum(per_bit_temp_list[:12])/12:.4f}",
            "coarse [0-47] avg entropy": f"{sum(entropy_result['entropy'][:48])/48:.4f}",
            "medium [48-63] avg entropy": f"{sum(entropy_result['entropy'][48:64])/16:.4f}",
            "per-AS temp profiles": len(per_as_bit_temps),
        },
        "Probes": {
            "preprocessing": preprocess_stats.get("total_probes", 0),
            "detection": tracker.detection_probes,
            "refinement": tracker.refinement_probes,
            "tree_align": tracker.ta_probes,
            "total": tracker.total_probes_sent,
        },
        "Cache": cache.stats(),
        "Timing": {
            "preprocessing": f"{preprocess_time:.1f}s",
            "generation": f"{gen_time:.1f}s",
            "scoring": f"{score_time:.1f}s",
            "active_learning": f"{al_time:.1f}s",
            "probing": f"{probe_time:.1f}s",
            "feedback": f"{feedback_time:.1f}s",
            "total": f"{total_elapsed:.1f}s",
        },
        "Feedback": feedback.get_stats(),
        "AS Performance": as_tracker.get_stats(),
    })

    save_output(args.output_dir, tracker, probe_results, scored, {
        "mode": "seed_expand",
        "config": config_display,
        "preprocess": preprocess_stats,
        "entropy_analysis": {
            "bits_0_11_avg_entropy": round(sum(entropy_result["entropy"][:12]) / 12, 4),
            "bits_0_11_avg_temperature": round(sum(per_bit_temp_list[:12]) / 12, 4),
            "coarse_avg_entropy": round(sum(entropy_result["entropy"][:48]) / 48, 4),
            "medium_avg_entropy": round(sum(entropy_result["entropy"][48:64]) / 16, 4),
            "fine_avg_entropy": round(sum(entropy_result["entropy"][64:128]) / 64, 4),
            "per_as_groups_computed": len(per_as_bit_temps),
        },
        "timing": {
            "preprocess_s": round(preprocess_time, 2),
            "generation_s": round(gen_time, 2),
            "scoring_s": round(score_time, 2),
            "al_s": round(al_time, 2),
            "probing_s": round(probe_time, 2),
            "feedback_s": round(feedback_time, 2),
            "total_s": round(total_elapsed, 2),
        },
        "results": {
            "seeds_input": len(seeds),
            "frps_after_preprocess": frps_after_preprocess,
            "candidates_generated": len(candidates),
            "candidates_probed": len(selected),
            "direct_frps": direct_frps,
            "ta_frps": ta_frps_count,
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
        "as_performance": as_tracker.get_stats(),
    })


# ============================================================================
# Mode B: Target Discovery (Enhanced)
# ============================================================================

def run_mode_target(args, cfg, generator, verifier):
    """
    Enhanced target discovery:
      1. Load existing seeds and meta data
      2. For each BGP prefix:
         a. Discover existing seeds within BGP → add to FRP set
         b. If seeds found → run seed_expand sub-pipeline on them
         c. Generate new candidates (mimic if seeds, innovate otherwise)
         d. Cross-domain transfer if no seeds and no meta match
         e. Score + select + probe + feedback
      3. Adaptive budget per AS based on historical performance
    """
    device = cfg.device
    formatter = OutputFormatter(verbose=args.verbose)

    # Load targets
    with open(args.targets) as f:
        targets = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    if not targets:
        logger.error("No targets found. Aborting.")
        sys.exit(1)

    default_meta = parse_meta(args.meta)

    # Load AS seed DB
    as_seed_db = None
    if args.as_seed_db and os.path.exists(args.as_seed_db):
        with open(args.as_seed_db) as f:
            raw = json.load(f)
            as_seed_db = {int(k): v for k, v in raw.items()}
        logger.info(f"Loaded AS seed DB: {len(as_seed_db)} entries")

    # Load existing seeds
    existing_seeds = set()
    if args.seeds and os.path.exists(args.seeds):
        with open(args.seeds) as f:
            existing_seeds = {normalize_prefix(l.strip())
                              for l in f if l.strip() and not l.startswith('#')}
        logger.info(f"Loaded {len(existing_seeds)} existing seeds")

    # Load meta.json
    meta_data = {}
    if args.meta_file and os.path.exists(args.meta_file):
        with open(args.meta_file) as f:
            meta_data = json.load(f)
        logger.info(f"Loaded meta data: {len(meta_data)} prefixes")

    # Initialize
    tracker = GlobalFRPManager()
    cache = ProbeCache()
    feedback = OnlineFeedback(generator, verifier, cfg, device=device)
    as_tracker = ASPerformanceTracker(
        base_candidates=args.candidates, base_budget=args.budget)

    # Add existing seeds to tracker
    for seed in existing_seeds:
        tracker.add_frp(seed, source="existing_seed")

    config_display = {
        "targets": len(targets),
        "existing_seeds": len(existing_seeds),
        "target_plen": f"/{args.target_plen}",
        "base_candidates/BGP": args.candidates,
        "base_budget/BGP": args.budget,
        "max_attempts": args.max_attempts,
        "concurrency": args.concurrency,
        "refine": not args.no_refine,
        "tree_align": not args.no_ta,
        "fast_refine": args.fast_refine,
        "seed_preprocessing": not args.no_preprocess,
        "device": device,
    }
    formatter.header("target_discovery (enhanced v3)", config_display)

    total_start = time.time()
    all_probe_results = []
    all_scored = []
    per_bgp_stats = []

    for bgp_idx, bgp_prefix in enumerate(targets):
        bgp_prefix = normalize_prefix(bgp_prefix)
        bgp_bits, bgp_len = ipv6_prefix_to_bits(bgp_prefix)

        # Find metadata for this BGP prefix
        bgp_meta = find_meta_for_bgp(bgp_prefix, meta_data, default_meta)
        as_id = int(bgp_meta.get("as_id", default_meta.get("as_id", 0)))

        # ─── Step 1: Discover existing seeds within this BGP ───
        seeds_in_bgp = []
        for seed in existing_seeds:
            if is_subnet_of(seed, bgp_prefix):
                seeds_in_bgp.append(seed)

        # Also check tracker for FRPs already in this BGP
        frps_in_bgp = tracker.get_frps_in_bgp(bgp_prefix)

        # Determine mode
        all_local_seeds = list(set(seeds_in_bgp + frps_in_bgp))
        as_seeds = []
        if as_seed_db and as_id in as_seed_db:
            as_seeds = as_seed_db[as_id][:cfg.model.max_seeds_per_batch]

        if all_local_seeds:
            mode_str = "seed_reuse"
            gen_seeds = all_local_seeds[:cfg.model.max_seeds_per_batch]
            temp = 1.0
        elif as_seeds:
            mode_str = "mimic"
            gen_seeds = as_seeds
            temp = 1.0
        else:
            mode_str = "innovate"
            gen_seeds = None
            temp = 1.3

        # Adaptive budget
        adaptive_candidates = as_tracker.get_adaptive_candidates(
            as_id, known_prefix_count=len(as_seeds) + len(all_local_seeds))
        adaptive_budget = as_tracker.get_adaptive_budget(as_id)

        print(f"\n{'─' * 78}")
        print(f"  BGP [{bgp_idx+1}/{len(targets)}]: {bgp_prefix}")
        print(f"  AS={as_id}, mode={mode_str}, "
              f"local_seeds={len(all_local_seeds)}, as_seeds={len(as_seeds)}")
        print(f"  Adaptive: candidates={adaptive_candidates}, budget={adaptive_budget}")
        print(f"{'─' * 78}")

        bgp_start = time.time()

        # ─── Step 2: Seed preprocessing for local seeds ───
        if all_local_seeds and not args.no_preprocess:
            logger.info(f"  Running seed preprocessing on {len(all_local_seeds)} "
                        f"local seeds...")
            preprocess_seeds(
                all_local_seeds[:8], tracker, cache, formatter,
                timeout=args.timeout,
                max_attempts=args.max_attempts,
                concurrency=args.concurrency,
                min_prefix_len=max(bgp_len, args.min_prefix_len),
                fast_refine=args.fast_refine,
                upward_expand_bits=min(args.upward_expand,
                                       all_local_seeds[0] and
                                       ipv6_prefix_to_bits(all_local_seeds[0])[1] - bgp_len
                                       if all_local_seeds else args.upward_expand),
                do_ta=not args.no_ta,
                max_ta_depth=args.max_ta_depth,
            )

        # ─── Step 3: Generate candidates ───
        candidates = generate_candidates_batch(
            generator, cfg, bgp_meta,
            seed_prefixes=gen_seeds,
            bgp_prefix=bgp_prefix,
            target_plen=args.target_plen,
            num_samples=adaptive_candidates,
            temperature=temp,
            device=device,
        )
        candidates = [c for c in candidates if not tracker.is_covered(c)]

        if not candidates:
            logger.info(f"  No new candidates for {bgp_prefix}")
            per_bgp_stats.append({
                "bgp": bgp_prefix, "mode": mode_str, "as_id": as_id,
                "generated": 0, "probed": 0, "frps": 0,
            })
            continue

        # ─── Step 4: Score ───
        scored = score_candidates_batch(
            candidates, generator, verifier, bgp_meta, cfg, device)
        all_scored.extend(scored)

        # ─── Step 5: Select ───
        selected = select_candidates(
            scored, adaptive_budget,
            already_known=tracker.get_prefixes(), cfg=cfg)

        logger.info(f"  Generated {len(candidates)}, "
                    f"selected {len(selected)} for probing")

        # ─── Step 6: Probe ───
        formatter.probe_table_header()
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
                source=f"target_{mode_str}", **probe_kwargs,
            )
            bgp_results.append(r)
            all_probe_results.append(r)

        # ─── Step 7: Online feedback ───
        do_online_feedback(feedback, bgp_results, generator, bgp_meta,
                           device, formatter)

        # Update AS tracker
        bgp_elapsed = time.time() - bgp_start
        bgp_frps = sum(1 for r in bgp_results if r["is_frp"])
        bgp_ta = sum(len(r.get("ta_frps", [])) for r in bgp_results)
        as_tracker.report(as_id, len(selected), bgp_frps + bgp_ta)

        per_bgp_stats.append({
            "bgp": bgp_prefix, "mode": mode_str, "as_id": as_id,
            "generated": len(candidates), "probed": len(selected),
            "frps": bgp_frps + bgp_ta, "time_s": round(bgp_elapsed, 1),
            "local_seeds": len(all_local_seeds),
        })

        print(f"  → {bgp_prefix}: +{bgp_frps} direct, +{bgp_ta} TA "
              f"({bgp_elapsed:.1f}s)")

        # Real-time sync
        frp_file = os.path.join(args.output_dir, "final_frps.txt")
        tracker.sync_to_file(frp_file)

    total_elapsed = time.time() - total_start

    # Final summary
    total_frps = tracker.count()
    total_probed = len(all_probe_results)
    direct_frps = sum(1 for r in all_probe_results if r["is_frp"])

    formatter.final_summary({
        "BGP prefixes": f"{len(targets)} input",
        "Existing seeds loaded": len(existing_seeds),
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
        "AS Performance": {
            k: f"probed={v['total_probed']}, frps={v['total_frps']}, "
               f"hit={v['hit_rate']:.1%}"
            for k, v in as_tracker.get_stats().items()
        },
    })

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
        "as_performance": as_tracker.get_stats(),
    })


# ============================================================================
# Output Saving
# ============================================================================

def save_output(output_dir: str, tracker: GlobalFRPManager,
                probe_results: List[Dict], scored: List[Dict],
                summary: Dict):
    """Save all results to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # FRP list
    frp_path = os.path.join(output_dir, "final_frps.txt")
    tracker.sync_to_file(frp_path)

    # Probe results
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

    # FRP detail (provenance)
    detail_path = os.path.join(output_dir, "frps_detail.json")
    with open(detail_path, 'w') as f:
        json.dump(tracker.confirmed, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\nResults saved to {output_dir}/")
    logger.info(f"  final_frps.txt        : {tracker.count()} FRPs")
    logger.info(f"  probed.json           : {len(probe_results)} probe records")
    logger.info(f"  candidates_scored.json: {len(scored)} scored candidates")
    logger.info(f"  summary.json          : run statistics")
    logger.info(f"  frps_detail.json      : FRP provenance details")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="6FRP-AR v3 Enhanced Standalone-APD Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Seed expansion with full preprocessing
  sudo python3 infer_standalone.py --mode seed_expand \\
      --seeds data/seeds.txt --checkpoint-dir output/models

  # Target discovery with seed reuse
  sudo python3 infer_standalone.py --mode target \\
      --targets data/bgp.txt --seeds data/seeds.txt \\
      --as-seed-db data/as_seeds.json --meta-file output/prepared_new/meta.json

  # Fast mode (skip preprocessing, fewer attempts)
  sudo python3 infer_standalone.py --mode seed_expand \\
      --seeds data/seeds.txt --no-preprocess --max-attempts 3 --fast-refine

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
                        help='Base candidates to generate per round')
    parser.add_argument('--budget', type=int, default=200,
                        help='Base active learning budget per round')

    # Probing
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help=f'Per-attempt timeout (default: {DEFAULT_TIMEOUT}s)')
    parser.add_argument('--max-attempts', type=int, default=DEFAULT_MAX_ATTEMPTS,
                        help=f'Max ping attempts per address (default: {DEFAULT_MAX_ATTEMPTS})')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Parallel probes within prefix (default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('--min-prefix-len', type=int, default=DEFAULT_MIN_PREFIX_LEN,
                        help=f'Min prefix length for refinement (default: {DEFAULT_MIN_PREFIX_LEN})')

    # Alignment & Refinement
    parser.add_argument('--no-refine', action='store_true',
                        help='Disable refinement')
    parser.add_argument('--no-ta', action='store_true',
                        help='Disable tree alignment')
    parser.add_argument('--fast-refine', action='store_true',
                        help='Use fast 4-bit-then-1-bit refinement')
    parser.add_argument('--max-ta-depth', type=int, default=DEFAULT_MAX_TA_DEPTH,
                        help=f'Max TA recursion depth (default: {DEFAULT_MAX_TA_DEPTH})')

    # Seed preprocessing
    parser.add_argument('--no-preprocess', action='store_true',
                        help='Skip seed preprocessing (APD+refine+expand)')
    parser.add_argument('--upward-expand', type=int, default=DEFAULT_UPWARD_EXPAND_BITS,
                        help=f'Bits to expand upward from refined seed (default: {DEFAULT_UPWARD_EXPAND_BITS})')

    # Mode A: seed_expand
    parser.add_argument('--seeds', default=None,
                        help='File with seed FRP prefixes')

    # Mode B: target
    parser.add_argument('--targets', default=None,
                        help='File with BGP prefixes')
    parser.add_argument('--target-plen', type=int, default=64,
                        help='Target prefix length (default: 64)')

    # Shared data files (used by both seed_expand and target)
    parser.add_argument('--meta-file',
                        default='output/prepared_new/meta.json',
                        help='Path to meta.json (prefix -> metadata). '
                             'Used to look up AS/org/geo info per seed.')
    parser.add_argument('--as-seed-db',
                        default='output/prepared_new/as_seed_db.json',
                        help='JSON: AS ID -> seed FRP list. '
                             'Used for metadata-guided generation.')

    # Shared
    parser.add_argument('--meta', default=None,
                        help='Default metadata: JSON or key=val,key=val')
    parser.add_argument('--simulate', default=None,
                        help='(Reserved) Simulate probing for testing')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Root check
    if not args.simulate:
        if hasattr(os, "geteuid") and os.geteuid() != 0:
            logger.error("Live probing requires root. Use: sudo python3 infer_standalone.py ...")
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
