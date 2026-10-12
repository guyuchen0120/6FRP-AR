#!/usr/bin/env python3
"""
Online Inference for DualScanHMamba FRP Generator.

This script implements the online inference loop:
  1. Generate candidate prefixes using the trained model.
  2. Filter candidates against the aligned seed set (filtered_seeds.txt)
     to avoid generating any sub-prefix of existing seeds.
  3. Run APD (Active Probing Detection) on candidates.
  4. Judge FRP status based on probe results.
  5. Fine-tune the model based on feedback (confirmed FRPs / non-FRPs).
  6. Iterate: generate more candidates with updated model.

Usage:
    # Basic online inference (requires root for raw sockets)
    sudo python3 online_infer_dual_scan.py \\
        --checkpoint checkpoints/dual_scan/best_model.pt \\
        --data-dir data_fixed \\
        --output-dir results/online_run1

    # With custom parameters
    sudo python3 online_infer_dual_scan.py \\
        --checkpoint checkpoints/dual_scan/best_model.pt \\
        --data-dir data_fixed \\
        --output-dir results/online_run1 \\
        --rounds 10 --candidates-per-round 50 --temperature 0.8
"""
import argparse
import gc
import ipaddress
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

# Add project root to path
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from generators.dual_scan_hmamba import (
    DualScanHMamba, DualScanConfig, MASK_ID
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Pretty-print helpers
# ============================================================================

def _banner(title: str, width: int = 78, char: str = "=") -> str:
    line = char * width
    return f"\n{line}\n{title.center(width)}\n{line}"


def _sub_banner(title: str, width: int = 70, char: str = "-") -> str:
    return f"{char * 3} {title} " + char * max(0, width - 5 - len(title))


# ============================================================================
# Known-FRP Set: hot set of FRPs used to short-circuit probing
# ============================================================================

class KnownFRPSet:
    """
    Maintains a growing set of confirmed FRP prefixes.

    Responsibilities
    ----------------
    * Load an initial known-FRP file (if provided) into memory.
    * Given a candidate prefix, decide whether it is a *sub-prefix* of
      any already-known FRP.  If so, we can trust the parent FRP's
      status and skip active probing entirely (this is the ``auto-
      confirmed`` path).
    * Persist newly-confirmed FRPs back to the on-disk file so runs are
      cumulative across sessions.
    """
    def __init__(self, known_frps_path: str):
        self.path = known_frps_path
        self.networks: List[ipaddress.IPv6Network] = []
        self.prefix_set: Set[str] = set()
        self._load()

    def _load(self):
        if self.path and os.path.exists(self.path):
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        net = ipaddress.IPv6Network(line, strict=False)
                    except Exception:
                        continue
                    key = str(net)
                    if key not in self.prefix_set:
                        self.networks.append(net)
                        self.prefix_set.add(key)
            logger.info(f"KnownFRPSet loaded {len(self.networks)} FRPs "
                        f"from {self.path}")
        else:
            logger.info(f"KnownFRPSet: no existing file at {self.path}, "
                        f"starting from empty set")

    def __len__(self) -> int:
        return len(self.networks)

    def contains_prefix(self, prefix_str: str) -> bool:
        """Exact-string check."""
        try:
            net = ipaddress.IPv6Network(prefix_str, strict=False)
        except Exception:
            return False
        return str(net) in self.prefix_set

    def covering_frp(self, candidate_str: str) -> Optional[str]:
        """
        Return the known-FRP prefix that *covers* the candidate
        (i.e. candidate is a sub-prefix of that known FRP), or None.

        Being covered by a known FRP means we can immediately mark the
        candidate as an FRP without any active probing.
        """
        try:
            cand = ipaddress.IPv6Network(candidate_str, strict=False)
        except Exception:
            return None
        for net in self.networks:
            if cand.subnet_of(net):
                return str(net)
        return None

    def add(self, prefix_str: str, persist: bool = True) -> bool:
        """Add a newly-confirmed FRP.  Returns True if it was new."""
        try:
            net = ipaddress.IPv6Network(prefix_str, strict=False)
        except Exception:
            return False
        key = str(net)
        if key in self.prefix_set:
            return False
        self.networks.append(net)
        self.prefix_set.add(key)
        if persist and self.path:
            try:
                with open(self.path, 'a') as f:
                    f.write(f"{key}\n")
            except Exception as e:
                logger.warning(f"KnownFRPSet: could not persist {key}: {e}")
        return True

    def add_many(self, prefixes: List[str], persist: bool = True) -> int:
        added = 0
        for p in prefixes:
            if self.add(p, persist=persist):
                added += 1
        return added

    def snapshot_to(self, out_path: str) -> int:
        """Write the full current set to `out_path` (overwriting)."""
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.',
                    exist_ok=True)
        with open(out_path, 'w') as f:
            for key in self.prefix_set:
                f.write(f"{key}\n")
        return len(self.prefix_set)


# ============================================================================
# Seed Filter: avoid generating sub-prefixes of aligned seeds
# ============================================================================

class SeedFilter:
    """
    Filters candidate prefixes against the aligned seed set.

    Two-level filtering:
      Level 1 – Exact duplicate: candidate == a known seed → reject outright
        (no exploration value, already known).
      Level 2 – Inside known range: candidate is a subnet of a seed
        → silently discard without APD and without negative-sample feedback
        (strategy C: count for temperature adaptation only).

    NOTE: A candidate that *covers* a seed (i.e. is a broader prefix) is
    intentionally NOT filtered here.  It goes to APD normally:
      - APD fails  → treated as a non-FRP negative sample (correct signal)
      - APD passes → treated as a new confirmed FRP (correct signal)
    """
    def __init__(self, filtered_seeds_path: str):
        self.seed_networks: List[ipaddress.IPv6Network] = []
        self.seed_set: Set[str] = set()

        if os.path.exists(filtered_seeds_path):
            with open(filtered_seeds_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        try:
                            net = ipaddress.IPv6Network(line, strict=False)
                            self.seed_networks.append(net)
                            self.seed_set.add(str(net))
                        except Exception:
                            continue
            logger.info(f"SeedFilter loaded {len(self.seed_networks)} aligned seeds")
        else:
            logger.warning(f"Filtered seeds file not found: {filtered_seeds_path}")

    def is_duplicate_seed(self, candidate_str: str) -> bool:
        """
        Level 1: Check if candidate is exactly equal to a known seed.

        Only exact duplicates are rejected outright – there is no point
        re-probing a prefix we already know about.
        Candidates that are *broader* than a seed (i.e. cover it) are
        allowed through to APD so they can produce a proper feedback signal.
        """
        try:
            candidate = ipaddress.IPv6Network(candidate_str, strict=False)
        except Exception:
            return True  # Invalid -> reject

        return str(candidate) in self.seed_set

    # Keep old name as an alias so any external callers are not broken.
    def is_sub_prefix_of_seed(self, candidate_str: str) -> bool:
        """Alias for is_duplicate_seed (kept for backward compatibility)."""
        return self.is_duplicate_seed(candidate_str)

    def is_subnet_of_any_seed(self, candidate_str: str) -> bool:
        """
        Direction 2: Check if candidate falls inside an already-known seed range.

        A candidate C is "in known range" if any seed S satisfies:
          C.subnet_of(S)  →  C is a more-specific prefix inside S.

        These candidates are NOT rejected outright (they may be real FRPs),
        but they are silently discarded without APD probing and without
        being used as negative samples.  Their count is tracked to drive
        temperature adaptation (strategy C).
        """
        try:
            candidate = ipaddress.IPv6Network(candidate_str, strict=False)
        except Exception:
            return False

        for seed_net in self.seed_networks:
            if candidate.subnet_of(seed_net):
                return True

        return False

    def filter_candidates(self, candidates: List[str]) -> List[str]:
        """Level-1 filter only: remove exact duplicate seeds."""
        filtered = []
        rejected = 0
        for c in candidates:
            if not self.is_duplicate_seed(c):
                filtered.append(c)
            else:
                rejected += 1
        if rejected > 0:
            logger.info(f"  SeedFilter: rejected {rejected}/{len(candidates)} "
                        f"candidates (exact duplicate seeds)")
        return filtered

    def filter_candidates_bidirectional(
            self, candidates: List[str]
    ) -> Tuple[List[str], int, int]:
        """
        Two-level filter.

        Returns:
            truly_new      – candidates that pass both levels (go to APD)
            dup_count      – rejected because candidate == a known seed (level 1)
            known_range_count – silently discarded because candidate is a
                               subnet of a known seed (level 2, strategy C)
        """
        truly_new: List[str] = []
        dup_count = 0
        known_range_count = 0

        for c in candidates:
            if self.is_duplicate_seed(c):
                dup_count += 1          # exact duplicate → reject
            elif self.is_subnet_of_any_seed(c):
                known_range_count += 1  # inside known range → silently discard
            else:
                truly_new.append(c)     # genuinely new → send to APD

        total = len(candidates)
        if dup_count > 0:
            logger.info(f"  SeedFilter [L1]: rejected {dup_count}/{total} "
                        f"candidates (exact duplicate seeds)")
        if known_range_count > 0:
            logger.info(f"  SeedFilter [L2]: discarded {known_range_count}/{total} "
                        f"candidates (inside known seed range, strategy C)")

        return truly_new, dup_count, known_range_count


# ============================================================================
# Bit <-> Prefix conversion utilities
# ============================================================================

def prefix_to_bits(prefix_str: str, seq_len: int = 128) -> List[int]:
    """Convert IPv6 prefix to bit list."""
    try:
        net = ipaddress.IPv6Network(prefix_str, strict=False)
    except Exception:
        return [0] * seq_len
    addr_int = int(net.network_address)
    bits = []
    for i in range(seq_len):
        bits.append((addr_int >> (127 - i)) & 1)
    return bits


def bits_to_prefix(bits: List[int], prefix_len: int) -> str:
    """Convert bit list to IPv6 prefix string."""
    addr_int = 0
    for i, b in enumerate(bits[:128]):
        addr_int = (addr_int << 1) | (b & 1)
    # Pad remaining bits with 0
    if len(bits) < 128:
        addr_int <<= (128 - len(bits))
    # Mask to prefix_len
    if prefix_len < 128:
        mask = ((1 << prefix_len) - 1) << (128 - prefix_len)
        addr_int &= mask
    addr = ipaddress.IPv6Address(addr_int)
    return f"{addr}/{prefix_len}"


# ============================================================================
# APD Integration
# ============================================================================

def run_apd_on_candidates(candidates: List[str],
                          timeout: float = 2.0,
                          max_attempts: int = 5,
                          concurrency: int = 16,
                          refine: bool = True,
                          workers: int = 8,
                          max_probes: Optional[int] = None,
                          quiet: bool = False) -> Dict:
    """
    Run APD detection on candidate prefixes.

    Uses the standalone_apd pipeline for probing (refine mode).

    Returns:
        dict with 'confirmed_frps', 'non_frps', 'stats', 'probes_used',
        'unprocessed'
    """
    try:
        from standalone_apd import run_apd_pipeline
    except ImportError:
        from utils.apd_core import (
            diagnose_prefix, determine_frp_status, refine_frp_prefix,
            has_raw_socket_capability
        )
        # Fallback: manual probing
        return _manual_apd(candidates, timeout, max_attempts, concurrency, refine)

    result = run_apd_pipeline(
        candidates,
        refine=refine,
        tree_align=False,
        timeout=timeout,
        max_attempts=max_attempts,
        concurrency=concurrency,
        workers=workers,
        max_probes=max_probes,
        quiet=quiet,
    )

    confirmed = result.get("confirmed_frps", [])
    all_results = result.get("all_results", [])

    non_frps = []
    for r in all_results:
        d = r.get("frp_decision", {})
        if not d.get("is_frp", False):
            non_frps.append(r.get("prefix", ""))

    stats = result.get("stats", {})
    probes_used = int(stats.get("total_probes_sent", 0)
                      or stats.get("total_probes", 0)
                      or 0)

    return {
        "confirmed_frps": confirmed,
        "non_frps": non_frps,
        "stats": stats,
        "probes_used": probes_used,
        "unprocessed": result.get("unprocessed", []),
    }


def _manual_apd(candidates, timeout, max_attempts, concurrency, refine):
    """Fallback APD using apd_core directly."""
    from utils.apd_core import (
        diagnose_prefix, determine_frp_status, refine_frp_prefix,
        has_raw_socket_capability
    )

    if not has_raw_socket_capability():
        logger.error("Need root/CAP_NET_RAW for APD probing")
        return {"confirmed_frps": [], "non_frps": candidates,
                "stats": {}, "probes_used": 0, "unprocessed": []}

    confirmed_frps = []
    non_frps = []

    for prefix in candidates:
        diag = diagnose_prefix(prefix, timeout=timeout,
                               max_attempts=max_attempts,
                               concurrency=concurrency)
        is_frp, reason = determine_frp_status(diag)

        if is_frp:
            if refine:
                ref = refine_frp_prefix(prefix, timeout=timeout,
                                        max_attempts=max_attempts,
                                        concurrency=concurrency)
                confirmed_frps.append(ref["refined"])
            else:
                confirmed_frps.append(prefix)
        else:
            non_frps.append(prefix)

    return {
        "confirmed_frps": confirmed_frps,
        "non_frps": non_frps,
        "stats": {"processed": len(candidates)},
        "probes_used": 0,
        "unprocessed": [],
    }


# ============================================================================
# Candidate Generation
# ============================================================================

def generate_candidates(model: DualScanHMamba,
                        meta_db: Dict,
                        as_seed_db: Dict,
                        seed_filter: SeedFilter,
                        num_candidates: int,
                        temperature: float = 0.8,
                        device: torch.device = torch.device("cpu"),
                        target_prefix_lens: Optional[List[int]] = None,
                        known_frps: Optional["KnownFRPSet"] = None,
                        ) -> Tuple[List[str], List[str], int, int]:
    """
    Generate candidate prefixes using the model.

    Strategy:
      - Sample AS ids proportional to their seed count.
      - For each AS, pick a random seed as BGP constraint.
      - Generate new prefixes using staged inference.
      - Bidirectional filter against aligned seed set (strategy C).
      - If a KnownFRPSet is supplied, any candidate that is a sub-prefix
        of an already-known FRP is auto-confirmed without probing.

    Returns:
        to_probe          -- candidates that need active APD probing
        auto_confirmed    -- candidates that are sub-prefixes of known FRPs
                             (skip APD, treated as positive feedback)
        known_range_count -- silently discarded (inside a known seed range, dir2)
        total_before_filter -- total raw candidates before any filtering
    """
    model.eval()

    # Sample AS ids weighted by seed count
    as_ids = list(as_seed_db.keys())
    as_weights = [len(as_seed_db[a]) for a in as_ids]
    total_w = sum(as_weights)
    as_probs = [w / total_w for w in as_weights]

    candidates = []
    attempts = 0
    max_attempts = num_candidates * 5  # avoid infinite loop

    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1
        batch_size = min(num_candidates - len(candidates), 32)

        # Sample AS for this batch
        chosen_as_ids = random.choices(as_ids, weights=as_probs, k=batch_size)

        # Build batch metadata
        batch_meta = defaultdict(list)
        bgp_bits_list = []
        bgp_len_list = []
        plen_list = []

        for as_id_str in chosen_as_ids:
            # Pick a random seed from this AS
            seeds = as_seed_db[as_id_str]
            seed_prefix = random.choice(seeds)

            if seed_prefix not in meta_db:
                # Fallback: use first available seed
                seed_prefix = seeds[0] if seeds[0] in meta_db else None
                if seed_prefix is None:
                    continue

            meta = meta_db[seed_prefix]
            batch_meta["as_id"].append(meta["as_id"])
            batch_meta["org_id"].append(meta["org_id"])
            batch_meta["cat1_id"].append(meta["cat1_id"])
            batch_meta["cat2_id"].append(meta["cat2_id"])
            batch_meta["proto_id"].append(meta["proto_id"])
            batch_meta["geo_id"].append(meta["geo_id"])
            batch_meta["rtt_stats"].append(meta["rtt_stats"])

            # BGP constraint: use the seed's network bits up to some length
            try:
                net = ipaddress.IPv6Network(seed_prefix, strict=False)
                seed_bits = prefix_to_bits(seed_prefix)
                # Use a portion of the seed as BGP constraint
                # (e.g., first 24-32 bits as the BGP routing prefix)
                bgp_len = min(net.prefixlen, 32)
                bgp_bits_list.append(seed_bits)
                bgp_len_list.append(bgp_len)

                # Target prefix length
                if target_prefix_lens:
                    plen = random.choice(target_prefix_lens)
                else:
                    # Generate prefixes of similar length to seeds
                    plen = random.choice([32, 40, 48, 56, 64])
                plen_list.append(plen)
            except Exception:
                bgp_bits_list.append([0] * 128)
                bgp_len_list.append(0)
                plen_list.append(48)

        if not batch_meta["as_id"]:
            continue

        # Convert to tensors
        actual_batch = len(batch_meta["as_id"])
        meta_tensors = {
            "as_id": torch.tensor(batch_meta["as_id"], dtype=torch.long, device=device),
            "org_id": torch.tensor(batch_meta["org_id"], dtype=torch.long, device=device),
            "cat1_id": torch.tensor(batch_meta["cat1_id"], dtype=torch.long, device=device),
            "cat2_id": torch.tensor(batch_meta["cat2_id"], dtype=torch.long, device=device),
            "proto_id": torch.tensor(batch_meta["proto_id"], dtype=torch.long, device=device),
            "geo_id": torch.tensor(batch_meta["geo_id"], dtype=torch.long, device=device),
            "rtt_stats": torch.tensor(batch_meta["rtt_stats"], dtype=torch.float32, device=device),
        }

        bgp_bits_t = torch.tensor(bgp_bits_list[:actual_batch], dtype=torch.long, device=device)
        bgp_len_t = torch.tensor(bgp_len_list[:actual_batch], dtype=torch.long, device=device)
        plen_t = torch.tensor(plen_list[:actual_batch], dtype=torch.long, device=device)

        # Generate
        with torch.no_grad():
            generated_bits = model.generate_staged(
                meta=meta_tensors,
                target_prefix_lens=plen_t,
                bgp_constraint_bits=bgp_bits_t,
                bgp_constraint_len=bgp_len_t,
                temperature=temperature,
                num_samples=1,
            )

        # Convert to prefix strings
        for i in range(generated_bits.size(0)):
            bits = generated_bits[i].cpu().tolist()
            plen = plen_list[i] if i < len(plen_list) else 48
            prefix_str = bits_to_prefix(bits, plen)
            candidates.append(prefix_str)

    # Deduplicate
    candidates = list(dict.fromkeys(candidates))

    # Bidirectional filter against aligned seeds (strategy C)
    # dir1: candidate covers a seed → reject
    # dir2: candidate is inside a known seed range → silently discard
    candidates, _dir1, _dir2 = seed_filter.filter_candidates_bidirectional(candidates)

    total_before = len(candidates) + _dir1 + _dir2
    candidates = candidates[:num_candidates]

    # Split by known-FRP coverage: sub-prefixes of known FRPs are auto-
    # confirmed and never probed.
    auto_confirmed: List[str] = []
    to_probe: List[str] = []
    if known_frps is not None and len(known_frps) > 0:
        for c in candidates:
            cov = known_frps.covering_frp(c)
            if cov is not None:
                auto_confirmed.append(c)
            else:
                to_probe.append(c)
    else:
        to_probe = list(candidates)

    return to_probe, auto_confirmed, _dir2, total_before


# ============================================================================
# Online Fine-tuning
# ============================================================================

# Maximum samples per mini-batch during fine-tuning to cap memory usage.
_FT_MINI_BATCH = 16


def finetune_on_feedback(model: DualScanHMamba,
                         confirmed_frps: List[str],
                         non_frps: List[str],
                         meta_db: Dict,
                         optimizer: optim.Optimizer,
                         device: torch.device,
                         finetune_steps: int = 10,
                         lr_scale: float = 0.1) -> Dict:
    """
    Fine-tune model based on APD feedback.

    Strategy:
      - Confirmed FRPs: train model to reconstruct these (positive signal).
      - Non-FRPs: use as negative examples with inverted loss weighting
        (penalize generating similar patterns).

    Memory safety: samples are split into mini-batches of at most
    _FT_MINI_BATCH entries so that a large non-FRP list (e.g. 182 items)
    never causes an OOM during backprop.

    Returns:
        dict with fine-tuning metrics.
    """
    if not confirmed_frps and not non_frps:
        return {"finetune_loss": 0.0, "steps": 0}

    model.train()

    # Scale learning rate for fine-tuning
    for pg in optimizer.param_groups:
        pg['lr'] = pg.get('base_lr', pg['lr']) * lr_scale

    total_loss = 0.0
    steps_done = 0

    def _run_mini_batches(prefixes: List[str],
                          loss_scale: float,
                          max_steps: int,
                          grad_clip: float):
        """Iterate over prefixes in mini-batches, accumulate loss."""
        nonlocal total_loss, steps_done
        step = 0
        for start in range(0, len(prefixes), _FT_MINI_BATCH):
            if step >= max_steps:
                break
            chunk = prefixes[start: start + _FT_MINI_BATCH]
            batch = _build_finetune_batch(chunk, meta_db, device)
            if batch is None:
                continue
            bits, meta, prefix_lens = batch
            try:
                result = model.compute_mlm_loss(
                    bit_target=bits, meta=meta, prefix_lens=prefix_lens)
                loss = loss_scale * result["loss"]

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip)
                optimizer.step()

                total_loss += abs(loss.item())
                steps_done += 1
                step += 1
            finally:
                # Explicitly free activations to keep peak RSS low
                del bits, meta, prefix_lens, batch
                gc.collect()

    # Positive examples: confirmed FRPs
    if confirmed_frps:
        _run_mini_batches(confirmed_frps,
                          loss_scale=1.0,
                          max_steps=min(finetune_steps, 5),
                          grad_clip=1.0)

    # Negative examples: non-FRPs (with reduced weight)
    # Shuffle so each run sees a different subset when capped by max_steps.
    if non_frps:
        shuffled_neg = list(non_frps)
        random.shuffle(shuffled_neg)
        _run_mini_batches(shuffled_neg,
                          loss_scale=-0.1,
                          max_steps=min(finetune_steps // 2, 3),
                          grad_clip=0.5)

    # Restore learning rate
    for pg in optimizer.param_groups:
        pg['lr'] = pg.get('base_lr', pg['lr']) / lr_scale

    return {
        "finetune_loss": total_loss / max(steps_done, 1),
        "steps": steps_done,
        "pos_samples": len(confirmed_frps),
        "neg_samples": len(non_frps),
    }


def _build_finetune_batch(prefixes: List[str], meta_db: Dict,
                          device: torch.device):
    """Build a batch from prefix list for fine-tuning."""
    bits_list = []
    meta_lists = defaultdict(list)
    plen_list = []

    for prefix in prefixes:
        # Try to find metadata (may not exist for newly discovered prefixes)
        meta = meta_db.get(prefix)
        if meta is None:
            # Try to find a covering prefix in meta_db
            try:
                net = ipaddress.IPv6Network(prefix, strict=False)
            except Exception:
                continue
            # Use default metadata
            meta = {
                "as_id": 0, "org_id": 0, "cat1_id": 0,
                "cat2_id": 0, "proto_id": 0, "geo_id": 0,
                "rtt_stats": [0.0, 0.0, 0.0],
            }

        bits = prefix_to_bits(prefix)
        bits_list.append(bits)
        meta_lists["as_id"].append(meta["as_id"])
        meta_lists["org_id"].append(meta["org_id"])
        meta_lists["cat1_id"].append(meta["cat1_id"])
        meta_lists["cat2_id"].append(meta["cat2_id"])
        meta_lists["proto_id"].append(meta["proto_id"])
        meta_lists["geo_id"].append(meta["geo_id"])
        meta_lists["rtt_stats"].append(meta["rtt_stats"])

        try:
            net = ipaddress.IPv6Network(prefix, strict=False)
            plen_list.append(net.prefixlen)
        except Exception:
            plen_list.append(48)

    if not bits_list:
        return None

    bits_t = torch.tensor(bits_list, dtype=torch.long, device=device)
    meta_t = {
        "as_id": torch.tensor(meta_lists["as_id"], dtype=torch.long, device=device),
        "org_id": torch.tensor(meta_lists["org_id"], dtype=torch.long, device=device),
        "cat1_id": torch.tensor(meta_lists["cat1_id"], dtype=torch.long, device=device),
        "cat2_id": torch.tensor(meta_lists["cat2_id"], dtype=torch.long, device=device),
        "proto_id": torch.tensor(meta_lists["proto_id"], dtype=torch.long, device=device),
        "geo_id": torch.tensor(meta_lists["geo_id"], dtype=torch.long, device=device),
        "rtt_stats": torch.tensor(meta_lists["rtt_stats"], dtype=torch.float32, device=device),
    }
    plen_t = torch.tensor(plen_list, dtype=torch.long, device=device)

    return bits_t, meta_t, plen_t


# ============================================================================
# Main Online Inference Loop
# ============================================================================

def online_inference_loop(args):
    """
    Main online inference loop.

    Execution flow (per round):
      1. Generate a batch of candidate prefixes with the model.
      2. Split candidates into:
           * auto-confirmed FRPs   -- covered by an already-known FRP
                                      (no active probing needed; still fed
                                      back to the model as positive signal)
           * to-probe              -- go through APD (refine mode)
      3. Run APD on the to-probe set.
      4. Merge newly-confirmed FRPs into the persistent known-FRP set.
      5. Once the *accumulated new probes* since the last fine-tune has
         reached ``--finetune-probe-threshold``, run a fine-tune pass on
         the freshly-collected feedback.
      6. When the *global* probe budget ``--max-total-probes`` is hit,
         stop the whole loop immediately.
      7. At the end, dump a merged known-FRP snapshot combining the
         initial seed set with everything discovered this run.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(_banner("ONLINE FRP INFERENCE PIPELINE"))
    logger.info(f"  Device            : {device}")
    logger.info(f"  Checkpoint        : {args.checkpoint}")
    logger.info(f"  Data dir          : {args.data_dir}")
    logger.info(f"  Output dir        : {args.output_dir}")
    logger.info(f"  Known-FRP file    : {args.known_frps}")
    logger.info(f"  Rounds            : {args.rounds}")
    logger.info(f"  Candidates/round  : {args.candidates_per_round}")
    logger.info(f"  Temperature       : {args.temperature}")
    logger.info(f"  APD refine mode   : {args.apd_refine}")
    logger.info(f"  Finetune threshold: {args.finetune_probe_threshold} probes")
    logger.info(f"  Max total probes  : {args.max_total_probes}")

    # ---- Load checkpoint & model ----
    logger.info(_sub_banner("Loading model"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = DualScanConfig.from_dict(ckpt["config"])
    model = DualScanHMamba(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("  Model loaded successfully")

    # ---- Load data ----
    logger.info(_sub_banner("Loading data"))
    data_dir = args.data_dir
    meta_path = os.path.join(data_dir, "meta.json")
    as_db_path = os.path.join(data_dir, "as_seed_db.json")
    filtered_seeds_path = os.path.join(data_dir, "filtered_seeds.txt")
    with open(meta_path) as f:
        meta_db = json.load(f)
    with open(as_db_path) as f:
        as_seed_db = json.load(f)
    logger.info(f"  meta.json         : {len(meta_db)} prefixes")
    logger.info(f"  as_seed_db        : {len(as_seed_db)} ASes")

    # ---- Filter & known-FRP set ----
    seed_filter = SeedFilter(filtered_seeds_path)

    os.makedirs(args.output_dir, exist_ok=True)
    # If no explicit --known-frps was supplied, default to
    # <output-dir>/known_frps.txt so the run is still cumulative.
    known_frps_path = args.known_frps or os.path.join(
        args.output_dir, "known_frps.txt")
    # Seed the known-FRP file with the initial seed file if it does not
    # exist yet (so first-run has a non-empty starting set).
    if not os.path.exists(known_frps_path) and args.bootstrap_known_from_seeds:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(known_frps_path))
                        or '.', exist_ok=True)
            if os.path.exists(filtered_seeds_path):
                with open(filtered_seeds_path) as fin, \
                     open(known_frps_path, 'w') as fout:
                    for line in fin:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            fout.write(line + "\n")
                logger.info(f"  Bootstrapped known-FRP file from filtered "
                            f"seeds: {known_frps_path}")
        except Exception as e:
            logger.warning(f"  Bootstrap of known-FRP file failed: {e}")

    known_frps = KnownFRPSet(known_frps_path)
    initial_known_count = len(known_frps)

    # ---- Optimizer ----
    optimizer = optim.AdamW(model.parameters(), lr=args.finetune_lr,
                            weight_decay=1e-4)
    for pg in optimizer.param_groups:
        pg['base_lr'] = pg['lr']

    # ---- State ----
    all_confirmed_frps: List[str] = []      # all FRPs discovered this run
    all_non_frps: List[str] = []
    round_results: List[Dict] = []

    total_probes_used = 0                   # cumulative APD probes
    probes_since_last_ft = 0                # probes accumulated since last ft
    pending_pos_feedback: List[str] = []    # FRPs waiting to be fed back
    pending_neg_feedback: List[str] = []    # non-FRPs waiting to be fed back
    stop_reason: Optional[str] = None

    # ---- Live-appending FRP log (real-time on-disk record) ----
    live_frp_path = os.path.join(args.output_dir, "confirmed_frps_live.txt")
    # Truncate at the start so this run has a clean record.
    with open(live_frp_path, 'w') as f:
        f.write("# prefix\tsource\tprobes_used_so_far\ttimestamp\n")
    live_frp_fh = open(live_frp_path, 'a', buffering=1)  # line-buffered

    def _record_frp(prefix: str, source: str):
        """Append a single FRP to the live log immediately."""
        try:
            live_frp_fh.write(
                f"{prefix}\t{source}\t{total_probes_used}\t"
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
        except Exception:
            pass

    # ---- Snapshot targets: 0.5M / 2.5M / 4.5M / 6.5M probes ----
    snapshot_targets_default = [500_000, 2_500_000, 4_500_000, 6_500_000]
    snapshot_targets = sorted(
        args.snapshot_probe_targets or snapshot_targets_default)
    # The last target is the hard-stop; anything at/above it terminates.
    hard_stop_probes = snapshot_targets[-1] if snapshot_targets else None
    snapshots_taken: List[Dict] = []
    next_snapshot_idx = 0

    def _take_snapshot(tag_probes: int):
        """Dump the current FRP list to a tagged file."""
        # e.g. 500000 -> "0.5M", 2_500_000 -> "2.5M"
        tag = f"{tag_probes/1_000_000:.1f}M"
        snap_path = os.path.join(
            args.output_dir, f"frps_at_{tag}_probes.txt")
        with open(snap_path, 'w') as f:
            f.write(f"# snapshot at {tag_probes} probes\n")
            f.write(f"# actual probes used: {total_probes_used}\n")
            f.write(f"# FRPs discovered so far: {len(all_confirmed_frps)}\n")
            f.write(f"# known-FRP set size    : {len(known_frps)}\n")
            for p in all_confirmed_frps:
                f.write(f"{p}\n")
        snapshots_taken.append({
            "target_probes": tag_probes,
            "actual_probes": total_probes_used,
            "frps_at_snapshot": len(all_confirmed_frps),
            "known_frp_size": len(known_frps),
            "path": snap_path,
        })
        logger.info(_sub_banner(f"SNAPSHOT @ {tag_probes:,} probes"))
        logger.info(f"        actual probes    : {total_probes_used:,}")
        logger.info(f"        FRPs so far      : {len(all_confirmed_frps):,}")
        logger.info(f"        known-FRP size   : {len(known_frps):,}")
        logger.info(f"        snapshot file    : {snap_path}")

    logger.info(_sub_banner("Initial state"))
    logger.info(f"  Aligned seeds     : {len(seed_filter.seed_networks)}")
    logger.info(f"  Known FRPs (init) : {initial_known_count}")
    logger.info(f"  Snapshot targets  : "
                f"{', '.join(f'{t:,}' for t in snapshot_targets)} probes")
    logger.info(f"  Hard stop         : {hard_stop_probes:,} probes"
                if hard_stop_probes else "  Hard stop         : (none)")
    logger.info(_banner("STARTING ROUNDS"))

    for round_idx in range(args.rounds):
        round_num = round_idx + 1
        t0 = time.time()
        logger.info("")
        logger.info(_sub_banner(f"Round {round_num}/{args.rounds}"))
        logger.info(f"  [budget] probes used so far: "
                    f"{total_probes_used}/{args.max_total_probes} "
                    f"| cumulative FRPs: {len(all_confirmed_frps)} "
                    f"| known FRPs: {len(known_frps)}")

        # -------- Step 1: Generate candidates --------
        logger.info("  [1/4] Generating candidates ...")
        to_probe, auto_confirmed, known_range_count, total_before_filter = \
            generate_candidates(
                model=model,
                meta_db=meta_db,
                as_seed_db=as_seed_db,
                seed_filter=seed_filter,
                num_candidates=args.candidates_per_round,
                temperature=args.temperature,
                device=device,
                known_frps=known_frps,
            )
        logger.info(f"        raw generated     : {total_before_filter}")
        logger.info(f"        in-known-range    : {known_range_count} "
                    f"(silently discarded)")
        logger.info(f"        auto-confirmed FRP: {len(auto_confirmed)} "
                    f"(covered by known FRP -> skip probing)")
        logger.info(f"        will probe (APD)  : {len(to_probe)}")

        # Save this round's candidates split into two files
        cand_path = os.path.join(args.output_dir,
                                 f"round{round_num}_candidates.txt")
        with open(cand_path, 'w') as f:
            for c in to_probe:
                f.write(f"{c}\n")
        if auto_confirmed:
            auto_path = os.path.join(args.output_dir,
                                     f"round{round_num}_auto_confirmed.txt")
            with open(auto_path, 'w') as f:
                for c in auto_confirmed:
                    f.write(f"{c}\n")

        # Adaptive temperature (strategy C: too many candidates fall inside
        # a known seed range -> raise temperature).
        if total_before_filter > 0 and args.adaptive_temperature:
            known_ratio = known_range_count / total_before_filter
            if known_ratio > 0.3:
                args.temperature = min(args.temperature * 1.1, 2.0)
                logger.info(f"        [strategy C] known-range ratio="
                            f"{known_ratio:.1%} > 30%, "
                            f"raised temperature to {args.temperature:.2f}")

        # -------- Step 2: APD probing on the to-probe set --------
        remaining_budget = args.max_total_probes - total_probes_used
        if remaining_budget <= 0:
            stop_reason = "max-total-probes reached before probing"
            confirmed_new: List[str] = []
            non_frps_new: List[str] = []
            probes_used_this_round = 0
        elif not to_probe:
            logger.info("  [2/4] APD skipped (no candidate needs probing)")
            confirmed_new = []
            non_frps_new = []
            probes_used_this_round = 0
        else:
            logger.info(f"  [2/4] Running APD (refine mode) on "
                        f"{len(to_probe)} candidates "
                        f"(budget left: {remaining_budget}) ...")
            if args.dry_run:
                apd_result = _simulate_apd(to_probe)
                # In dry-run each candidate counts as 1 "probe"
                apd_result["probes_used"] = len(to_probe)
                logger.info("        (DRY RUN: simulated APD results)")
            else:
                apd_result = run_apd_on_candidates(
                    to_probe,
                    timeout=args.apd_timeout,
                    max_attempts=args.apd_max_attempts,
                    concurrency=args.apd_concurrency,
                    refine=args.apd_refine,
                    workers=args.apd_workers,
                    max_probes=remaining_budget,
                    quiet=args.quiet_apd,
                )
            confirmed_new = apd_result["confirmed_frps"]
            non_frps_new = apd_result["non_frps"]
            probes_used_this_round = int(apd_result.get("probes_used", 0))
            total_probes_used += probes_used_this_round

            logger.info(f"        confirmed FRPs    : {len(confirmed_new)}")
            logger.info(f"        non-FRPs          : {len(non_frps_new)}")
            logger.info(f"        probes used       : {probes_used_this_round} "
                        f"(total {total_probes_used}/{args.max_total_probes})")
            if apd_result.get("unprocessed"):
                logger.info(f"        unprocessed (budget cut): "
                            f"{len(apd_result['unprocessed'])}")

        # -------- Step 3: Merge results & feedback pools --------
        # Auto-confirmed FRPs act as free positive samples.
        for p in auto_confirmed:
            all_confirmed_frps.append(p)
            pending_pos_feedback.append(p)
            _record_frp(p, "auto")

        # Freshly discovered FRPs update the persistent known set.
        newly_added_to_known = 0
        for p in confirmed_new:
            all_confirmed_frps.append(p)
            pending_pos_feedback.append(p)
            _record_frp(p, "probed")
            if known_frps.add(p, persist=True):
                newly_added_to_known += 1
                # Also treat the new FRP as an aligned seed so future
                # generations don't produce trivially-covered candidates.
                try:
                    net = ipaddress.IPv6Network(p, strict=False)
                    seed_filter.seed_networks.append(net)
                    seed_filter.seed_set.add(str(net))
                except Exception:
                    pass
        for p in non_frps_new:
            all_non_frps.append(p)
            pending_neg_feedback.append(p)

        probes_since_last_ft += probes_used_this_round

        # -------- Take probe-count snapshots if any target is crossed --------
        while (next_snapshot_idx < len(snapshot_targets)
               and total_probes_used >= snapshot_targets[next_snapshot_idx]):
            _take_snapshot(snapshot_targets[next_snapshot_idx])
            next_snapshot_idx += 1

        logger.info(f"        known-FRP file    : +{newly_added_to_known} "
                    f"(now {len(known_frps)} entries -> {known_frps.path})")

        # -------- Step 4: Fine-tune when threshold reached --------
        threshold = args.finetune_probe_threshold
        # Trigger conditions:
        #   * accumulated new probes reached threshold, OR
        #   * we are about to stop (max budget hit) and have pending feedback,
        #   * OR we have auto-confirmed positives to reinforce (only if some
        #     minimum amount of positive feedback exists).
        trigger_ft = False
        if threshold > 0 and probes_since_last_ft >= threshold:
            trigger_ft = True
            ft_reason = f"probe threshold ({probes_since_last_ft} >= {threshold})"
        elif (total_probes_used >= args.max_total_probes
              and (pending_pos_feedback or pending_neg_feedback)):
            trigger_ft = True
            ft_reason = "final flush before stopping (max-probes hit)"
        elif (round_num == args.rounds
              and (pending_pos_feedback or pending_neg_feedback)):
            trigger_ft = True
            ft_reason = "final flush before stopping (last round)"

        if trigger_ft:
            logger.info(f"  [3/4] Fine-tuning triggered: {ft_reason}")
            logger.info(f"        pos samples: {len(pending_pos_feedback)} | "
                        f"neg samples: {len(pending_neg_feedback)}")
            ft_result = finetune_on_feedback(
                model=model,
                confirmed_frps=pending_pos_feedback,
                non_frps=pending_neg_feedback,
                meta_db=meta_db,
                optimizer=optimizer,
                device=device,
                finetune_steps=args.finetune_steps,
            )
            logger.info(f"        fine-tune loss={ft_result['finetune_loss']:.4f}, "
                        f"steps={ft_result['steps']}")
            # Reset accumulators
            pending_pos_feedback = []
            pending_neg_feedback = []
            probes_since_last_ft = 0
        else:
            logger.info(f"  [3/4] Fine-tune deferred "
                        f"(probes since last: {probes_since_last_ft}/{threshold})")
            ft_result = {"finetune_loss": 0.0, "steps": 0}

        # -------- Step 5: Bookkeeping --------
        elapsed = time.time() - t0
        round_info = {
            "round": round_num,
            "raw_generated": total_before_filter,
            "known_range_discarded": known_range_count,
            "auto_confirmed": len(auto_confirmed),
            "probed": len(to_probe),
            "confirmed_frps": len(confirmed_new),
            "confirmed_frps_list": confirmed_new,
            "auto_confirmed_list": auto_confirmed,
            "non_frps": len(non_frps_new),
            "probes_used": probes_used_this_round,
            "cumulative_probes": total_probes_used,
            "cumulative_frps": len(all_confirmed_frps),
            "known_frp_size": len(known_frps),
            "finetuned": trigger_ft,
            "finetune_loss": ft_result["finetune_loss"],
            "finetune_steps": ft_result["steps"],
            "elapsed_seconds": round(elapsed, 1),
        }
        round_results.append(round_info)

        # Adaptive temperature based on FRP hit rate
        total_probed = len(to_probe)
        if args.adaptive_temperature and total_probed > 0:
            if len(confirmed_new) == 0:
                args.temperature = min(args.temperature * 1.1, 2.0)
                logger.info(f"        adaptive temp -> {args.temperature:.2f} "
                            f"(no new FRPs this round)")
            elif len(confirmed_new) > total_probed * 0.5:
                args.temperature = max(args.temperature * 0.95, 0.3)
                logger.info(f"        adaptive temp -> {args.temperature:.2f} "
                            f"(FRP hit rate high)")

        logger.info(f"  [4/4] Round {round_num} done in {elapsed:.1f}s | "
                    f"+{len(confirmed_new)+len(auto_confirmed)} FRPs "
                    f"({len(confirmed_new)} probed + {len(auto_confirmed)} auto) "
                    f"| cumulative={len(all_confirmed_frps)}")

        # Persist incremental results
        _save_results(args.output_dir, round_results,
                      all_confirmed_frps, all_non_frps)

        # Periodic checkpoint
        if args.save_every > 0 and round_num % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir,
                                     f"model_round{round_num}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg.to_dict(),
                "round": round_num,
                "cumulative_frps": len(all_confirmed_frps),
                "cumulative_probes": total_probes_used,
            }, ckpt_path)
            logger.info(f"        checkpoint saved: {ckpt_path}")

        # Cooperative yield to keep CPU usage tame
        if args.round_sleep > 0:
            time.sleep(args.round_sleep)

        # Stop when we've hit the hard-stop snapshot (e.g. 6.5M probes)
        if hard_stop_probes is not None and total_probes_used >= hard_stop_probes:
            stop_reason = (f"reached hard-stop snapshot "
                           f"({total_probes_used:,}/{hard_stop_probes:,})")
            break

        # Stop when we've hit the global probe budget
        if total_probes_used >= args.max_total_probes:
            stop_reason = (f"reached max-total-probes "
                           f"({total_probes_used:,}/{args.max_total_probes:,})")
            break

    # ================================================================
    # Final summary & merged known-FRP snapshot
    # ================================================================
    logger.info("")
    logger.info(_banner("PIPELINE COMPLETE"))
    if stop_reason:
        logger.info(f"  Stop reason        : {stop_reason}")
    else:
        logger.info(f"  Stop reason        : all {args.rounds} rounds finished")
    total_seen = len(all_confirmed_frps) + len(all_non_frps)
    logger.info(f"  Rounds executed    : {len(round_results)}/{args.rounds}")
    logger.info(f"  Total probes used  : {total_probes_used}/"
                f"{args.max_total_probes}")
    logger.info(f"  Confirmed FRPs     : {len(all_confirmed_frps)}")
    logger.info(f"  Non-FRPs           : {len(all_non_frps)}")
    if total_seen > 0:
        logger.info(f"  FRP discovery rate : "
                    f"{len(all_confirmed_frps) / total_seen * 100:.1f}%")
    logger.info(f"  Known-FRP file     : {known_frps_path} "
                f"(size: {len(known_frps)}, "
                f"delta: +{len(known_frps) - initial_known_count})")

    # Ensure any snapshot targets not yet taken (in case we stopped early) are
    # skipped, but always write the final snapshot for the run itself.
    try:
        live_frp_fh.close()
    except Exception:
        pass
    logger.info(f"  Live FRP log       : {live_frp_path}")
    if snapshots_taken:
        logger.info(_sub_banner("Snapshots recorded"))
        for s in snapshots_taken:
            logger.info(f"    - @ {s['target_probes']:>10,} probes -> "
                        f"{s['frps_at_snapshot']:>7,} FRPs  ({s['path']})")

    # Write the merged final snapshot (initial known + newly-discovered)
    merged_path = os.path.join(args.output_dir, "known_frps_merged.txt")
    known_frps.snapshot_to(merged_path)
    logger.info(f"  Merged snapshot    : {merged_path} "
                f"({len(known_frps)} unique FRPs)")

    # Save final model
    final_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg.to_dict(),
        "total_rounds": len(round_results),
        "cumulative_frps": len(all_confirmed_frps),
        "cumulative_probes": total_probes_used,
    }, final_path)
    logger.info(f"  Final model        : {final_path}")
    logger.info("=" * 78)

    return {
        "confirmed_frps": all_confirmed_frps,
        "non_frps": all_non_frps,
        "round_results": round_results,
        "known_frps_path": known_frps_path,
        "merged_snapshot": merged_path,
        "total_probes_used": total_probes_used,
        "stop_reason": stop_reason,
        "snapshots": snapshots_taken,
        "live_frp_log": live_frp_path,
    }


def _simulate_apd(candidates: List[str]) -> Dict:
    """Simulate APD results for dry-run mode (no network access needed)."""
    # Simulate ~20% FRP rate
    confirmed = []
    non_frps = []
    for c in candidates:
        if random.random() < 0.2:
            confirmed.append(c)
        else:
            non_frps.append(c)
    return {
        "confirmed_frps": confirmed,
        "non_frps": non_frps,
        "stats": {"simulated": True},
    }


def _save_results(output_dir: str, round_results: List[Dict],
                  confirmed_frps: List[str], non_frps: List[str]):
    """Save incremental results to disk."""
    # Round results
    results_path = os.path.join(output_dir, "round_results.json")
    with open(results_path, 'w') as f:
        json.dump(round_results, f, indent=2, ensure_ascii=False)

    # Confirmed FRPs
    frp_path = os.path.join(output_dir, "confirmed_frps.txt")
    with open(frp_path, 'w') as f:
        for p in confirmed_frps:
            f.write(f"{p}\n")

    # Summary
    summary = {
        "total_rounds": len(round_results),
        "total_confirmed_frps": len(confirmed_frps),
        "total_non_frps": len(non_frps),
        "discovery_rate": len(confirmed_frps) / max(len(confirmed_frps) + len(non_frps), 1),
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Online Inference for DualScanHMamba FRP Generator")

    # Model & data
    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', default='data_fixed',
                        help='Directory with preprocessed data')
    parser.add_argument('--output-dir', default='results/online_infer',
                        help='Output directory for results')

    # Generation parameters
    parser.add_argument('--rounds', type=int, default=100000,
                        help='Upper bound on rounds. Real stop is governed '
                             'by --max-total-probes / snapshot hard-stop.')
    parser.add_argument('--candidates-per-round', type=int, default=500,
                        help='Candidates to generate per round '
                             '(bigger => fewer rounds, less overhead).')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--adaptive-temperature', action='store_true',
                        help='Adaptively adjust temperature based on FRP rate')
    parser.add_argument('--round-sleep', type=float, default=0.05,
                        help='Sleep (seconds) between rounds to yield CPU. '
                             'Set to 0 to disable.')

    # APD parameters
    parser.add_argument('--apd-timeout', type=float, default=2.0,
                        help='APD per-attempt timeout (seconds)')
    parser.add_argument('--apd-max-attempts', type=int, default=5,
                        help='APD max ping attempts per sub-prefix')
    parser.add_argument('--apd-concurrency', type=int, default=8,
                        help='APD parallel probes within a prefix '
                             '(lower => less CPU)')
    parser.add_argument('--apd-workers', type=int, default=4,
                        help='APD prefix-level worker threads '
                             '(lower => less CPU)')
    parser.add_argument('--apd-refine', action='store_true', default=True,
                        help='Enable APD granularity refinement (default: on)')
    parser.add_argument('--no-apd-refine', dest='apd_refine',
                        action='store_false',
                        help='Disable APD granularity refinement')
    parser.add_argument('--quiet-apd', action='store_true', default=True,
                        help='Suppress per-prefix APD progress output '
                             '(default: on, saves stdout overhead)')
    parser.add_argument('--verbose-apd', dest='quiet_apd',
                        action='store_false',
                        help='Show per-prefix APD progress output')

    # Fine-tuning parameters
    parser.add_argument('--finetune-lr', type=float, default=1e-4,
                        help='Fine-tuning learning rate')
    parser.add_argument('--finetune-steps', type=int, default=10,
                        help='Fine-tuning steps per round')
    parser.add_argument('--finetune-probe-threshold', type=int, default=200,
                        help='Trigger fine-tuning once this many new APD '
                             'probes have accumulated since the last '
                             'fine-tune. 0 = never trigger by threshold.')

    # Global stop condition
    parser.add_argument('--max-total-probes', type=int, default=6_500_000,
                        help='Global cap on total APD probes. When reached '
                             'the pipeline stops (after a final fine-tune '
                             'flush). Default: 6.5M.')
    parser.add_argument('--snapshot-probe-targets', type=int, nargs='+',
                        default=None,
                        help='Probe counts (space separated) at which to '
                             'dump the current FRP list as a snapshot. '
                             'Default: 500000 2500000 4500000 6500000. '
                             'The last one is also the hard stop.')

    # Known-FRP set
    parser.add_argument('--known-frps', type=str, default=None,
                        help='Path to a persistent known-FRP file. '
                             'Candidates that are sub-prefixes of an entry '
                             'in this file are auto-confirmed without '
                             'probing. New FRPs are appended in real time. '
                             'Default: <output-dir>/known_frps.txt')
    parser.add_argument('--bootstrap-known-from-seeds', action='store_true',
                        help='If the known-FRP file does not exist yet, '
                             'initialise it from data_fixed/filtered_seeds.txt')

    # Misc
    parser.add_argument('--save-every', type=int, default=0,
                        help='Save checkpoint every N rounds (0 to disable). '
                             'Default 0 – snapshots are already probe-based.')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run (simulate APD, no network access)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--torch-threads', type=int, default=2,
                        help='torch.set_num_threads value (lower => less CPU)')

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Reduce CPU-side thread contention (BLAS / torch intra-op)
    try:
        torch.set_num_threads(max(1, args.torch_threads))
        torch.set_num_interop_threads(max(1, min(2, args.torch_threads)))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, args.torch_threads)))
    os.environ.setdefault("MKL_NUM_THREADS", str(max(1, args.torch_threads)))

    online_inference_loop(args)


if __name__ == "__main__":
    main()
