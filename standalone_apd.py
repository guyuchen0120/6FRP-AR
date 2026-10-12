#!/usr/bin/env python3
"""
Standalone APD Detection + Granularity Alignment + Tree Alignment.

Ping-style probing: for each prefix, probe 16 sub-prefixes with up to 5
ICMPv6 packets each (1/sec). Any reply = reachable. All 16 = FRP.

Usage:
    # Basic detection
    sudo python3 standalone_apd.py -i prefixes.txt -o result.json

    # Detection + granularity alignment (find shortest FRP)
    sudo python3 standalone_apd.py -i prefixes.txt -o result.json --refine

    # Detection + tree alignment (find sub-FRPs from partial prefixes)
    sudo python3 standalone_apd.py -i prefixes.txt -o result.json --ta

    # Full pipeline: detection + refinement + tree alignment
    sudo python3 standalone_apd.py -i prefixes.txt -o result.json --refine --ta

    # Custom parameters
    sudo python3 standalone_apd.py -i prefixes.txt -o result.json \\
        --refine --ta \\
        --timeout 2 --max-attempts 5 --concurrency 16 --min-prefix-len 16

    # Quick test (first 10 prefixes only)
    sudo python3 standalone_apd.py -i prefixes.txt -o result.json --limit 10

Input file format (prefixes.txt):
    # Comments start with #
    2a02:26f0::/32
    2600:9000:201d::/48
    2a06:3040:c301::/48

Output:
    result.json         - detailed probe results per prefix (skipped in --quiet)
    result.jsonl        - incremental per-prefix results (always written)
    result_frp.txt      - final confirmed FRP list
    result_summary.json - pipeline statistics
"""

import argparse
import ipaddress
import json
import logging
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Import apd_core
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from utils.apd_core import (
    diagnose_prefix,
    determine_frp_status,
    refine_frp_prefix,
    tree_align_prefix,
    is_covered_by,
    has_raw_socket_capability,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _extract_failed_probes(diag_result):
    """Extract failed (unreachable) probe addresses from a diagnose result."""
    probes = diag_result.get("probes", [])
    if not probes:
        return []
    failed = []
    for p in probes:
        if p and not p.get("reachable", False):
            failed.append({
                "address": p.get("address", "?"),
                "nibble_index": p.get("nibble_index", -1),
                "classification": p.get("last_classification", "UNKNOWN"),
                "attempts_used": p.get("attempts_used", 0),
                "total_time_ms": p.get("total_time_ms", 0),
            })
    return failed


def _format_duration(seconds):
    """Format seconds into human-readable string like '1h 23m 45s'."""
    seconds = int(seconds)
    if seconds < 60:
        return "%ds" % seconds
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return "%dm %ds" % (m, s)
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return "%dh %dm %ds" % (h, m, s)


# ============================================================================
# Core Pipeline
# ============================================================================

def run_apd_pipeline(
    prefixes,
    refine=True,
    tree_align=False,
    timeout=2.0,
    max_attempts=5,
    concurrency=16,
    min_prefix_len=16,
    output_path=None,
    frp_output_path=None,
    max_ta_depth=4,
    fast_refine=False,
    workers=8,
    quiet=False,
    max_probes=None,
    seed_frps=None,
    tune_time_limit=None,
    tune_probe_limit=None,
    incremental_mode=False,
    refine_mode=None,
    refine_timeout=None,
    refine_max_attempts=None,
):
    """
    Complete APD pipeline: detect -> judge -> refine -> tree-align -> filter.

    Runs prefix-level tasks in parallel with a thread pool. Each worker
    completes detect (+optional refine / tree-align) for one prefix,
    independent of others. Shared state (confirmed FRPs, counters,
    incremental output) is protected by a lock. When a broader FRP is
    later confirmed, previously-added narrower FRPs it covers are
    marked as superseded.

    Args:
        prefixes: list of IPv6 prefix strings
        refine: whether to do granularity alignment
        tree_align: whether to do tree-based alignment for non-FRP prefixes
        timeout: per-attempt timeout in seconds
        max_attempts: max ping attempts per sub-prefix
        concurrency: parallel probes within a single prefix
        min_prefix_len: minimum prefix length for refinement
        output_path: path for detailed JSON output
        frp_output_path: path for FRP list output
        workers: number of prefix-level worker threads (default 8)
        quiet: if True, suppress real-time per-prefix output
        max_probes: if set, stop submitting new tasks once total probes
                    reaches this limit (already-running tasks finish)
        seed_frps:  optional list of already-known FRP prefix strings.
                    When provided, seeds are pre-loaded into confirmed_frps
                    (with source="seed") and:
                      * participate in the coverage dedup (new sub-prefixes
                        covered by a seed are dropped without extra probing);
                      * are written to `frp_output_path` at start;
                      * are included in the final `confirmed_frps` output.
                    If a newly-confirmed direct/tree-align FRP is BROADER
                    than a seed, the seed will be removed by the normal
                    "remove narrower ones" logic.
        tune_time_limit: if set, stop submitting new tasks after this many
                    seconds. Used by --auto-tune mode.
        tune_probe_limit: like max_probes but only used as a soft cap for
                    the tune stage; separate name to avoid confusion.
        incremental_mode: if True, do NOT truncate the .jsonl / _frp.txt
                    files at start (append to whatever the caller already
                    wrote). Used by --auto-tune stages so tune-stage output
                    is preserved when the final stage runs.

    Returns:
        dict with pipeline results
    """
    # Pre-flight check
    if not has_raw_socket_capability():
        raise PermissionError(
            "Need root/CAP_NET_RAW for raw ICMPv6 sockets. "
            "Run with: sudo python3 standalone_apd.py ..."
        )

    # Prepare incremental output files
    jsonl_path = None
    if output_path:
        base = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
        jsonl_path = base + ".jsonl"
        if not incremental_mode:
            with open(jsonl_path, 'w') as f:
                f.write("")
        if not quiet:
            logger.info("Incremental results: %s", jsonl_path)
    if frp_output_path and not incremental_mode:
        with open(frp_output_path, 'w') as f:
            f.write("")

    # ---- Shared state ----
    state_lock = threading.Lock()
    confirmed_frps = []       # list of {"prefix": ..., "original": ..., "source": ...}
    all_results = []
    skipped = []
    ta_results = []
    processed_count = 0
    total_probes = 0
    detection_probes = 0
    refinement_probes = 0
    ta_probes = 0
    ta_new_frps = 0
    ta_candidates_checked = 0
    submitted_count = 0
    stop_submitting = False   # flipped when max_probes reached
    # Track how many of the pre-seeded FRPs survived to the end (for stats)
    initial_seed_count = 0
    # For quiet-mode progress line
    _quiet_last_len = [0]     # mutable holder so nested fn can mutate

    start_time = time.time()

    # ---- Pre-seed confirmed FRPs (option B: seed-mode=include) ----
    if seed_frps:
        seen = set()
        for s in seed_frps:
            try:
                s_net = ipaddress.IPv6Network(s, strict=False)
            except Exception as e:
                logger.warning("Seed %r invalid, skipped: %s", s, e)
                continue
            key = str(s_net)
            if key in seen:
                continue
            seen.add(key)
            confirmed_frps.append({
                "prefix": key,
                "original": key,
                "source": "seed",
            })
            initial_seed_count += 1
        # Write seeds to the FRP list file so consumers see them from t=0
        if frp_output_path and initial_seed_count > 0:
            with open(frp_output_path, 'a') as f:
                for entry in confirmed_frps:
                    if entry["source"] == "seed":
                        f.write("%s\n" % entry["prefix"])
        logger.info(
            "Pre-seeded %d FRPs into confirmed set (mode=include)",
            initial_seed_count,
        )

    logger.info("=" * 60)
    logger.info("APD Pipeline: %d prefixes", len(prefixes))
    logger.info("  ping-style: max %d attempts/sub-prefix, timeout=%gs",
                max_attempts, timeout)
    logger.info("  refine=%s, tree_align=%s, concurrency=%d, workers=%d",
                refine, tree_align, concurrency, workers)
    if max_probes is not None:
        logger.info("  max_probes=%d (auto-stop after this many probes)", max_probes)
    logger.info("=" * 60)

    # Real-time header
    if not quiet:
        print("\n" + "=" * 78)
        print("APD REAL-TIME RESULTS (ping-style: %d attempts, refine=%s, ta=%s, workers=%d)"
              % (max_attempts, refine, tree_align, workers))
        print("=" * 78)
        print("%-5s %-42s %-10s %-8s %s" % ("#", "Prefix", "Status", "Reach", "Detail"))
        print("%-5s %-42s %-10s %-8s %s" % ("-" * 5, "-" * 42, "-" * 10, "-" * 8, "-" * 30))
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Helpers (closures capturing shared state)
    # ------------------------------------------------------------------
    def _is_covered_by_any_locked(prefix_net):
        """Return covering FRP prefix string, or None. Caller must hold lock."""
        for f in confirmed_frps:
            try:
                f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
            except Exception:
                continue
            if is_covered_by(prefix_net, f_net):
                return f["prefix"]
        return None

    def _write_jsonl(diag):
        if not jsonl_path:
            return
        compact = {k: v for k, v in diag.items() if k != "probes"}
        compact["failed_probes"] = _extract_failed_probes(diag)
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(compact, ensure_ascii=False) + "\n")

    def _append_frp_file(prefix_str):
        if not frp_output_path:
            return
        with open(frp_output_path, 'a') as f:
            f.write("%s\n" % prefix_str)

    def _emit_quiet_progress():
        """Single-line overwriting progress for --quiet mode.

        Caller must hold state_lock (reads shared counters).
        """
        if not quiet:
            return
        elapsed_s = time.time() - start_time
        total_input = len(prefixes)
        line = (
            "[Progress] %d/%d probed | %d FRP | %d probes (%d det/%d ref/%d ta) | %s"
            % (
                processed_count,
                total_input,
                len(confirmed_frps),
                total_probes,
                detection_probes,
                refinement_probes,
                ta_probes,
                _format_duration(elapsed_s),
            )
        )
        # Pad to clear any leftover chars from previous longer line
        pad = max(0, _quiet_last_len[0] - len(line))
        sys.stdout.write("\r" + line + (" " * pad))
        sys.stdout.flush()
        _quiet_last_len[0] = len(line)

    # ------------------------------------------------------------------
    # Per-prefix worker (no shared-state mutation inside!)
    # ------------------------------------------------------------------
    def _process_one(prefix):
        """
        Run the full detect [+ refine | + tree_align] pipeline for a single
        prefix. Returns a dict describing the outcome; the caller (main
        thread) applies the outcome to shared state under lock.
        """
        outcome = {
            "prefix": prefix,
            "diag": None,
            "is_frp": False,
            "refined": prefix,
            "refinement_info": None,
            "ta_result": None,
            "probes_used": 0,
            "detection_probes": 0,
            "refinement_probes": 0,
            "ta_probes": 0,
            "error": None,
        }

        try:
            diag = diagnose_prefix(
                prefix,
                timeout=timeout,
                max_attempts=max_attempts,
                concurrency=concurrency,
            )
        except Exception as e:
            outcome["error"] = "diagnose failed: %s" % e
            return outcome

        outcome["diag"] = diag
        outcome["detection_probes"] = diag["total_probes_sent"]
        outcome["probes_used"] += diag["total_probes_sent"]

        is_frp, reason = determine_frp_status(diag)
        diag["frp_decision"] = {"is_frp": is_frp, "reason": reason}
        outcome["is_frp"] = is_frp

        # Build a live coverage_checker that consults the *current*
        # confirmed_frps set (updated by parallel workers) under the shared
        # lock. Passed to refine + tree_align so they can early-stop when a
        # broader FRP has already been established elsewhere.
        def _live_coverage_checker(prefix_net):
            with state_lock:
                return _is_covered_by_any_locked(prefix_net)

        if is_frp:
            if refine:
                try:
                    ref = refine_frp_prefix(
                        prefix,
                        timeout=timeout,
                        max_attempts=max_attempts,
                        min_prefix_len=min_prefix_len,
                        concurrency=concurrency,
                        fast_refine=fast_refine,
                        refine_mode=refine_mode,
                        refine_timeout=refine_timeout,
                        refine_max_attempts=refine_max_attempts,
                        coverage_checker=_live_coverage_checker,
                    )
                    outcome["refined"] = ref["refined"]
                    outcome["refinement_info"] = ref
                    outcome["refinement_probes"] = ref["total_refinement_probes"]
                    outcome["probes_used"] += ref["total_refinement_probes"]
                    diag["refinement"] = ref
                except Exception as e:
                    outcome["error"] = "refine failed: %s" % e
        else:
            if tree_align and diag["reachable_count"] > 0:
                # tree_align_prefix needs a "known_frps" set for dedup;
                # we snapshot the current confirmed set (best-effort dedup).
                # The coverage_checker will keep this fresh across probes.
                with state_lock:
                    known_snapshot = set(f["prefix"] for f in confirmed_frps)
                try:
                    ta_res = tree_align_prefix(
                        prefix,
                        diag,
                        known_snapshot,
                        timeout=timeout,
                        max_attempts=max_attempts,
                        min_prefix_len=min_prefix_len,
                        concurrency=concurrency,
                        refine=refine,
                        max_depth=max_ta_depth,
                        fast_refine=fast_refine,
                        refine_mode=refine_mode,
                        refine_timeout=refine_timeout,
                        refine_max_attempts=refine_max_attempts,
                        coverage_checker=_live_coverage_checker,
                    )
                    outcome["ta_result"] = ta_res
                    outcome["ta_probes"] = ta_res["total_probes"]
                    outcome["probes_used"] += ta_res["total_probes"]
                    diag["tree_alignment"] = ta_res
                except Exception as e:
                    outcome["error"] = "tree_align failed: %s" % e

        return outcome

    # ------------------------------------------------------------------
    # Outcome-merger: applies a completed outcome to shared state
    # ------------------------------------------------------------------
    def _merge_outcome(outcome):
        nonlocal processed_count, total_probes
        nonlocal detection_probes, refinement_probes, ta_probes
        nonlocal ta_candidates_checked, ta_new_frps

        prefix = outcome["prefix"]
        diag = outcome["diag"]
        if diag is None:
            # Diagnose itself failed; log and drop
            if not quiet:
                logger.warning("[ERROR] %s: %s", prefix, outcome.get("error"))
            return

        with state_lock:
            processed_count += 1
            local_idx = processed_count
            detection_probes += outcome["detection_probes"]
            refinement_probes += outcome["refinement_probes"]
            ta_probes += outcome["ta_probes"]
            total_probes += outcome["probes_used"]

            reach_str = "%d/16" % diag["reachable_count"]

            if outcome["is_frp"]:
                refined_str = outcome["refined"]
                try:
                    refined_net = ipaddress.IPv6Network(refined_str, strict=False)
                except Exception:
                    refined_net = None

                # If a broader confirmed FRP already covers us, drop.
                dropped_covered_by = None
                if refined_net is not None:
                    dropped_covered_by = _is_covered_by_any_locked(refined_net)

                if dropped_covered_by:
                    skipped.append({
                        "prefix": prefix,
                        "covered_by": dropped_covered_by,
                        "post_hoc": True,
                    })
                    if not quiet:
                        print("%-5d %-42s %-10s %-8s %s" % (
                            local_idx, prefix, "~drop", reach_str,
                            "covered by %s" % dropped_covered_by
                        ))
                        sys.stdout.flush()
                else:
                    # Remove any narrower existing FRPs covered by this one
                    if refined_net is not None:
                        kept = []
                        for f in confirmed_frps:
                            try:
                                f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
                            except Exception:
                                kept.append(f)
                                continue
                            if not is_covered_by(f_net, refined_net):
                                kept.append(f)
                        confirmed_frps[:] = kept
                        confirmed_frps.append({
                            "prefix": str(refined_net),
                            "original": prefix,
                            "source": "direct",
                            "refinement": outcome["refinement_info"],
                        })
                        _append_frp_file(str(refined_net))

                    if not quiet:
                        refined_display = ""
                        if refine and refined_str != prefix:
                            refined_display = " -> %s" % refined_str
                        print("%-5d %-42s %-10s %-8s %s" % (
                            local_idx, prefix, "✓ FRP", reach_str, refined_display
                        ))
                        sys.stdout.flush()

                all_results.append(diag)
                _write_jsonl(diag)

            else:
                # Not FRP
                if not quiet:
                    reason = diag.get("frp_decision", {}).get("reason", "")
                    print("%-5d %-42s %-10s %-8s %s" % (
                        local_idx, prefix, "NOT FRP", reach_str, reason[:40]
                    ))
                    sys.stdout.flush()

                # Merge tree-alignment new FRPs, if any
                ta_res = outcome.get("ta_result")
                if ta_res is not None:
                    ta_candidates_checked += len(ta_res["candidates"])
                    ta_results.append(ta_res)

                    for new_frp in ta_res["new_frps"]:
                        new_prefix = new_frp["prefix"]
                        try:
                            new_net = ipaddress.IPv6Network(new_prefix, strict=False)
                        except Exception:
                            continue

                        cov = _is_covered_by_any_locked(new_net)
                        if cov:
                            skipped.append({
                                "prefix": new_prefix,
                                "covered_by": cov,
                                "post_hoc": True,
                                "source": "tree_align",
                            })
                            continue

                        # Remove narrower ones
                        kept = []
                        for f in confirmed_frps:
                            try:
                                f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
                            except Exception:
                                kept.append(f)
                                continue
                            if not is_covered_by(f_net, new_net):
                                kept.append(f)
                        confirmed_frps[:] = kept
                        confirmed_frps.append({
                            "prefix": new_prefix,
                            "original": new_frp.get("original_candidate", new_prefix),
                            "source": "tree_align",
                            "source_prefix": prefix,
                        })
                        ta_new_frps += 1
                        _append_frp_file(new_prefix)

                        if not quiet:
                            print("  %-3s %-42s %-10s %-8s %s" % (
                                "↳", new_prefix, "✓ TA-FRP", "",
                                "from %s" % prefix
                            ))
                            sys.stdout.flush()

                all_results.append(diag)
                _write_jsonl(diag)

            # Emit progress line while still under lock (fresh counters)
            _emit_quiet_progress()

    # Combine max_probes with tune_probe_limit (take the tighter of the two)
    effective_probe_cap = max_probes
    if tune_probe_limit is not None:
        if effective_probe_cap is None or tune_probe_limit < effective_probe_cap:
            effective_probe_cap = tune_probe_limit

    tune_deadline = None
    if tune_time_limit is not None:
        tune_deadline = start_time + tune_time_limit

    # ------------------------------------------------------------------
    # Submit / drain loop
    # ------------------------------------------------------------------
    remaining = list(prefixes)  # input order for submission
    in_flight = {}  # future -> prefix

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:

        def _try_submit_next():
            """Try to submit the next eligible prefix (skip covered ones)."""
            nonlocal submitted_count, stop_submitting
            while remaining:
                # max_probes gate
                with state_lock:
                    if effective_probe_cap is not None and total_probes >= effective_probe_cap:
                        stop_submitting = True
                        return False
                # tune-mode time gate
                if tune_deadline is not None and time.time() >= tune_deadline:
                    stop_submitting = True
                    return False

                prefix = remaining.pop(0)

                # Parse
                try:
                    prefix_net = ipaddress.IPv6Network(prefix, strict=False)
                except Exception as e:
                    if not quiet:
                        logger.warning("[SKIP] Invalid prefix %s: %s", prefix, e)
                    continue

                # Pre-submit coverage filter (best-effort against currently
                # confirmed FRPs; may still race, but re-checked in merge)
                with state_lock:
                    cov = _is_covered_by_any_locked(prefix_net)
                if cov:
                    with state_lock:
                        skipped.append({"prefix": prefix, "covered_by": cov})
                    continue

                fut = pool.submit(_process_one, prefix)
                in_flight[fut] = prefix
                submitted_count += 1
                return True
            return False

        # Prime the pool
        for _ in range(max(1, workers)):
            if not _try_submit_next():
                break

        # Drain + submit-more loop
        while in_flight:
            done_set, _ = _wait_any(list(in_flight.keys()))
            for fut in done_set:
                prefix = in_flight.pop(fut)
                try:
                    outcome = fut.result()
                except Exception as e:
                    outcome = {
                        "prefix": prefix,
                        "diag": None,
                        "is_frp": False,
                        "error": "worker exception: %s" % e,
                        "probes_used": 0,
                        "detection_probes": 0,
                        "refinement_probes": 0,
                        "ta_probes": 0,
                        "refined": prefix,
                        "refinement_info": None,
                        "ta_result": None,
                    }
                _merge_outcome(outcome)

                # Submit next (may be blocked by max_probes)
                _try_submit_next()

    elapsed = time.time() - start_time

    # Terminate quiet-mode progress line with a newline before final summary
    if quiet and _quiet_last_len[0] > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    # Footer
    if not quiet:
        print("%-5s %-42s %-10s %-8s %s" % ("-" * 5, "-" * 42, "-" * 10, "-" * 8, "-" * 30))
        print("Done. %d FRPs confirmed from %d probed (%d skipped). Time: %.1fs"
              % (len(confirmed_frps), processed_count, len(skipped), elapsed))
        print("Total probes: %d (detection: %d, refinement: %d, tree-align: %d)"
              % (total_probes, detection_probes, refinement_probes, ta_probes))
        if tree_align:
            print("Tree alignment: %d candidates checked, %d new FRPs found"
                  % (ta_candidates_checked, ta_new_frps))
        if stop_submitting:
            print("NOTE: stopped early after reaching max_probes=%d" % max_probes)
        print("=" * 78)
        sys.stdout.flush()

    # Stats
    surviving_seeds = sum(1 for f in confirmed_frps if f.get("source") == "seed")
    new_direct_frps = sum(1 for f in confirmed_frps if f.get("source") == "direct")
    new_ta_frps = sum(1 for f in confirmed_frps if f.get("source") == "tree_align")
    stats = {
        "total_input": len(prefixes),
        "processed": processed_count,
        "submitted": submitted_count,
        "skipped_covered": len(skipped),
        "confirmed_frps": len(confirmed_frps),
        "non_frps": processed_count - new_direct_frps,
        "elapsed_seconds": round(elapsed, 1),
        "avg_seconds_per_prefix": round(elapsed / max(processed_count, 1), 1),
        "total_probes_sent": total_probes,
        "detection_probes": detection_probes,
        "refinement_probes": refinement_probes,
        "tree_align_probes": ta_probes,
        "tree_align_candidates_checked": ta_candidates_checked,
        "tree_align_new_frps": ta_new_frps,
        "tree_align_success_rate": round(
            ta_new_frps / max(ta_candidates_checked, 1) * 100, 1
        ),
        "workers": workers,
        "max_probes": max_probes,
        "stopped_early": stop_submitting,
        "seed_frps_provided": initial_seed_count,
        "seed_frps_surviving": surviving_seeds,
        "seed_frps_evicted_by_broader": initial_seed_count - surviving_seeds,
        "new_direct_frps": new_direct_frps,
        "new_tree_align_frps": new_ta_frps,
        "probes_per_second": round(total_probes / max(elapsed, 0.001), 1),
        "processed_prefixes_remaining": len(remaining),
    }

    return {
        "confirmed_frps": [f["prefix"] for f in confirmed_frps],
        "confirmed_frps_detail": confirmed_frps,
        "all_results": all_results,
        "tree_alignment_results": ta_results,
        "skipped": skipped,
        "stats": stats,
        "unprocessed": list(remaining),
    }


def _wait_any(futures):
    """
    Wait until at least one future completes. Returns (done_set, not_done_set).
    Thin wrapper so we can keep the main loop concise.
    """
    from concurrent.futures import wait, FIRST_COMPLETED
    return wait(futures, return_when=FIRST_COMPLETED)


# ============================================================================
# Auto-tune: sweep workers on a small sample to find the throughput optimum
# ============================================================================

def auto_tune_workers(
    prefixes,
    candidate_workers,
    per_stage_seconds,
    per_stage_max_probes,
    sample_size_per_stage,
    pipeline_kwargs,
    output_path,
    frp_output_path,
    quiet,
):
    """Run several short pipeline invocations with different ``workers`` values
    over disjoint slices of ``prefixes``, then return:

        (best_workers, tune_summary, confirmed_seed_frps, prefixes_left, tune_results)

    Design decisions:
      * Each stage consumes a *disjoint* chunk of the input candidates so
        no candidate is probed twice. Any FRP found during tune stages is
        threaded into the final stage as a pre-loaded seed (via the
        pipeline's ``seed_frps`` argument).
      * Each stage stops when either ``per_stage_seconds`` elapses OR the
        stage-local probe budget (``per_stage_max_probes``) is hit OR the
        stage-local sample is exhausted.
      * We score stages by ``probes_per_second`` (throughput). Best worker
        count is the arg-max. Ties: prefer smaller workers (safer).
    """
    idx = 0
    total = len(prefixes)
    tune_stage_summaries = []
    seeds_from_tune = []       # list[str]
    tune_all_results = []      # list of all_results from each stage
    tune_all_ta_results = []
    tune_skipped = []
    tune_totals = {
        "processed": 0,
        "total_probes_sent": 0,
        "detection_probes": 0,
        "refinement_probes": 0,
        "tree_align_probes": 0,
        "tree_align_candidates_checked": 0,
        "tree_align_new_frps": 0,
        "elapsed_seconds": 0.0,
        "confirmed_frps_from_tune": 0,
    }

    if not quiet:
        print("\n" + "=" * 78)
        print("AUTO-TUNE: sweeping workers on a small sample to find best throughput")
        print("  candidate workers: %s" % candidate_workers)
        print("  per-stage limit:   %ds  OR  %d probes  OR  %d prefixes"
              % (per_stage_seconds, per_stage_max_probes, sample_size_per_stage))
        print("=" * 78)
        sys.stdout.flush()

    for w in candidate_workers:
        if idx >= total:
            if not quiet:
                logger.warning("Auto-tune: ran out of candidates before "
                               "sweeping workers=%d", w)
            break
        chunk = prefixes[idx: idx + sample_size_per_stage]
        idx += len(chunk)

        # Include any FRPs already discovered by earlier tune stages as seeds
        # so we don't re-probe candidates they cover.
        stage_seeds = list(seeds_from_tune)

        # Build kwargs for the stage, overriding workers + probe/time limits.
        stage_kwargs = dict(pipeline_kwargs)
        stage_kwargs.update(dict(
            workers=w,
            max_probes=None,          # we use tune_probe_limit instead
            tune_time_limit=per_stage_seconds,
            tune_probe_limit=per_stage_max_probes,
            seed_frps=stage_seeds if stage_seeds else None,
            incremental_mode=True,    # keep prior tune-stage output on disk
            output_path=output_path,
            frp_output_path=frp_output_path,
            quiet=quiet,
        ))

        if not quiet:
            print("\n[tune] workers=%d  sample=%d  ..." % (w, len(chunk)))
            sys.stdout.flush()

        t0 = time.time()
        result = run_apd_pipeline(chunk, **stage_kwargs)
        stage_elapsed = time.time() - t0

        stats = result["stats"]
        probes = stats["total_probes_sent"]
        pps = probes / max(stage_elapsed, 1e-3)

        tune_stage_summaries.append({
            "workers": w,
            "sample_prefixes": len(chunk),
            "processed": stats["processed"],
            "total_probes": probes,
            "elapsed_seconds": round(stage_elapsed, 2),
            "probes_per_second": round(pps, 1),
            "new_frps": stats["new_direct_frps"] + stats["new_tree_align_frps"],
            "stopped_early": stats["stopped_early"],
        })

        # Roll into the aggregate tune totals for the final report.
        tune_totals["processed"] += stats["processed"]
        tune_totals["total_probes_sent"] += stats["total_probes_sent"]
        tune_totals["detection_probes"] += stats["detection_probes"]
        tune_totals["refinement_probes"] += stats["refinement_probes"]
        tune_totals["tree_align_probes"] += stats["tree_align_probes"]
        tune_totals["tree_align_candidates_checked"] += stats["tree_align_candidates_checked"]
        tune_totals["tree_align_new_frps"] += stats["tree_align_new_frps"]
        tune_totals["elapsed_seconds"] += stage_elapsed

        # Merge FRPs discovered in this stage into the running seed set.
        # Note: run_apd_pipeline already emitted them to _frp.txt / .jsonl
        # because we asked for incremental_mode=True.
        for f in result["confirmed_frps_detail"]:
            # Skip re-adding entries that were themselves seeds (already in the pool).
            if f.get("source") == "seed":
                continue
            if f["prefix"] not in seeds_from_tune:
                seeds_from_tune.append(f["prefix"])

        # Collect stage-level artifacts so final aggregate output is complete.
        tune_all_results.extend(result["all_results"])
        tune_all_ta_results.extend(result["tree_alignment_results"])
        tune_skipped.extend(result["skipped"])

        # If this stage didn't consume all its sample (e.g., time limit hit
        # or probe cap hit), push the leftovers back so subsequent tune
        # stages (or the final stage) still get a shot at them.
        #
        # Layout before:
        #   prefixes = [ ...done..., <-- chunk (len=C) -->, ...future... ]
        #                             ^ old_idx           ^ idx
        # We drop the fully-consumed part of `chunk` (C - L items) and
        # splice the L leftover items just after old_idx, then rewind idx
        # so they are visible to future stages.
        leftover = result.get("unprocessed", [])
        if leftover:
            old_idx = idx - len(chunk)
            prefixes = prefixes[:old_idx] + leftover + prefixes[idx:]
            idx = old_idx  # leftover starts here; next stage will pick from it

        if not quiet:
            print("[tune] workers=%d done: %d prefixes, %d probes, %.1fs, "
                  "%.1f probes/s, %d new FRPs%s"
                  % (w, stats["processed"], probes, stage_elapsed, pps,
                     stats["new_direct_frps"] + stats["new_tree_align_frps"],
                     " (stopped early)" if stats["stopped_early"] else ""))
            sys.stdout.flush()

    # Pick best workers: max probes_per_second; on ties, prefer smaller workers.
    if not tune_stage_summaries:
        return None, {
            "stages": [],
            "best_workers": None,
            "aggregate": tune_totals,
        }, seeds_from_tune, prefixes[idx:], {
            "all_results": tune_all_results,
            "tree_alignment_results": tune_all_ta_results,
            "skipped": tune_skipped,
        }

    best_stage = None
    for s in tune_stage_summaries:
        # Require at least some probes to be a valid data point.
        if s["total_probes"] <= 0:
            continue
        if (best_stage is None
                or s["probes_per_second"] > best_stage["probes_per_second"]
                or (s["probes_per_second"] == best_stage["probes_per_second"]
                    and s["workers"] < best_stage["workers"])):
            best_stage = s

    best_workers = best_stage["workers"] if best_stage else tune_stage_summaries[0]["workers"]
    tune_totals["confirmed_frps_from_tune"] = len(seeds_from_tune)

    if not quiet:
        print("\n" + "-" * 78)
        print("AUTO-TUNE result: best workers = %d  (%.1f probes/s over %d probes)"
              % (best_workers,
                 best_stage["probes_per_second"] if best_stage else 0.0,
                 best_stage["total_probes"] if best_stage else 0))
        print("Stage-by-stage:")
        for s in tune_stage_summaries:
            marker = "  * " if s is best_stage else "    "
            print("%sworkers=%3d | %6d probes | %6.1fs | %8.1f probes/s | %d FRPs%s"
                  % (marker, s["workers"], s["total_probes"], s["elapsed_seconds"],
                     s["probes_per_second"], s["new_frps"],
                     "  (early stop)" if s["stopped_early"] else ""))
        print("-" * 78)
        sys.stdout.flush()

    tune_summary = {
        "stages": tune_stage_summaries,
        "best_workers": best_workers,
        "aggregate": tune_totals,
    }
    remaining_after_tune = prefixes[idx:]
    tune_artifacts = {
        "all_results": tune_all_results,
        "tree_alignment_results": tune_all_ta_results,
        "skipped": tune_skipped,
    }
    return best_workers, tune_summary, seeds_from_tune, remaining_after_tune, tune_artifacts


# ============================================================================
# CLI
# ============================================================================

def load_prefixes(filepath):
    """Load prefixes from file (one per line, # = comment)."""
    prefixes = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                prefixes.append(parts[0])
    logger.info("Loaded %d prefixes from %s", len(prefixes), filepath)
    return prefixes


def load_seed_frps(filepath):
    """Load already-known FRP seeds from file (one prefix per line, # = comment).

    Returns a list of ipaddress.IPv6Network. Invalid lines are skipped
    with a warning.
    """
    seed_nets = []
    if not filepath:
        return seed_nets
    if not os.path.exists(filepath):
        logger.warning("Seed FRP file not found: %s (skipping seed filter)", filepath)
        return seed_nets
    with open(filepath, 'r') as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            token = line.split()[0]
            try:
                net = ipaddress.IPv6Network(token, strict=False)
            except Exception as e:
                logger.warning("Seed FRP file %s line %d: invalid prefix %r (%s)",
                               filepath, lineno, token, e)
                continue
            seed_nets.append(net)
    logger.info("Loaded %d seed FRPs from %s", len(seed_nets), filepath)
    return seed_nets


def filter_by_seed_frps(prefixes, seed_nets):
    """Drop candidate prefixes that are covered by (or equal to) any seed FRP.

    Coverage rule: candidate is filtered iff there exists a seed S such that
    candidate is a subnet of S (i.e. candidate.prefixlen >= S.prefixlen and
    candidate lies within S). Candidates that are SHORTER than every seed
    (i.e. candidates that themselves cover a seed) are KEPT.

    Returns:
        (kept_prefixes, dropped_info)
            kept_prefixes: list[str]
            dropped_info:  list[{"prefix": str, "covered_by": str}]
    """
    if not seed_nets:
        return list(prefixes), []

    kept = []
    dropped = []
    for p in prefixes:
        try:
            cand_net = ipaddress.IPv6Network(p, strict=False)
        except Exception as e:
            logger.warning("Invalid candidate prefix %r skipped in seed filter: %s", p, e)
            kept.append(p)  # let downstream re-report the error
            continue

        cov = None
        for s in seed_nets:
            if is_covered_by(cand_net, s):
                cov = s
                break

        if cov is not None:
            dropped.append({"prefix": p, "covered_by": str(cov)})
        else:
            kept.append(p)

    return kept, dropped


def print_summary(result, quiet=False):
    """Print human-readable summary to stdout.

    When quiet=True, only the most essential information is printed:
    total probes (with breakdown) and the final confirmed FRP list.
    """
    stats = result["stats"]
    frps = result["confirmed_frps"]
    all_results = result["all_results"]

    if quiet:
        print("\n" + "=" * 60)
        print("APD RESULT (quiet)")
        print("=" * 60)
        print("Total probes sent:   %d" % stats["total_probes_sent"])
        print("  detection:         %d" % stats["detection_probes"])
        print("  refinement:        %d" % stats["refinement_probes"])
        print("  tree-align:        %d" % stats["tree_align_probes"])
        if stats.get("stopped_early"):
            print("Stopped early:       max_probes=%s reached"
                  % stats.get("max_probes"))
        if stats.get("seed_frps_provided", 0) > 0:
            print("Seeds provided:      %d (surviving: %d, new-direct: %d, new-ta: %d)" % (
                stats.get("seed_frps_provided", 0),
                stats.get("seed_frps_surviving", 0),
                stats.get("new_direct_frps", 0),
                stats.get("new_tree_align_frps", 0),
            ))
        print("Confirmed FRPs (%d):" % len(frps))
        for p in frps:
            print("  %s" % p)
        print("=" * 60)
        return

    print("\n" + "=" * 78)
    print("APD DETECTION SUMMARY")
    print("=" * 78)
    print("Total input prefixes:      %d" % stats["total_input"])
    print("Processed (probed):        %d" % stats["processed"])
    print("Skipped (covered):         %d" % stats["skipped_covered"])
    print("Confirmed FRPs:            %d" % stats["confirmed_frps"])
    print("  Direct FRPs:             %d" % len([
        f for f in result["confirmed_frps_detail"] if f.get("source") == "direct"
    ]))
    print("  Tree-align FRPs:         %d" % stats["tree_align_new_frps"])
    if stats.get("seed_frps_provided", 0) > 0:
        print("  Seed FRPs (surviving):   %d / %d provided (%d evicted by broader)" % (
            stats.get("seed_frps_surviving", 0),
            stats.get("seed_frps_provided", 0),
            stats.get("seed_frps_evicted_by_broader", 0),
        ))
    print("Non-FRPs:                  %d" % stats["non_frps"])
    print("Total time:                %.1fs" % stats["elapsed_seconds"])
    print("Avg per prefix:            %.1fs" % stats["avg_seconds_per_prefix"])
    print("Workers:                   %d" % stats.get("workers", 1))
    if stats.get("stopped_early"):
        print("Stopped early:             max_probes=%s reached"
              % stats.get("max_probes"))
    print("")
    print("Probe statistics:")
    print("  Total probes sent:       %d" % stats["total_probes_sent"])
    print("  Detection probes:        %d" % stats["detection_probes"])
    print("  Refinement probes:       %d" % stats["refinement_probes"])
    print("  Tree-align probes:       %d" % stats["tree_align_probes"])
    if stats["tree_align_candidates_checked"] > 0:
        print("")
        print("Tree alignment stats:")
        print("  Candidates checked:      %d" % stats["tree_align_candidates_checked"])
        print("  New FRPs found:          %d" % stats["tree_align_new_frps"])
        print("  Success rate:            %.1f%%" % stats["tree_align_success_rate"])

    # Per-prefix decisions
    print("\nPer-prefix results:")
    for r in all_results:
        d = r.get("frp_decision", {})
        mark = "+" if d.get("is_frp") else "-"
        prefix = r.get("prefix", "?")
        reach = "%d/16" % r.get("reachable_count", 0)
        print("  [%s] %-42s %s" % (mark, prefix, reach))
        if d.get("is_frp") and "refinement" in r:
            ref = r["refinement"]
            print("       Refined: /%d -> /%d (%s)"
                  % (ref["original_len"], ref["refined_len"], ref["refined"]))
        if "tree_alignment" in r:
            ta = r["tree_alignment"]
            if ta["new_frps"]:
                for nf in ta["new_frps"]:
                    print("       TA-FRP: %s" % nf["prefix"])

    # Final FRP list
    if frps:
        print("\nFinal confirmed FRPs (%d):" % len(frps))
        for p in frps:
            print("  %s" % p)

    # Skipped
    if result["skipped"]:
        print("\nSkipped (covered, first 20):")
        for s in result["skipped"][:20]:
            print("  %s <- covered by %s" % (s["prefix"], s["covered_by"]))
        if len(result["skipped"]) > 20:
            print("  ... and %d more" % (len(result["skipped"]) - 20))

    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description='APD Detection with Ping-style Probing + Tree Alignment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic detection
  sudo python3 standalone_apd.py -i prefixes.txt -o result.json

  # Detection + refinement
  sudo python3 standalone_apd.py -i prefixes.txt -o result.json --refine

  # Detection + tree alignment
  sudo python3 standalone_apd.py -i prefixes.txt -o result.json --ta

  # Full pipeline with parallel workers
  sudo python3 standalone_apd.py -i prefixes.txt -o result.json \\
      --refine --ta --workers 16

  # Quiet mode + hard probe budget
  sudo python3 standalone_apd.py -i prefixes.txt -o result.json \\
      --refine --ta --workers 16 --quiet --max-probes 500000

  # Quick test
  sudo python3 standalone_apd.py -i prefixes.txt -o result.json --limit 10
        """
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input prefix file (one per line)')
    parser.add_argument('-o', '--output', default='apd_result.json',
                        help='Output JSON file (default: apd_result.json)')
    parser.add_argument('--refine', action='store_true',
                        help='Enable granularity alignment (find shortest FRP prefix)')
    parser.add_argument('--fast-refine', action='store_true',
                        help='Legacy alias for --refine-mode=fast (kept for backward '
                             'compatibility). Ignored if --refine-mode is given.')
    parser.add_argument('--refine-mode',
                        choices=['bisect', 'linear', 'fast'],
                        default=fast,
                        help='Refinement search strategy. '
                             '"bisect" (recommended, fastest): binary search on '
                             'prefix length, ~log2(N) diagnoses. '
                             '"linear": precise 1-bit steps (old default). '
                             '"fast": 4-bit jumps then 1-bit fine-tuning. '
                             'If unset, defaults to "bisect" (or "fast" when '
                             '--fast-refine is given, for backward compat).')
    parser.add_argument('--refine-timeout', type=float, default=None,
                        help='Per-attempt timeout (seconds) used ONLY during '
                             'refinement / tree-align refine. Falls back to '
                             '--timeout when unset. Setting this smaller than '
                             '--timeout (e.g. 1.0) can significantly speed up '
                             'refinement without affecting detection quality.')
    parser.add_argument('--refine-max-attempts', type=int, default=None,
                        help='Max ping attempts per sub-prefix used ONLY during '
                             'refinement / tree-align refine. Falls back to '
                             '--max-attempts when unset. Lowering to 2 or 3 is '
                             'usually safe for refinement.')
    parser.add_argument('--ta', action='store_true',
                        help='Enable tree-based alignment (find sub-FRPs from partial prefixes)')
    parser.add_argument('--max-ta-depth', type=int, default=4,
                        help='Max recursion depth for tree alignment (default: 4)')
    parser.add_argument('--timeout', type=float, default=2.0,
                        help='Per-attempt timeout in seconds (default: 2.0)')
    parser.add_argument('--max-attempts', type=int, default=5,
                        help='Max ping attempts per sub-prefix (default: 5)')
    parser.add_argument('--concurrency', type=int, default=16,
                        help='Parallel probes within a prefix (default: 16)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel prefix-level workers (default: 8). '
                             'Total socket concurrency ~= workers * --concurrency.')
    parser.add_argument('--min-prefix-len', type=int, default=16,
                        help='Minimum prefix length for refinement (default: 16)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process first N prefixes')
    parser.add_argument('--frp-output', default=None,
                        help='FRP list output file (default: <output>_frp.txt)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress real-time per-prefix output. Only print '
                             'total probe counts (with breakdown) and the final '
                             'confirmed FRP list at the end.')
    parser.add_argument('--max-probes', type=int, default=None,
                        help='Hard cap on total probes sent. Once reached, no '
                             'new prefixes are submitted; already-running tasks '
                             'finish. Unset by default.')
    parser.add_argument('--auto-tune', action='store_true',
                        help='Before running the full pipeline, sweep several '
                             'worker counts on small disjoint slices of the '
                             'input to find the workers value with the highest '
                             'probes-per-second, then use it for the remaining '
                             'candidates. Tune-stage results (probes + FRPs) '
                             'are kept, not thrown away.')
    parser.add_argument('--tune-workers', default='8,16,32,48,64',
                        help='Comma-separated worker counts to sweep in '
                             '--auto-tune mode (default: 8,16,32,48,64).')
    parser.add_argument('--tune-stage-seconds', type=int, default=30,
                        help='Max seconds spent per tune stage (default: 30).')
    parser.add_argument('--tune-stage-probes', type=int, default=20000,
                        help='Max probes per tune stage (default: 20000). '
                             'Each stage stops at whichever limit hits first.')
    parser.add_argument('--tune-stage-sample', type=int, default=200,
                        help='Max prefixes per tune stage (default: 200).')
    parser.add_argument('--seed-frps', default=None,
                        help='Path to a known-FRP seed file (one prefix per line). '
                             'If provided and file exists, candidates that are '
                             'covered by (or equal to) any seed are filtered out '
                             'BEFORE probing starts. Candidates that are shorter '
                             'than every seed are kept. See also --seed-mode.')
    parser.add_argument('--seed-mode',
                        choices=['filter', 'include'],
                        default='filter',
                        help='How seeds interact with the confirmed-FRP set. '
                             '"filter" (default, option A): seeds are only used '
                             'as a pre-probe filter and never appear in outputs. '
                             '"include" (option B): seeds also become part of '
                             'the initial confirmed FRP set, participate in '
                             'coverage dedup (new sub-FRPs covered by a seed '
                             'are dropped without extra work), and appear in '
                             'the final _frp.txt / .jsonl / _summary.json. '
                             'If a newly-confirmed broader FRP covers a seed, '
                             'that seed is evicted from the final set.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose logging (DEBUG level)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        # Quiet mode: hide per-prefix INFO log spam from apd_core, keep warnings+
        logging.getLogger().setLevel(logging.WARNING)

    # Resolve refine mode: explicit --refine-mode wins; then --fast-refine;
    # else default to "bisect" (fastest, always-safe on monotone FRP).
    effective_refine_mode = args.refine_mode
    if effective_refine_mode is None:
        effective_refine_mode = "fast" if args.fast_refine else "bisect"

    # Root check
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        logger.error("Need root for raw ICMPv6 sockets. Run with: sudo python3 standalone_apd.py ...")
        sys.exit(1)

    # Load prefixes
    prefixes = load_prefixes(args.input)
    if args.limit:
        prefixes = prefixes[:args.limit]
        logger.info("Limited to first %d prefixes", len(prefixes))

    if not prefixes:
        logger.error("No prefixes to process")
        sys.exit(1)

    # Optional: pre-filter candidates by known-FRP seeds (before any probing)
    seed_dropped = []
    seed_prefixes_for_pipeline = None   # only populated in --seed-mode=include
    if args.seed_frps:
        seed_nets = load_seed_frps(args.seed_frps)
        if seed_nets:
            before = len(prefixes)
            prefixes, seed_dropped = filter_by_seed_frps(prefixes, seed_nets)
            logger.info(
                "Seed filter: %d/%d candidates dropped (covered by seeds); %d remain",
                len(seed_dropped), before, len(prefixes),
            )
            # In include mode (option B) we ALSO pass the seed list into the
            # pipeline so it participates in coverage dedup + final output.
            if args.seed_mode == 'include':
                seed_prefixes_for_pipeline = [str(n) for n in seed_nets]
                logger.info(
                    "Seed mode = include: %d seeds will be pre-loaded into "
                    "confirmed_frps", len(seed_prefixes_for_pipeline),
                )
            if not prefixes:
                if args.seed_mode == 'include' and seed_prefixes_for_pipeline:
                    logger.warning(
                        "All candidates were covered by seed FRPs; nothing to "
                        "probe. Will still emit outputs containing the seeds."
                    )
                    # Fall through: pipeline will run with 0 prefixes but the
                    # seeded confirmed set will be written to outputs.
                else:
                    logger.error("All candidates were covered by seed FRPs; nothing to probe")
                    sys.exit(0)

    # Prepare output paths
    frp_txt_path = args.frp_output
    if not frp_txt_path:
        base = args.output.rsplit('.', 1)[0] if '.' in args.output else args.output
        frp_txt_path = base + "_frp.txt"
    summary_path = (
        args.output.rsplit('.', 1)[0] + "_summary.json"
        if '.' in args.output else args.output + "_summary.json"
    )

    # ------------------------------------------------------------------
    # Auto-tune stage (optional): find the best "workers" value before
    # committing the whole input to a single choice.
    # ------------------------------------------------------------------
    tune_summary = None
    tune_artifacts = None
    effective_workers = args.workers
    seed_from_tune = []
    remaining_prefixes = prefixes

    if args.auto_tune:
        try:
            candidate_workers = [int(x) for x in args.tune_workers.split(',') if x.strip()]
        except ValueError:
            logger.error("--tune-workers must be a comma-separated list of ints, "
                         "got: %r", args.tune_workers)
            sys.exit(1)
        if not candidate_workers:
            logger.error("--tune-workers cannot be empty")
            sys.exit(1)

        # Shared kwargs for both tune stages and the final stage.
        base_pipeline_kwargs = dict(
            refine=args.refine,
            tree_align=args.ta,
            timeout=args.timeout,
            max_attempts=args.max_attempts,
            concurrency=args.concurrency,
            min_prefix_len=args.min_prefix_len,
            max_ta_depth=args.max_ta_depth,
            fast_refine=args.fast_refine,
            refine_mode=effective_refine_mode,
            refine_timeout=args.refine_timeout,
            refine_max_attempts=args.refine_max_attempts,
        )

        best_w, tune_summary, seed_from_tune, remaining_prefixes, tune_artifacts = \
            auto_tune_workers(
                prefixes=prefixes,
                candidate_workers=candidate_workers,
                per_stage_seconds=args.tune_stage_seconds,
                per_stage_max_probes=args.tune_stage_probes,
                sample_size_per_stage=args.tune_stage_sample,
                pipeline_kwargs=base_pipeline_kwargs,
                output_path=args.output,
                frp_output_path=frp_txt_path,
                quiet=args.quiet,
            )
        if best_w is not None:
            effective_workers = best_w
            logger.info("Auto-tune selected workers=%d for the final stage "
                        "(%d prefixes remaining)",
                        effective_workers, len(remaining_prefixes))
        else:
            logger.warning("Auto-tune produced no data points; falling back to "
                           "--workers=%d", args.workers)

        # Merge tune seeds into the pipeline seed set (option-B style).
        if seed_from_tune:
            if seed_prefixes_for_pipeline is None:
                seed_prefixes_for_pipeline = []
            for p in seed_from_tune:
                if p not in seed_prefixes_for_pipeline:
                    seed_prefixes_for_pipeline.append(p)

    # Run pipeline
    result = run_apd_pipeline(
        remaining_prefixes,
        refine=args.refine,
        tree_align=args.ta,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        concurrency=args.concurrency,
        min_prefix_len=args.min_prefix_len,
        output_path=args.output,
        frp_output_path=frp_txt_path,
        max_ta_depth=args.max_ta_depth,
        fast_refine=args.fast_refine,
        workers=effective_workers,
        quiet=args.quiet,
        max_probes=args.max_probes,
        seed_frps=seed_prefixes_for_pipeline,
        incremental_mode=bool(args.auto_tune),
        refine_mode=effective_refine_mode,
        refine_timeout=args.refine_timeout,
        refine_max_attempts=args.refine_max_attempts,
    )

    # If we did auto-tune, splice tune-stage artifacts into the final result
    # so the aggregate outputs (summary + optional detailed JSON) reflect
    # everything that happened, not just the final stage.
    if args.auto_tune and tune_artifacts is not None:
        result["all_results"] = tune_artifacts["all_results"] + result["all_results"]
        result["tree_alignment_results"] = (
            tune_artifacts["tree_alignment_results"]
            + result["tree_alignment_results"]
        )
        result["skipped"] = tune_artifacts["skipped"] + result["skipped"]

        # Accumulate probe counters across tune + final stages.
        stats = result["stats"]
        agg = tune_summary["aggregate"]
        stats["processed"] += agg["processed"]
        stats["total_probes_sent"] += agg["total_probes_sent"]
        stats["detection_probes"] += agg["detection_probes"]
        stats["refinement_probes"] += agg["refinement_probes"]
        stats["tree_align_probes"] += agg["tree_align_probes"]
        stats["tree_align_candidates_checked"] += agg["tree_align_candidates_checked"]
        stats["tree_align_new_frps"] += agg["tree_align_new_frps"]
        stats["elapsed_seconds"] = round(stats["elapsed_seconds"] + agg["elapsed_seconds"], 1)
        stats["auto_tune"] = tune_summary
        stats["workers_final_stage"] = effective_workers
        stats["workers_requested"] = args.workers

    # Save detailed results (with failed probe addresses, without full probe data)
    # In --quiet mode we skip this big JSON; the same info is already in the
    # incremental .jsonl file, and the summary file below.
    if not args.quiet:
        output_data = []
        for r in result["all_results"]:
            compact = {k: v for k, v in r.items() if k != "probes"}
            compact["failed_probes"] = _extract_failed_probes(r)
            output_data.append(compact)
        output_wrapper = {
            "results": output_data,
            "elapsed_seconds": result["stats"]["elapsed_seconds"],
            "run_time_human": _format_duration(result["stats"]["elapsed_seconds"]),
        }
        with open(args.output, 'w') as f:
            json.dump(output_wrapper, f, indent=2, ensure_ascii=False)
        logger.info("Detailed results saved to: %s", args.output)

    # Save FRP list
    with open(frp_txt_path, 'w') as f:
        for p in result["confirmed_frps"]:
            f.write("%s\n" % p)
    if not args.quiet:
        logger.info("FRP list saved to: %s (%d prefixes)", frp_txt_path, len(result["confirmed_frps"]))

    # Save summary
    with open(summary_path, 'w') as f:
        summary_payload = {
            "confirmed_frps": result["confirmed_frps"],
            "confirmed_frps_detail": result["confirmed_frps_detail"],
            "tree_alignment_results": result["tree_alignment_results"],
            "stats": result["stats"],
            "skipped": result["skipped"],
            "elapsed_seconds": result["stats"]["elapsed_seconds"],
            "run_time_human": _format_duration(result["stats"]["elapsed_seconds"]),
        }
        if args.seed_frps:
            summary_payload["seed_frps_file"] = args.seed_frps
            summary_payload["seed_mode"] = args.seed_mode
            summary_payload["seed_dropped_count"] = len(seed_dropped)
            # Keep only first 200 dropped entries to avoid gigantic summaries
            summary_payload["seed_dropped_sample"] = seed_dropped[:200]
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    if not args.quiet:
        logger.info("Summary saved to: %s", summary_path)

    # Print final summary
    print_summary(result, quiet=args.quiet)


if __name__ == "__main__":
    main()
