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
    result.json         - detailed probe results per prefix
    result.frp.txt      - final confirmed FRP list
    result_summary.json - pipeline statistics
"""

import argparse
import ipaddress
import json
import logging
import os
import sys
import time
from collections import Counter

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
):
    """
    Complete APD pipeline: detect -> judge -> refine -> tree-align -> filter.

    Args:
        prefixes: list of IPv6 prefix strings
        refine: whether to do granularity alignment
        tree_align: whether to do tree-based alignment for non-FRP prefixes
        timeout: per-attempt timeout in seconds
        max_attempts: max ping attempts per sub-prefix (default 5)
        concurrency: parallel probes within a prefix
        min_prefix_len: minimum prefix length for refinement
        output_path: path for detailed JSON output
        frp_output_path: path for FRP list output

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
        with open(jsonl_path, 'w') as f:
            f.write("")
        logger.info("Incremental results: %s", jsonl_path)
    if frp_output_path:
        with open(frp_output_path, 'w') as f:
            f.write("")

    # State
    confirmed_frps = []  # list of {"prefix": ..., "original": ..., "source": ...}
    all_results = []
    skipped = []
    ta_results = []  # tree alignment results

    candidate_queue = list(prefixes)
    processed_count = 0
    total_probes = 0
    detection_probes = 0
    refinement_probes = 0
    ta_probes = 0
    ta_new_frps = 0
    ta_candidates_checked = 0
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("APD Pipeline: %d prefixes", len(candidate_queue))
    logger.info("  ping-style: max %d attempts/sub-prefix, timeout=%gs",
                max_attempts, timeout)
    logger.info("  refine=%s, tree_align=%s, concurrency=%d",
                refine, tree_align, concurrency)
    logger.info("=" * 60)

    # Real-time header
    print("\n" + "=" * 78)
    print("APD REAL-TIME RESULTS (ping-style: %d attempts, refine=%s, ta=%s)"
          % (max_attempts, refine, tree_align))
    print("=" * 78)
    print("%-5s %-42s %-10s %-8s %s" % ("#", "Prefix", "Status", "Reach", "Detail"))
    print("%-5s %-42s %-10s %-8s %s" % ("-" * 5, "-" * 42, "-" * 10, "-" * 8, "-" * 30))
    sys.stdout.flush()

    while candidate_queue:
        prefix = candidate_queue.pop(0)

        # Parse prefix
        try:
            prefix_net = ipaddress.IPv6Network(prefix, strict=False)
        except Exception as e:
            logger.warning("[SKIP] Invalid prefix %s: %s", prefix, e)
            continue

        # Coverage filter
        covering = None
        for f in confirmed_frps:
            f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
            if is_covered_by(prefix_net, f_net):
                covering = f["prefix"]
                break

        if covering:
            skipped.append({"prefix": prefix, "covered_by": covering})
            continue

        processed_count += 1
        logger.info("\n[%d] Processing: %s (queue: %d remaining)",
                    processed_count, prefix, len(candidate_queue))

        # ---- Step 1: Diagnose with ping-style ----
        diag = diagnose_prefix(
            prefix,
            timeout=timeout,
            max_attempts=max_attempts,
            concurrency=concurrency,
        )
        detection_probes += diag["total_probes_sent"]
        total_probes += diag["total_probes_sent"]

        # ---- Step 2: Determine FRP status ----
        is_frp, reason = determine_frp_status(diag)
        diag["frp_decision"] = {"is_frp": is_frp, "reason": reason}

        reach_str = "%d/16" % diag["reachable_count"]

        if not is_frp:
            # Real-time output
            print("%-5d %-42s %-10s %-8s %s" % (
                processed_count, prefix, "NOT FRP", reach_str, reason[:40]
            ))
            sys.stdout.flush()

            # ---- Tree alignment for non-FRP with partial reachability ----
            if tree_align and diag["reachable_count"] > 0:
                known_frp_set = set(f["prefix"] for f in confirmed_frps)
                ta_result = tree_align_prefix(
                    prefix,
                    diag,
                    known_frp_set,
                    timeout=timeout,
                    max_attempts=max_attempts,
                    min_prefix_len=min_prefix_len,
                    concurrency=concurrency,
                    refine=refine,
                    max_depth=max_ta_depth,
                    fast_refine=fast_refine,
                )
                ta_probes += ta_result["total_probes"]
                total_probes += ta_result["total_probes"]
                ta_candidates_checked += len(ta_result["candidates"])
                diag["tree_alignment"] = ta_result

                # Add newly discovered FRPs
                for new_frp in ta_result["new_frps"]:
                    ta_new_frps += 1
                    new_prefix = new_frp["prefix"]
                    new_net = ipaddress.IPv6Network(new_prefix, strict=False)

                    # Remove existing FRPs covered by new one
                    kept = []
                    for f in confirmed_frps:
                        f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
                        if not is_covered_by(f_net, new_net):
                            kept.append(f)
                    confirmed_frps = kept

                    confirmed_frps.append({
                        "prefix": new_prefix,
                        "original": new_frp.get("original_candidate", new_prefix),
                        "source": "tree_align",
                        "source_prefix": prefix,
                    })

                    # Filter queue
                    new_queue = []
                    for q in candidate_queue:
                        try:
                            q_net = ipaddress.IPv6Network(q, strict=False)
                            if not is_covered_by(q_net, new_net):
                                new_queue.append(q)
                            else:
                                skipped.append({"prefix": q, "covered_by": new_prefix})
                        except Exception:
                            new_queue.append(q)
                    candidate_queue = new_queue

                    print("  %-3s %-42s %-10s %-8s %s" % (
                        "↳", new_prefix, "✓ TA-FRP", "",
                        "from %s" % prefix
                    ))
                    sys.stdout.flush()

                    # Incremental FRP write
                    if frp_output_path:
                        with open(frp_output_path, 'a') as f:
                            f.write("%s\n" % new_prefix)

                ta_results.append(ta_result)

            all_results.append(diag)

            # Incremental write
            if jsonl_path:
                with open(jsonl_path, 'a') as f:
                    # Include failed probe addresses instead of full probe details
                    compact = {k: v for k, v in diag.items() if k != "probes"}
                    compact["failed_probes"] = _extract_failed_probes(diag)
                    f.write(json.dumps(compact, ensure_ascii=False) + "\n")
            continue

        # ---- Step 3: FRP confirmed - do refinement ----
        refined_str = prefix
        refinement_info = None
        if refine:
            logger.info("  [REFINE] Finding shortest FRP prefix...")
            ref = refine_frp_prefix(
                prefix,
                timeout=timeout,
                max_attempts=max_attempts,
                min_prefix_len=min_prefix_len,
                concurrency=concurrency,
                fast_refine=fast_refine,
            )
            refined_str = ref["refined"]
            refinement_info = ref
            refinement_probes += ref["total_refinement_probes"]
            total_probes += ref["total_refinement_probes"]
            logger.info("  Refined: /%d -> /%d (%s, -%d bits)",
                        ref["original_len"], ref["refined_len"],
                        refined_str, ref["reduction_bits"])
            diag["refinement"] = ref

        refined_net = ipaddress.IPv6Network(refined_str, strict=False)

        # ---- Step 4: Filter existing FRPs covered by new broader one ----
        kept = []
        for f in confirmed_frps:
            f_net = ipaddress.IPv6Network(f["prefix"], strict=False)
            if not is_covered_by(f_net, refined_net):
                kept.append(f)
        confirmed_frps = kept
        confirmed_frps.append({
            "prefix": str(refined_net),
            "original": prefix,
            "source": "direct",
            "refinement": refinement_info,
        })

        # ---- Step 5: Filter remaining candidates ----
        new_queue = []
        for q in candidate_queue:
            try:
                q_net = ipaddress.IPv6Network(q, strict=False)
                if not is_covered_by(q_net, refined_net):
                    new_queue.append(q)
                else:
                    skipped.append({"prefix": q, "covered_by": str(refined_net)})
            except Exception:
                new_queue.append(q)
        candidate_queue = new_queue

        # Real-time output
        refined_display = ""
        if refine and refined_str != prefix:
            refined_display = " -> %s" % refined_str
        print("%-5d %-42s %-10s %-8s %s" % (
            processed_count, prefix, "✓ FRP", reach_str, refined_display
        ))
        sys.stdout.flush()

        all_results.append(diag)

        # Incremental writes
        if jsonl_path:
            with open(jsonl_path, 'a') as f:
                compact = {k: v for k, v in diag.items() if k != "probes"}
                compact["failed_probes"] = _extract_failed_probes(diag)
                f.write(json.dumps(compact, ensure_ascii=False) + "\n")
        if frp_output_path:
            with open(frp_output_path, 'a') as f:
                f.write("%s\n" % refined_str)

    elapsed = time.time() - start_time

    # Footer
    print("%-5s %-42s %-10s %-8s %s" % ("-" * 5, "-" * 42, "-" * 10, "-" * 8, "-" * 30))
    print("Done. %d FRPs confirmed from %d probed (%d skipped). Time: %.1fs"
          % (len(confirmed_frps), processed_count, len(skipped), elapsed))
    print("Total probes: %d (detection: %d, refinement: %d, tree-align: %d)"
          % (total_probes, detection_probes, refinement_probes, ta_probes))
    if tree_align:
        print("Tree alignment: %d candidates checked, %d new FRPs found"
              % (ta_candidates_checked, ta_new_frps))
    print("=" * 78)
    sys.stdout.flush()

    # Stats
    stats = {
        "total_input": len(prefixes),
        "processed": processed_count,
        "skipped_covered": len(skipped),
        "confirmed_frps": len(confirmed_frps),
        "non_frps": processed_count - len([f for f in confirmed_frps if f["source"] == "direct"]),
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
    }

    return {
        "confirmed_frps": [f["prefix"] for f in confirmed_frps],
        "confirmed_frps_detail": confirmed_frps,
        "all_results": all_results,
        "tree_alignment_results": ta_results,
        "skipped": skipped,
        "stats": stats,
    }


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


def print_summary(result):
    """Print human-readable summary to stdout."""
    stats = result["stats"]
    frps = result["confirmed_frps"]
    all_results = result["all_results"]

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
    print("Non-FRPs:                  %d" % stats["non_frps"])
    print("Total time:                %.1fs" % stats["elapsed_seconds"])
    print("Avg per prefix:            %.1fs" % stats["avg_seconds_per_prefix"])
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

  # Full pipeline
  sudo python3 standalone_apd.py -i prefixes.txt -o result.json --refine --ta

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
                        help='Use fast 4-bit-then-1-bit refinement (default: precise 1-bit only)')
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
    parser.add_argument('--min-prefix-len', type=int, default=16,
                        help='Minimum prefix length for refinement (default: 16)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process first N prefixes')
    parser.add_argument('--frp-output', default=None,
                        help='FRP list output file (default: <output>.frp.txt)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose logging (DEBUG level)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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

    # Prepare output paths
    frp_txt_path = args.frp_output
    if not frp_txt_path:
        base = args.output.rsplit('.', 1)[0] if '.' in args.output else args.output
        frp_txt_path = base + ".frp.txt"
    summary_path = (
        args.output.rsplit('.', 1)[0] + "_summary.json"
        if '.' in args.output else args.output + "_summary.json"
    )

    # Run pipeline
    result = run_apd_pipeline(
        prefixes,
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
    )

    # Save detailed results (with failed probe addresses, without full probe data)
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
    logger.info("FRP list saved to: %s (%d prefixes)", frp_txt_path, len(result["confirmed_frps"]))

    # Save summary
    with open(summary_path, 'w') as f:
        json.dump({
            "confirmed_frps": result["confirmed_frps"],
            "confirmed_frps_detail": result["confirmed_frps_detail"],
            "tree_alignment_results": result["tree_alignment_results"],
            "stats": result["stats"],
            "skipped": result["skipped"],
            "elapsed_seconds": result["stats"]["elapsed_seconds"],
            "run_time_human": _format_duration(result["stats"]["elapsed_seconds"]),
        }, f, indent=2, ensure_ascii=False)
    logger.info("Summary saved to: %s", summary_path)

    # Print final summary
    print_summary(result)


if __name__ == "__main__":
    main()
