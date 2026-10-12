#!/usr/bin/env python3
"""
APD core - reusable ICMPv6 FRP-detection primitives.

Ping-style probing: for each sub-prefix, send up to 5 ICMPv6 Echo Requests
(1 per second). Any reply means "reachable". All 16 sub-prefixes reachable
means FRP.

Tree-based alignment (--ta): when a prefix is NOT FRP, analyze which
sub-prefixes are reachable, then identify maximal candidate sub-FRPs
by grouping contiguous reachable nibbles into the shortest covering
prefixes.
"""

import ipaddress
import logging
import os
import random
import socket
import struct
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ICMPv6 constants
# ---------------------------------------------------------------------------

ICMPV6_ECHO_REQUEST = 128
ICMPV6_ECHO_REPLY = 129
ICMPV6_DEST_UNREACHABLE = 1
ICMPV6_PACKET_TOO_BIG = 2
ICMPV6_TIME_EXCEEDED = 3
ICMPV6_PARAM_PROBLEM = 4

ICMPV6_NO_ROUTE = 0
ICMPV6_ADMIN_PROHIBITED = 1
ICMPV6_BEYOND_SCOPE = 2
ICMPV6_ADDR_UNREACHABLE = 3
ICMPV6_PORT_UNREACHABLE = 4
ICMPV6_FAILED_POLICY = 5
ICMPV6_REJECT_ROUTE = 6

ICMPV6_TYPE_NAMES = {
    1: "Destination Unreachable",
    2: "Packet Too Big",
    3: "Time Exceeded",
    4: "Parameter Problem",
    128: "Echo Request",
    129: "Echo Reply",
    130: "Multicast Listener Query",
    131: "Multicast Listener Report",
    132: "Multicast Listener Done",
    133: "Router Solicitation",
    134: "Router Advertisement",
    135: "Neighbor Solicitation",
    136: "Neighbor Advertisement",
    137: "Redirect Message",
}

ICMPV6_DEST_UNREACH_CODES = {
    0: "No route to destination",
    1: "Administratively prohibited",
    2: "Beyond scope of source address",
    3: "Address unreachable",
    4: "Port unreachable",
    5: "Source address failed policy",
    6: "Reject route to destination",
}


# ---------------------------------------------------------------------------
# Capability check
# ---------------------------------------------------------------------------

_CAP_NET_RAW_CHECKED = False
_CAP_NET_RAW_OK = False


def _ensure_raw_socket_capability():
    global _CAP_NET_RAW_CHECKED, _CAP_NET_RAW_OK
    if _CAP_NET_RAW_CHECKED:
        if not _CAP_NET_RAW_OK:
            raise PermissionError(
                "ICMPv6 raw socket requires CAP_NET_RAW; run as root or use sudo"
            )
        return

    _CAP_NET_RAW_CHECKED = True
    try:
        probe_sock = socket.socket(
            socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6
        )
        probe_sock.close()
        _CAP_NET_RAW_OK = True
    except (PermissionError, OSError) as exc:
        _CAP_NET_RAW_OK = False
        raise PermissionError(
            "ICMPv6 raw socket requires CAP_NET_RAW; run as root or use sudo"
        ) from exc


def has_raw_socket_capability():
    """Non-raising variant for health checks and CLI pre-flights."""
    global _CAP_NET_RAW_CHECKED, _CAP_NET_RAW_OK
    if _CAP_NET_RAW_CHECKED:
        return _CAP_NET_RAW_OK
    try:
        _ensure_raw_socket_capability()
    except PermissionError:
        return False
    return True


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def classify_response(response_bytes):
    """Parse an ICMPv6 response buffer and classify it."""
    if len(response_bytes) < 4:
        return {
            "type": None,
            "type_name": "Too short",
            "code": None,
            "code_name": "",
            "classification": "MALFORMED",
            "raw_hex": response_bytes.hex(),
        }

    icmp_type, icmp_code = struct.unpack("!BB", response_bytes[:2])
    type_name = ICMPV6_TYPE_NAMES.get(icmp_type, "Unknown(%d)" % icmp_type)

    result = {
        "type": icmp_type,
        "type_name": type_name,
        "code": icmp_code,
        "code_name": "",
        "classification": "UNKNOWN",
        "raw_hex": response_bytes[:16].hex(),
    }

    if icmp_type == ICMPV6_ECHO_REPLY:
        result["classification"] = "SUCCESS"
        result["code_name"] = "Echo Reply"
    elif icmp_type == ICMPV6_DEST_UNREACHABLE:
        result["code_name"] = ICMPV6_DEST_UNREACH_CODES.get(
            icmp_code, "Unknown code(%d)" % icmp_code
        )
        if icmp_code == ICMPV6_ADMIN_PROHIBITED:
            result["classification"] = "ADMIN_PROHIBITED"
        elif icmp_code == ICMPV6_NO_ROUTE:
            result["classification"] = "NO_ROUTE"
        elif icmp_code == ICMPV6_ADDR_UNREACHABLE:
            result["classification"] = "ADDR_UNREACHABLE"
        elif icmp_code == ICMPV6_PORT_UNREACHABLE:
            result["classification"] = "PORT_UNREACHABLE"
        elif icmp_code == ICMPV6_FAILED_POLICY:
            result["classification"] = "POLICY_FAILED"
        elif icmp_code == ICMPV6_REJECT_ROUTE:
            result["classification"] = "REJECT_ROUTE"
        else:
            result["classification"] = "DEST_UNREACHABLE_OTHER"
    elif icmp_type == ICMPV6_TIME_EXCEEDED:
        result["classification"] = "TIME_EXCEEDED"
        if icmp_code == 0:
            result["code_name"] = "Hop limit exceeded"
        elif icmp_code == 1:
            result["code_name"] = "Fragment reassembly time exceeded"
    elif icmp_type == ICMPV6_PACKET_TOO_BIG:
        result["classification"] = "PACKET_TOO_BIG"
    elif icmp_type == ICMPV6_PARAM_PROBLEM:
        result["classification"] = "PARAM_PROBLEM"
    else:
        result["classification"] = "ICMPV6_TYPE_%d" % icmp_type

    return result


# ---------------------------------------------------------------------------
# Ping-style probing: send up to max_attempts packets (1/sec), any reply = OK
# ---------------------------------------------------------------------------


def ping_probe_address(address, max_attempts=5, timeout_per_attempt=2.0):
    """
    Ping-style probe: send up to max_attempts ICMPv6 Echo Requests to address,
    one per second. If ANY reply is received, return success immediately.

    This handles packet loss by retrying, similar to `ping -c 5`.

    Returns:
        dict with keys: address, reachable (bool), attempts_used, total_time_ms,
                        last_classification, detail
    """
    _ensure_raw_socket_capability()

    result = {
        "address": address,
        "reachable": False,
        "attempts_used": 0,
        "total_time_ms": 0,
        "rtt_ms": None,
        "last_classification": None,
        "detail": None,
    }

    start_time = time.time()

    for attempt in range(max_attempts):
        result["attempts_used"] = attempt + 1

        # Wait 1 second between attempts (not before first)
        if attempt > 0:
            time.sleep(1.0)

        try:
            sock = socket.socket(
                socket.AF_INET6, socket.SOCK_RAW, socket.IPPROTO_ICMPV6
            )
            sock.settimeout(timeout_per_attempt)

            identifier = random.randint(0, 65535)
            sequence = attempt
            header = struct.pack(
                "!BBHHH", ICMPV6_ECHO_REQUEST, 0, 0, identifier, sequence
            )
            payload = struct.pack("!d", time.time())
            packet = header + payload

            send_time = time.time()
            sock.sendto(packet, (address, 0, 0, 0))

            # Wait for response
            while True:
                elapsed = time.time() - send_time
                remaining = timeout_per_attempt - elapsed
                if remaining <= 0:
                    result["last_classification"] = "TIMEOUT"
                    break

                sock.settimeout(remaining)
                try:
                    response_data, addr_info = sock.recvfrom(1500)
                    resp_detail = classify_response(response_data)

                    if resp_detail["type"] == ICMPV6_ECHO_REPLY:
                        # Check identifier match
                        if len(response_data) >= 8:
                            resp_id, resp_seq = struct.unpack(
                                "!HH", response_data[4:8]
                            )
                            if resp_id != identifier:
                                continue  # Not our packet
                        rtt = (time.time() - send_time) * 1000
                        result["reachable"] = True
                        result["rtt_ms"] = round(rtt, 2)
                        result["last_classification"] = "SUCCESS"
                        result["detail"] = resp_detail
                        result["total_time_ms"] = round(
                            (time.time() - start_time) * 1000, 2
                        )
                        sock.close()
                        return result

                    elif resp_detail["type"] in (
                        ICMPV6_DEST_UNREACHABLE,
                        ICMPV6_TIME_EXCEEDED,
                        ICMPV6_PACKET_TOO_BIG,
                        ICMPV6_PARAM_PROBLEM,
                    ):
                        result["last_classification"] = resp_detail["classification"]
                        result["detail"] = resp_detail
                        # Definitive negative: no point retrying for some codes
                        if resp_detail["classification"] in (
                            "NO_ROUTE",
                            "ADMIN_PROHIBITED",
                            "REJECT_ROUTE",
                        ):
                            result["total_time_ms"] = round(
                                (time.time() - start_time) * 1000, 2
                            )
                            sock.close()
                            return result
                        break  # Other errors: try next attempt
                    else:
                        continue  # Ignore unrelated packets

                except socket.timeout:
                    result["last_classification"] = "TIMEOUT"
                    break

            sock.close()

        except OSError as e:
            if e.errno == 101:
                result["last_classification"] = "NETWORK_UNREACHABLE"
                result["total_time_ms"] = round(
                    (time.time() - start_time) * 1000, 2
                )
                return result
            elif e.errno == 113:
                result["last_classification"] = "NO_ROUTE"
                result["total_time_ms"] = round(
                    (time.time() - start_time) * 1000, 2
                )
                return result
            else:
                result["last_classification"] = "OS_ERROR"
                # Continue to next attempt
        except Exception as e:
            result["last_classification"] = "EXCEPTION"
            # Continue to next attempt

    result["total_time_ms"] = round((time.time() - start_time) * 1000, 2)
    return result


# ---------------------------------------------------------------------------
# Address sampling
# ---------------------------------------------------------------------------


def generate_probe_addresses(prefix, count=16):
    """
    Generate ``count`` probe addresses evenly covering 16 sub-spaces of the
    prefix (the 4 bits immediately following the prefix are enumerated
    0..15; the remaining tail bits are random).

    Returns list of (nibble_index, address) tuples.
    """
    network = ipaddress.IPv6Network(prefix, strict=False)
    plen = network.prefixlen
    base_int = int(network.network_address)
    remaining_bits = 128 - plen

    if remaining_bits < 4:
        # Prefix too long for 16-way split; just generate random addresses
        addresses = []
        for i in range(count):
            rand_int = random.randint(
                int(network.network_address), int(network.broadcast_address)
            )
            addresses.append((i, str(ipaddress.IPv6Address(rand_int))))
        return addresses

    tail_bits = remaining_bits - 4
    addresses = []
    for nibble in range(16):
        nibble_shifted = nibble << tail_bits
        if tail_bits > 0:
            tail_random = random.randint(0, (1 << tail_bits) - 1)
        else:
            tail_random = 0
        addr_int = base_int | nibble_shifted | tail_random
        addresses.append((nibble, str(ipaddress.IPv6Address(addr_int))))

    return addresses[:count]


# Keep backward-compatible alias
def generate_random_addresses(prefix, count=16):
    """Backward-compatible wrapper returning just addresses."""
    return [addr for _, addr in generate_probe_addresses(prefix, count)]


# ---------------------------------------------------------------------------
# Ping-style prefix diagnosis
# ---------------------------------------------------------------------------


def diagnose_prefix(
    prefix,
    timeout=2.0,
    probe_count=16,
    max_attempts=5,
    concurrency=16,
    inter_probe_delay=0.0,
):
    """
    Diagnose a prefix using ping-style probing.

    For each of the 16 sub-prefixes, send up to max_attempts ICMPv6 packets
    (1 per second). Any reply = reachable. All 16 reachable = FRP.

    Args:
        prefix: IPv6 prefix string (e.g. "2001:db8::/32")
        timeout: timeout per individual probe attempt (seconds)
        probe_count: number of sub-prefixes to probe (default 16)
        max_attempts: max ping attempts per sub-prefix (default 5)
        concurrency: parallel probes (default 16, one per sub-prefix)
        inter_probe_delay: delay between launching concurrent probes

    Returns:
        dict with detailed results per sub-prefix
    """
    logger.info("Diagnosing prefix: %s (ping-style, max %d attempts/sub)",
                prefix, max_attempts)

    probe_addrs = generate_probe_addresses(prefix, probe_count)

    # Probe all sub-prefixes concurrently
    probe_results = [None] * len(probe_addrs)
    max_workers = min(concurrency, len(probe_addrs))

    def _task(idx, nibble, addr):
        if idx > 0 and inter_probe_delay > 0:
            time.sleep(idx * inter_probe_delay)
        return idx, nibble, ping_probe_address(
            addr, max_attempts=max_attempts, timeout_per_attempt=timeout
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_task, i, nibble, addr)
            for i, (nibble, addr) in enumerate(probe_addrs)
        ]
        for fut in as_completed(futures):
            idx, nibble, result = fut.result()
            result["nibble_index"] = nibble
            probe_results[idx] = result

    # Analyze results
    reachable_count = sum(1 for r in probe_results if r["reachable"])
    reachable_nibbles = [r["nibble_index"] for r in probe_results if r["reachable"]]
    unreachable_nibbles = [
        r["nibble_index"] for r in probe_results if not r["reachable"]
    ]

    total_attempts = sum(r["attempts_used"] for r in probe_results)
    is_frp = reachable_count == probe_count

    # Classification summary
    classifications = Counter(r["last_classification"] for r in probe_results)

    # RTT stats for reachable probes
    rtts = [r["rtt_ms"] for r in probe_results if r["reachable"] and r["rtt_ms"]]
    rtt_stats = {}
    if rtts:
        rtt_stats = {
            "min_ms": round(min(rtts), 2),
            "max_ms": round(max(rtts), 2),
            "avg_ms": round(sum(rtts) / len(rtts), 2),
            "median_ms": round(sorted(rtts)[len(rtts) // 2], 2),
        }

    # Diagnosis messages
    diagnosis = []
    if is_frp:
        diagnosis.append("FRP: 16/16 sub-prefixes reachable")
    elif reachable_count == 0:
        diagnosis.append("DEAD: 0/16 sub-prefixes reachable")
        if "TIMEOUT" in classifications:
            diagnosis.append("  All timed out after %d attempts each" % max_attempts)
        if "NO_ROUTE" in classifications:
            diagnosis.append("  No route to destination")
        if "ADMIN_PROHIBITED" in classifications:
            diagnosis.append("  Administratively prohibited")
    else:
        diagnosis.append(
            "PARTIAL: %d/16 reachable (nibbles: %s)"
            % (reachable_count, reachable_nibbles)
        )
        diagnosis.append(
            "  Unreachable nibbles: %s" % unreachable_nibbles
        )

    return {
        "prefix": prefix,
        "timestamp": datetime.now().isoformat(),
        "probe_count": probe_count,
        "max_attempts": max_attempts,
        "reachable_count": reachable_count,
        "reachable_rate": round(reachable_count / probe_count, 4) if probe_count > 0 else 0,
        "is_frp": is_frp,
        "reachable_nibbles": sorted(reachable_nibbles),
        "unreachable_nibbles": sorted(unreachable_nibbles),
        "total_probes_sent": total_attempts,
        "classification_summary": dict(classifications),
        "rtt_stats": rtt_stats,
        "diagnosis": diagnosis,
        "probes": probe_results,
    }


# ---------------------------------------------------------------------------
# FRP determination (simplified for ping-style)
# ---------------------------------------------------------------------------


def determine_frp_status(result):
    """
    Simple FRP determination based on ping-style results.
    All 16 sub-prefixes reachable = FRP. Otherwise not.

    Returns: (is_frp, reason)
    """
    if result["is_frp"]:
        return True, "FRP: all %d/%d sub-prefixes reachable" % (
            result["reachable_count"], result["probe_count"]
        )
    elif result["reachable_count"] == 0:
        return False, "NOT_FRP: 0/%d reachable (prefix dead)" % result["probe_count"]
    else:
        return False, "NOT_FRP: %d/%d reachable (partial)" % (
            result["reachable_count"], result["probe_count"]
        )


# ---------------------------------------------------------------------------
# Granularity refinement (ping-style)
# ---------------------------------------------------------------------------


def refine_frp_prefix(
    prefix,
    timeout=2.0,
    max_attempts=5,
    min_prefix_len=16,
    concurrency=16,
    fast_refine=False,
    refine_mode=None,
    refine_timeout=None,
    refine_max_attempts=None,
    coverage_checker=None,
):
    """
    Shrink prefix as much as possible while it remains FRP.
    Uses ping-style probing for consistency.

    Args:
        refine_mode: one of "bisect" (default, fastest), "linear" (1-bit steps),
                     "fast" (4-bit then 1-bit). Overrides ``fast_refine``.
                     For backward compat: if ``refine_mode`` is None, use
                     "fast" when fast_refine=True else "linear".
        refine_timeout: per-attempt timeout for refine probes. Falls back to
                     ``timeout`` if not provided. Can be smaller than the
                     detection timeout to speed up refinement.
        refine_max_attempts: max ping attempts per sub-prefix during refine.
                     Falls back to ``max_attempts`` if not provided.
        coverage_checker: optional callable ``fn(ipaddress.IPv6Network) -> Optional[str]``
                     that returns a covering FRP prefix string if the given
                     network is already covered by a known broader FRP.
                     If it fires, refinement short-circuits to that covering
                     FRP (no more probing).

    Refine modes (probing-count roughly):
        linear:  (orig_len - final_len) full diagnoses
        fast:    O(1) 4-bit jumps + a few 1-bit fine-tunings
        bisect:  ~log2(orig_len - min_prefix_len) full diagnoses
    """
    network = ipaddress.IPv6Network(prefix, strict=False)
    current_len = network.prefixlen
    base_addr = network.network_address
    steps = []

    # Resolve refine-stage probe parameters (allow override).
    r_timeout = refine_timeout if refine_timeout is not None else timeout
    r_attempts = (
        refine_max_attempts if refine_max_attempts is not None else max_attempts
    )

    # Resolve mode. Explicit refine_mode wins over legacy fast_refine flag.
    if refine_mode is None:
        refine_mode = "fast" if fast_refine else "linear"
    if refine_mode not in ("linear", "fast", "bisect"):
        logger.warning("Unknown refine_mode=%r, falling back to 'linear'", refine_mode)
        refine_mode = "linear"

    logger.info("  Refining %s (starting at /%d, min /%d, mode=%s)",
                prefix, current_len, min_prefix_len, refine_mode)

    # Cache diagnose results by prefix_len so bisect / retries don't re-probe.
    diag_cache = {}
    # Coverage short-circuit result (if any).
    coverage_hit = {"prefix": None, "len": None}

    def _try_prefix_len(plen):
        """Diagnose prefix at length ``plen``. Returns (is_frp, step_dict).

        Uses cache so a prefix_len is probed at most once per refine call.
        Also honors ``coverage_checker`` for short-circuit early stop.
        """
        if plen in diag_cache:
            return diag_cache[plen]

        candidate = "%s/%d" % (str(base_addr), plen)

        # Coverage short-circuit: if this (or any broader) prefix is already
        # covered by a known FRP, we can stop refining right away.
        if coverage_checker is not None:
            try:
                cand_net = ipaddress.IPv6Network(candidate, strict=False)
                covering = coverage_checker(cand_net)
            except Exception:
                covering = None
            if covering:
                logger.info(
                    "    /%d already covered by known FRP %s, short-circuit",
                    plen, covering,
                )
                coverage_hit["prefix"] = covering
                try:
                    coverage_hit["len"] = ipaddress.IPv6Network(
                        covering, strict=False
                    ).prefixlen
                except Exception:
                    coverage_hit["len"] = plen
                # Fake a positive step so caller treats plen as FRP; the
                # outer code will replace the result with the covering FRP.
                step = {
                    "prefix": candidate,
                    "prefix_len": plen,
                    "is_frp": True,
                    "reachable_count": 16,
                    "reachable_rate": 1.0,
                    "total_probes_sent": 0,
                    "classification_summary": {},
                    "coverage_short_circuit": covering,
                }
                steps.append(step)
                diag_cache[plen] = (True, step)
                return True, step

        dr = diagnose_prefix(
            candidate,
            timeout=r_timeout,
            max_attempts=r_attempts,
            concurrency=concurrency,
        )
        step = {
            "prefix": candidate,
            "prefix_len": plen,
            "is_frp": dr["is_frp"],
            "reachable_count": dr["reachable_count"],
            "reachable_rate": dr["reachable_rate"],
            "total_probes_sent": dr["total_probes_sent"],
            "classification_summary": dr["classification_summary"],
        }
        steps.append(step)
        result = (dr["is_frp"], step)
        diag_cache[plen] = result
        return result

    if refine_mode == "fast":
        # Fast mode: 4-bit steps first, then 1-bit fine-tuning
        step_size = 4
        while current_len - step_size >= min_prefix_len:
            new_len = current_len - step_size
            is_frp, _ = _try_prefix_len(new_len)
            if coverage_hit["prefix"]:
                current_len = coverage_hit["len"]
                break
            if is_frp:
                current_len = new_len
                logger.info("    /%d still FRP, continuing (4-bit step)...", new_len)
            else:
                logger.info("    /%d NOT FRP, switching to 1-bit step", new_len)
                break

        # Fine-tune with 1-bit steps (unless coverage already short-circuited)
        if not coverage_hit["prefix"]:
            while current_len - 1 >= min_prefix_len:
                new_len = current_len - 1
                is_frp, _ = _try_prefix_len(new_len)
                if coverage_hit["prefix"]:
                    current_len = coverage_hit["len"]
                    break
                if is_frp:
                    current_len = new_len
                    logger.info("    /%d still FRP, continuing (1-bit step)...", new_len)
                else:
                    logger.info("    /%d NOT FRP, stopping", new_len)
                    break

    elif refine_mode == "bisect":
        # Binary search on prefix length.
        # Invariant: prefix at `hi` is FRP; prefix at `lo - 1` is NOT FRP
        # (or lo == min_prefix_len meaning we can't go shorter).
        #
        # We want the SMALLEST plen in [min_prefix_len, current_len] such
        # that the prefix is still FRP. That plen becomes the final refined
        # length.
        hi = current_len  # known FRP (input assumption)
        lo = min_prefix_len

        if hi <= lo:
            # Already at or below min; nothing to do.
            pass
        else:
            # First: try min_prefix_len directly. If it's already FRP, we're done.
            is_frp, _ = _try_prefix_len(lo)
            if coverage_hit["prefix"]:
                current_len = coverage_hit["len"]
            elif is_frp:
                current_len = lo
                logger.info("    /%d is FRP, min reached in 1 probe (bisect)", lo)
            else:
                # Binary search between (lo, hi]: find smallest FRP plen.
                # Invariant during loop:
                #   * plen=hi is FRP  * plen=lo is NOT FRP
                # We keep shrinking the window.
                while hi - lo > 1:
                    if coverage_hit["prefix"]:
                        break
                    mid = (lo + hi) // 2
                    is_frp, _ = _try_prefix_len(mid)
                    if coverage_hit["prefix"]:
                        break
                    if is_frp:
                        hi = mid
                        logger.info("    /%d FRP (bisect: window=[%d,%d])",
                                    mid, lo, hi)
                    else:
                        lo = mid
                        logger.info("    /%d NOT FRP (bisect: window=[%d,%d])",
                                    mid, lo, hi)

                if coverage_hit["prefix"]:
                    current_len = coverage_hit["len"]
                else:
                    current_len = hi  # smallest known FRP plen

    else:
        # linear mode: precise 1-bit steps only
        while current_len - 1 >= min_prefix_len:
            new_len = current_len - 1
            is_frp, _ = _try_prefix_len(new_len)
            if coverage_hit["prefix"]:
                current_len = coverage_hit["len"]
                break
            if is_frp:
                current_len = new_len
                logger.info("    /%d still FRP, continuing...", new_len)
            else:
                logger.info("    /%d NOT FRP, stopping", new_len)
                break

    # If coverage short-circuited, we return the COVERING FRP as the refined
    # result (since it's broader than anything we could have proven ourselves).
    if coverage_hit["prefix"]:
        refined = coverage_hit["prefix"]
    else:
        refined = "%s/%d" % (str(base_addr), current_len)
    total_probes = sum(s["total_probes_sent"] for s in steps)

    return {
        "original": prefix,
        "refined": refined,
        "original_len": network.prefixlen,
        "refined_len": current_len,
        "reduction_bits": network.prefixlen - current_len,
        "mode": refine_mode,
        "steps": steps,
        "total_refinement_probes": total_probes,
        "coverage_short_circuited": coverage_hit["prefix"] is not None,
        "coverage_covered_by": coverage_hit["prefix"],
    }


# ---------------------------------------------------------------------------
# Tree-based alignment (--ta): find sub-FRPs from partial-reachable prefixes
# ---------------------------------------------------------------------------


def find_candidate_sub_frps(prefix, reachable_nibbles, unreachable_nibbles):
    """
    Given a non-FRP prefix where some nibbles are reachable and some are not,
    identify candidate sub-prefixes that might be FRPs.

    Strategy: group contiguous reachable nibbles into the shortest covering
    prefixes (maximally aggregated).

    Example: prefix = 2001:3500::/29, reachable nibbles = [0..6, 8..15]
    (nibbles 0-6 and 8-15 are reachable, nibble 7 is not)
    -> candidates: 2001:3500::/31 (covers nibbles 0-3), 2001:3504::/30 (covers 4-7 minus 7)
    Actually we find maximal power-of-2 aligned groups.

    The nibble index (0-15) represents the 4 bits immediately after the prefix.
    So for prefix /N, nibble i corresponds to sub-prefix (base | i<<(128-N-4)) / (N+4).

    We want to find the SHORTEST (broadest) prefixes that cover ONLY reachable
    nibbles. This means finding maximal aligned groups of reachable nibbles.

    Returns:
        list of candidate prefix strings to verify
    """
    if not reachable_nibbles or not unreachable_nibbles:
        return []

    network = ipaddress.IPv6Network(prefix, strict=False)
    plen = network.prefixlen
    base_int = int(network.network_address)
    remaining_bits = 128 - plen

    if remaining_bits < 4:
        return []

    reachable_set = set(reachable_nibbles)

    # Find maximal power-of-2 aligned groups of reachable nibbles
    # A group of size 2^k starting at nibble n*2^k is valid if all nibbles
    # n*2^k .. n*2^k + 2^k - 1 are reachable.
    # We want the LARGEST such groups (fewest candidates).

    candidates = []
    covered = set()

    # Try from largest group size down to 1
    for k in range(4, -1, -1):  # k=4 means group of 16 (the whole prefix)
        group_size = 1 << k
        num_groups = 16 // group_size

        for g in range(num_groups):
            start_nibble = g * group_size
            group_nibbles = set(range(start_nibble, start_nibble + group_size))

            # Skip if any nibble in this group is already covered
            if group_nibbles & covered:
                continue

            # Check if ALL nibbles in this group are reachable
            if group_nibbles.issubset(reachable_set):
                # This is a valid candidate group
                covered |= group_nibbles

                # Calculate the sub-prefix
                # The group starts at nibble start_nibble within the 4-bit space
                # New prefix length = plen + (4 - k)
                new_plen = plen + (4 - k)
                # The bits for this group: start_nibble >> k gives the group index
                # at the new prefix level
                shift = remaining_bits - 4
                group_base = base_int | (start_nibble << shift)
                # Mask to the new prefix length
                mask = ((1 << 128) - 1) ^ ((1 << (128 - new_plen)) - 1)
                group_base = group_base & mask

                candidate_prefix = "%s/%d" % (
                    str(ipaddress.IPv6Address(group_base)), new_plen
                )
                candidates.append({
                    "prefix": candidate_prefix,
                    "prefix_len": new_plen,
                    "nibbles_covered": sorted(group_nibbles),
                    "group_size": group_size,
                    "extra_bits": 4 - k,
                })

    # Filter out the full prefix itself (group_size=16 means it's the original)
    candidates = [c for c in candidates if c["group_size"] < 16]

    # Sort by prefix length (shorter = broader = better)
    candidates.sort(key=lambda c: c["prefix_len"])

    return candidates


def tree_align_prefix(
    prefix,
    diag_result,
    known_frps,
    timeout=2.0,
    max_attempts=5,
    min_prefix_len=16,
    concurrency=16,
    refine=True,
    max_depth=4,
    _current_depth=0,
    fast_refine=False,
    refine_mode=None,
    refine_timeout=None,
    refine_max_attempts=None,
    coverage_checker=None,
    _defer_refine=None,
):
    """
    Tree-based alignment for a non-FRP prefix.

    Given the diagnosis result showing which nibbles are reachable,
    find candidate sub-FRPs and verify them. If a candidate fails APD
    but still has reachable nibbles, recursively apply tree alignment
    on that candidate (up to max_depth levels).

    IMPORTANT (new behavior, C-plan): refinement is DEFERRED. During the
    recursive walk, whenever we discover an FRP we only record it. At the
    top-level call (``_current_depth == 0``), after the whole tree has
    been explored, we compute the maximal (broadest) set of FRPs and
    refine ONLY those. This prevents wasted refinement effort on FRPs
    that turn out to be covered by a broader FRP found in a sibling.

    Args:
        prefix: the original non-FRP prefix
        diag_result: result from diagnose_prefix (must have reachable_nibbles)
        known_frps: set of known FRP prefix strings (for dedup); this set is
            mutated in-place as new FRPs are found so recursive calls see
            them immediately.
        refine: whether to also refine discovered FRPs at the end
        max_depth: maximum recursion depth for tree alignment (default 4)
        _current_depth: internal counter for current recursion depth
        refine_mode / refine_timeout / refine_max_attempts:
            forwarded to refine_frp_prefix.
        coverage_checker: optional live-view callable
            ``fn(ipaddress.IPv6Network) -> Optional[str]``. If given, it is
            consulted BEFORE probing each candidate; a hit means the
            candidate is already covered by a broader known FRP and we skip.
            Also forwarded to refine.
        _defer_refine: internal; top-level call sets this to an empty list
            and all recursion appends "pending FRPs" to it. When the
            top-level call returns from recursion it drains the list.
    """
    reachable_nibbles = diag_result.get("reachable_nibbles", [])
    unreachable_nibbles = diag_result.get("unreachable_nibbles", [])

    is_top_level = _defer_refine is None
    if is_top_level:
        _defer_refine = []

    result = {
        "source_prefix": prefix,
        "reachable_nibbles": reachable_nibbles,
        "unreachable_nibbles": unreachable_nibbles,
        "candidates": [],
        "new_frps": [],
        "total_probes": 0,
    }

    if not reachable_nibbles or not unreachable_nibbles:
        # Nothing to explore. If we're at top level and there are deferred
        # FRPs somehow (shouldn't happen unless caller misuses), just drain.
        if is_top_level and _defer_refine:
            _drain_deferred_refine(
                _defer_refine, result, known_frps,
                timeout=timeout,
                max_attempts=max_attempts,
                min_prefix_len=min_prefix_len,
                concurrency=concurrency,
                refine=refine,
                refine_mode=refine_mode,
                refine_timeout=refine_timeout,
                refine_max_attempts=refine_max_attempts,
                coverage_checker=coverage_checker,
                fast_refine=fast_refine,
            )
        return result

    # Find candidate sub-prefixes
    candidates = find_candidate_sub_frps(prefix, reachable_nibbles, unreachable_nibbles)

    if not candidates:
        if is_top_level and _defer_refine:
            _drain_deferred_refine(
                _defer_refine, result, known_frps,
                timeout=timeout,
                max_attempts=max_attempts,
                min_prefix_len=min_prefix_len,
                concurrency=concurrency,
                refine=refine,
                refine_mode=refine_mode,
                refine_timeout=refine_timeout,
                refine_max_attempts=refine_max_attempts,
                coverage_checker=coverage_checker,
                fast_refine=fast_refine,
            )
        return result

    logger.info("  [TA] Found %d candidate sub-prefixes for %s",
                len(candidates), prefix)

    for cand in candidates:
        cand_prefix = cand["prefix"]
        cand_result = {
            "prefix": cand_prefix,
            "prefix_len": cand["prefix_len"],
            "nibbles_covered": cand["nibbles_covered"],
            "already_known": False,
            "verified_frp": False,
            "refined_prefix": None,
            "probes_used": 0,
        }

        # Check if already in known FRPs (union of caller-provided known_frps
        # AND the live coverage_checker view, when available).
        try:
            cand_net = ipaddress.IPv6Network(cand_prefix, strict=False)
        except Exception:
            result["candidates"].append(cand_result)
            continue

        already_covered = False
        covering_prefix = None

        # 1) Live coverage checker (sees FRPs found by parallel workers too)
        if coverage_checker is not None:
            try:
                covering_prefix = coverage_checker(cand_net)
            except Exception:
                covering_prefix = None

        # 2) Local known_frps set (updated as we recurse)
        if covering_prefix is None:
            for known in known_frps:
                try:
                    known_net = ipaddress.IPv6Network(known, strict=False)
                    if is_covered_by(cand_net, known_net):
                        covering_prefix = known
                        break
                except Exception:
                    continue

        if covering_prefix is not None:
            already_covered = True
            cand_result["already_known"] = True
            cand_result["covered_by"] = covering_prefix

        if already_covered:
            logger.info("    [TA] %s already covered by known FRP %s",
                        cand_prefix, covering_prefix)
            result["candidates"].append(cand_result)
            continue

        # Verify: run full APD detection on candidate
        indent = "    " * (_current_depth + 1)
        logger.info("%s[TA-L%d] Verifying candidate: %s",
                    indent, _current_depth, cand_prefix)
        diag = diagnose_prefix(
            cand_prefix,
            timeout=timeout,
            max_attempts=max_attempts,
            concurrency=concurrency,
        )
        cand_result["probes_used"] = diag["total_probes_sent"]
        result["total_probes"] += diag["total_probes_sent"]

        if diag["is_frp"]:
            cand_result["verified_frp"] = True
            logger.info("%s[TA-L%d] \u2713 %s confirmed as FRP (refine deferred)",
                        indent, _current_depth, cand_prefix)

            # Add to known_frps so later candidates (this call or nested
            # recursion) can skip anything it covers.
            known_frps.add(cand_prefix)

            # Defer refinement to the top-level drain step.
            pending = {
                "prefix": cand_prefix,
                "original_candidate": cand_prefix,
                "source_prefix": prefix,
                "depth": _current_depth,
                "refinement": None,
            }
            _defer_refine.append(pending)
            result["new_frps"].append(pending)
        else:
            # Candidate is NOT FRP
            cand_result["diag_detail"] = {
                "reachable_count": diag["reachable_count"],
                "reachable_nibbles": diag["reachable_nibbles"],
                "unreachable_nibbles": diag["unreachable_nibbles"],
            }

            # Recursive tree alignment: if candidate has reachable nibbles
            # and we haven't exceeded max depth, recurse
            if (diag["reachable_count"] > 0
                    and _current_depth < max_depth
                    and cand["prefix_len"] + 4 <= 128):
                logger.info(
                    "%s[TA-L%d] \u2717 %s NOT FRP (%d/16 reachable), "
                    "recursing deeper (depth %d/%d)...",
                    indent, _current_depth, cand_prefix,
                    diag["reachable_count"],
                    _current_depth + 1, max_depth,
                )
                sub_ta = tree_align_prefix(
                    cand_prefix,
                    diag,
                    known_frps,
                    timeout=timeout,
                    max_attempts=max_attempts,
                    min_prefix_len=min_prefix_len,
                    concurrency=concurrency,
                    refine=refine,
                    max_depth=max_depth,
                    _current_depth=_current_depth + 1,
                    fast_refine=fast_refine,
                    refine_mode=refine_mode,
                    refine_timeout=refine_timeout,
                    refine_max_attempts=refine_max_attempts,
                    coverage_checker=coverage_checker,
                    _defer_refine=_defer_refine,
                )
                cand_result["recursive_ta"] = sub_ta
                result["total_probes"] += sub_ta["total_probes"]
                # Merge newly found FRPs from recursive call into this level's
                # summary (they're already in _defer_refine, don't add twice).
                for new_frp in sub_ta["new_frps"]:
                    result["new_frps"].append(new_frp)
            else:
                if _current_depth >= max_depth:
                    logger.info(
                        "%s[TA-L%d] \u2717 %s NOT FRP (%d/16 reachable), "
                        "max depth reached, stopping.",
                        indent, _current_depth, cand_prefix,
                        diag["reachable_count"],
                    )
                else:
                    logger.info(
                        "%s[TA-L%d] \u2717 %s NOT FRP (%d/16 reachable)",
                        indent, _current_depth, cand_prefix,
                        diag["reachable_count"],
                    )

        result["candidates"].append(cand_result)

    # At the top level: drain deferred refinement (only on maximal FRPs).
    if is_top_level and _defer_refine:
        _drain_deferred_refine(
            _defer_refine, result, known_frps,
            timeout=timeout,
            max_attempts=max_attempts,
            min_prefix_len=min_prefix_len,
            concurrency=concurrency,
            refine=refine,
            refine_mode=refine_mode,
            refine_timeout=refine_timeout,
            refine_max_attempts=refine_max_attempts,
            coverage_checker=coverage_checker,
            fast_refine=fast_refine,
        )

    return result


def _drain_deferred_refine(
    pending_list,
    result,
    known_frps,
    timeout,
    max_attempts,
    min_prefix_len,
    concurrency,
    refine,
    refine_mode,
    refine_timeout,
    refine_max_attempts,
    coverage_checker,
    fast_refine,
):
    """Refine only the maximal (broadest) deferred FRPs.

    ``pending_list`` is a list of pending FRP dicts (mutated in-place; each
    dict has ``prefix``, ``refinement`` fields). We:

    1. Compute the maximal set: drop any FRP that is covered by another
       FRP in the same list.
    2. For each surviving FRP, run refine (if enabled). Others are left
       as-is (their prefix stays what tree-align found).
    3. Update result["total_probes"] with any refinement probes.
    """
    if not pending_list:
        return

    # Compute maximal set. Sort by prefix_len ascending so broader (shorter)
    # come first; a subsequent longer one is covered iff it lies within one
    # of the shorter accepted ones.
    parsed = []
    for pf in pending_list:
        try:
            net = ipaddress.IPv6Network(pf["prefix"], strict=False)
            parsed.append((pf, net))
        except Exception:
            continue
    parsed.sort(key=lambda x: x[1].prefixlen)

    maximal = []       # list of (pf, net) that are NOT covered by any earlier maximal
    covered_by_map = {}
    for pf, net in parsed:
        covered = False
        for mpf, mnet in maximal:
            if is_covered_by(net, mnet):
                covered = True
                covered_by_map[id(pf)] = mpf["prefix"]
                break
        if not covered:
            maximal.append((pf, net))

    # Also drop any that are covered by an already-known external FRP via
    # coverage_checker (some other worker may have found a broader one).
    truly_maximal = []
    for pf, net in maximal:
        cov = None
        if coverage_checker is not None:
            try:
                cov = coverage_checker(net)
            except Exception:
                cov = None
        if cov is not None and cov != pf["prefix"]:
            covered_by_map[id(pf)] = cov
            logger.info("  [TA] pending FRP %s superseded by live FRP %s (skip refine)",
                        pf["prefix"], cov)
        else:
            truly_maximal.append((pf, net))

    logger.info(
        "  [TA] Deferred-refine drain: %d pending -> %d maximal (refine=%s)",
        len(pending_list), len(truly_maximal), refine,
    )

    for pf, net in truly_maximal:
        if not refine:
            continue
        if net.prefixlen <= min_prefix_len:
            continue
        cand_prefix = pf["prefix"]
        logger.info("  [TA] Refining maximal FRP %s ...", cand_prefix)
        try:
            ref = refine_frp_prefix(
                cand_prefix,
                timeout=timeout,
                max_attempts=max_attempts,
                min_prefix_len=min_prefix_len,
                concurrency=concurrency,
                fast_refine=fast_refine,
                refine_mode=refine_mode,
                refine_timeout=refine_timeout,
                refine_max_attempts=refine_max_attempts,
                coverage_checker=coverage_checker,
            )
        except Exception as e:
            logger.warning("  [TA] refine failed for %s: %s", cand_prefix, e)
            continue

        pf["prefix"] = ref["refined"]
        pf["refinement"] = ref
        result["total_probes"] += ref["total_refinement_probes"]
        known_frps.add(ref["refined"])

    # Mark covered pending FRPs with a note so callers can see what happened.
    for pf in pending_list:
        if id(pf) in covered_by_map:
            pf["covered_by_sibling"] = covered_by_map[id(pf)]


# ---------------------------------------------------------------------------
# Coverage helper
# ---------------------------------------------------------------------------


def is_covered_by(child_net, parent_net):
    """Return True iff child_net is fully contained in parent_net."""
    if child_net.prefixlen < parent_net.prefixlen:
        return False
    return (int(child_net.network_address) & int(parent_net.netmask)) == int(
        parent_net.network_address
    )


# ---------------------------------------------------------------------------
# Legacy compatibility wrappers
# ---------------------------------------------------------------------------


def diagnose_prefix_multi_round(
    prefix, rounds=1, timeout=3.0, probe_count=16,
    inter_probe_delay=0.05, inter_round_delay=5.0, concurrency=1,
):
    """Legacy wrapper - now just calls diagnose_prefix once with ping-style."""
    return diagnose_prefix(
        prefix, timeout=timeout, probe_count=probe_count,
        max_attempts=5, concurrency=concurrency,
    )


__all__ = [
    # constants
    "ICMPV6_ECHO_REQUEST",
    "ICMPV6_ECHO_REPLY",
    "ICMPV6_DEST_UNREACHABLE",
    "ICMPV6_PACKET_TOO_BIG",
    "ICMPV6_TIME_EXCEEDED",
    "ICMPV6_PARAM_PROBLEM",
    "ICMPV6_TYPE_NAMES",
    "ICMPV6_DEST_UNREACH_CODES",
    # capability check
    "has_raw_socket_capability",
    # core functions
    "classify_response",
    "ping_probe_address",
    "generate_probe_addresses",
    "generate_random_addresses",
    "diagnose_prefix",
    "diagnose_prefix_multi_round",
    "determine_frp_status",
    "refine_frp_prefix",
    # tree alignment
    "find_candidate_sub_frps",
    "tree_align_prefix",
    # helpers
    "is_covered_by",
]
