"""
APD (Aliased Prefix Detection) probing module — single-server version.

This module provides:
    * ``APDResult``           — probe result dataclass.
    * ``APDProber``           — abstract base class.
    * ``LocalAPDProber``      — direct ICMPv6 probing via apd_core (requires root).
    * ``SimulatedAPDProber``  — mock prober for offline testing.

Since the entire system runs on a single server with IPv6 global address,
there is no need for HTTP communication. LocalAPDProber calls apd_core
functions directly with concurrent in-prefix probing.
"""
import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .ipv6_utils import generate_apd_addresses, is_subnet_of, normalize_prefix

logger = logging.getLogger(__name__)


class APDResult(object):
    """APD probe result."""
    def __init__(self, prefix, is_frp, success_count, total_count, success_rate,
                 rtt_mean=None, rtt_std=None, rtt_min=None,
                 classification_summary=None):
        self.prefix = prefix
        self.is_frp = is_frp
        self.success_count = success_count
        self.total_count = total_count
        self.success_rate = success_rate
        self.rtt_mean = rtt_mean
        self.rtt_std = rtt_std
        self.rtt_min = rtt_min
        # ICMPv6 classification breakdown (e.g. {"SUCCESS": 16} or {"TIMEOUT": 10, "SUCCESS": 6})
        self.classification_summary = classification_summary


class APDProber:
    """Abstract base class for APD probing."""

    def __init__(self, probe_count: int = 16, timeout: float = 3.0,
                 success_rate_threshold: float = 0.875):
        self.probe_count = probe_count
        self.timeout = timeout
        self.success_rate_threshold = success_rate_threshold

    def _probe_address(self, address: str) -> Tuple[bool, Optional[float]]:
        """Probe a single address. Returns (success, rtt_ms)."""
        raise NotImplementedError

    def probe_prefix(self, prefix: str, extra_rounds: int = 0) -> APDResult:
        """Probe a prefix with probe_count addresses."""
        addresses = generate_apd_addresses(prefix, self.probe_count)
        successes = 0
        rtts = []

        for addr in addresses:
            success, rtt = self._probe_address(addr)
            if success:
                successes += 1
                if rtt is not None:
                    rtts.append(rtt)

        rate = successes / self.probe_count if self.probe_count > 0 else 0
        rtt_mean = sum(rtts) / len(rtts) if rtts else None
        rtt_min = min(rtts) if rtts else None

        return APDResult(
            prefix=normalize_prefix(prefix),
            is_frp=(successes == self.probe_count),
            success_count=successes,
            total_count=self.probe_count,
            success_rate=rate,
            rtt_mean=rtt_mean,
            rtt_min=rtt_min,
        )


class LocalAPDProber(APDProber):
    """
    Direct ICMPv6 probing via apd_core — runs on the same server.

    This is the primary prober for the single-server architecture.
    It calls apd_core.diagnose_prefix with concurrency support for
    parallel in-prefix probing (16 packets sent concurrently).

    Requires: root privileges or CAP_NET_RAW capability.
    """

    def __init__(self, probe_count: int = 16, timeout: float = 3.0,
                 success_rate_threshold: float = 0.875,
                 concurrency: int = 16,
                 inter_probe_delay: float = 0.05):
        super().__init__(probe_count, timeout, success_rate_threshold)
        self.concurrency = concurrency
        self.inter_probe_delay = inter_probe_delay

        # Import apd_core lazily to allow SimulatedAPDProber usage without root
        from . import apd_core
        self._core = apd_core

    def _probe_address(self, address: str) -> Tuple[bool, Optional[float]]:
        """Probe a single address using raw ICMPv6 socket."""
        result = self._core.probe_single_address(address, timeout=self.timeout)
        return bool(result.get("success")), result.get("response_time_ms")

    def probe_prefix(self, prefix: str, extra_rounds: int = 0) -> APDResult:
        """
        Probe a prefix using concurrent in-prefix probing.

        Uses apd_core.diagnose_prefix with concurrency parameter for
        parallel packet sending. Optionally runs extra rounds for
        borderline cases (success_rate >= 7/8 but < 16/16).
        """
        result = self._core.diagnose_prefix(
            prefix,
            timeout=self.timeout,
            probe_count=self.probe_count,
            inter_probe_delay=self.inter_probe_delay,
            concurrency=self.concurrency,
        )

        is_frp = bool(result["is_frp"])
        success_count = int(result["success_count"])
        success_rate = float(result["success_rate"])

        # Extra rounds for borderline cases
        if not is_frp and extra_rounds > 0 and success_rate >= 7 / 8:
            for _ in range(extra_rounds):
                time.sleep(1.0)  # brief pause between rounds
                extra_result = self._core.diagnose_prefix(
                    prefix,
                    timeout=self.timeout,
                    probe_count=self.probe_count,
                    inter_probe_delay=self.inter_probe_delay,
                    concurrency=self.concurrency,
                )
                if extra_result["is_frp"]:
                    is_frp = True
                    success_count = int(extra_result["success_count"])
                    success_rate = float(extra_result["success_rate"])
                    result = extra_result
                    break

        rtt_stats = result.get("rtt_stats") or {}

        return APDResult(
            prefix=normalize_prefix(prefix),
            is_frp=is_frp,
            success_count=success_count,
            total_count=self.probe_count,
            success_rate=success_rate,
            rtt_mean=rtt_stats.get("avg_ms"),
            rtt_min=rtt_stats.get("min_ms"),
            classification_summary=result.get("classification_summary"),
        )


class SimulatedAPDProber(APDProber):
    """
    Simulated prober for offline testing / training.

    Uses a list of known FRP prefixes to simulate probe results.
    """

    def __init__(self, known_frps: List[str], probe_count: int = 16,
                 timeout: float = 3.0, success_rate_threshold: float = 0.875,
                 noise_prob: float = 0.02):
        super().__init__(probe_count, timeout, success_rate_threshold)
        self.known_frps = [normalize_prefix(p) for p in known_frps]
        self.noise_prob = noise_prob

    def _is_in_frp(self, address: str) -> bool:
        """Check if address falls within any known FRP."""
        for frp in self.known_frps:
            if is_subnet_of(f"{address}/128", frp):
                return True
        return False

    def _probe_address(self, address: str) -> Tuple[bool, Optional[float]]:
        covered = self._is_in_frp(address)
        if covered:
            success = random.random() > self.noise_prob
            rtt = random.gauss(200, 30) if success else None
        else:
            success = random.random() < 0.01
            rtt = random.gauss(500, 100) if success else None
        return success, rtt


__all__ = [
    "APDResult",
    "APDProber",
    "LocalAPDProber",
    "SimulatedAPDProber",
]
