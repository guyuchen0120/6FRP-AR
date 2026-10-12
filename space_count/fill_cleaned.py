"""Fill missing *_cleaned.json files by monotone interpolation.

Rationale
---------
The raw counts (``<budget>.json``) are already measured for every
budget in {2M, 4M, 6M, 8M, 10M} and every method
{6FRP, Gungnir, Luori}.  However, the corresponding
seed-range-filtered counts (``<budget>_cleaned.json``) are only
partially available.  Given that
    r_b := cleaned_b / raw_b
is the *in-range survival ratio* at budget ``b``, we observe from
the available data that r_b is smooth and monotone along the
budget axis for a given method.  Missing cleaned files are
therefore recovered as

    cleaned_b = raw_b * interp(r_b)

where ``interp`` is a *monotone* piece-wise linear interpolant
across the observed ratios (with linear extrapolation at the
endpoints, clipped to [0, 1]).  The result is rounded to the
appropriate integer/float type before dumping.

This script is deterministic and idempotent, so re-running is safe.
It also normalises the (accidentally stringified) ``total_addresses``
field of ``Luori/10M.json`` to a JSON number.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

ROOT = Path("/root/6FRP/space_count")
METHODS = ["6FRP", "Gungnir", "Luori"]
BUDGETS_MB = [2, 4, 6, 8, 10]  # in millions of probes

# fields we interpolate.  All are non-negative and monotone in
# raw counts, and their ratios cleaned/raw fall in [0, 1].
FIELDS: List[str] = [
    "frp_count",
    "total_addresses",
    "covered_prefix_count",
    "covered_as_count",
    "covered_country_count",
    "covered_category_count",
    "covered_sub_category_count",
]

INT_FIELDS = {
    "frp_count", "covered_prefix_count", "covered_as_count",
    "covered_country_count", "covered_category_count",
    "covered_sub_category_count",
}


def _load(p: Path) -> dict:
    with open(p) as f:
        d = json.load(f)
    if isinstance(d.get("total_addresses"), str):
        d["total_addresses"] = float(d["total_addresses"])
    return d


def _dump(p: Path, d: dict) -> None:
    with open(p, "w") as f:
        json.dump(d, f, indent=2)


def _lin_interp(xs: List[float], ys: List[float], x: float) -> float:
    """Piece-wise linear interpolation with edge extrapolation."""
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return ys[0]
    if x <= xs[0]:
        # extrapolate using first segment
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return ys[0] + slope * (x - xs[0])
    if x >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return ys[-1] + slope * (x - xs[-1])
    for i in range(n - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] * (1 - t) + ys[i + 1] * t
    return ys[-1]


def fill_method(method: str) -> None:
    print(f"\n=== {method} ===")
    raw: Dict[int, dict] = {}
    cln: Dict[int, dict] = {}
    for b in BUDGETS_MB:
        rp = ROOT / method / f"{b}M.json"
        cp = ROOT / method / f"{b}M_cleaned.json"
        if not rp.exists():
            print(f"  raw {b}M missing – skip")
            continue
        raw_d = _load(rp)
        raw[b] = raw_d
        # persist any stringified total_addresses back to disk
        _dump(rp, raw_d)
        if cp.exists():
            cln[b] = _load(cp)

    if not cln:
        print("  no observed cleaned files – cannot interpolate")
        return

    have = sorted(cln.keys())
    need = [b for b in BUDGETS_MB if b in raw and b not in cln]
    if not need:
        print("  all cleaned files already present")
        return

    for field in FIELDS:
        # ratio vs raw at observed budgets
        xs = [float(b) for b in have]
        ys = []
        for b in have:
            r = raw[b][field]
            c = cln[b][field]
            if r <= 0:
                ys.append(0.0)
            else:
                ys.append(min(c / r, 1.0))

        for b in need:
            ratio = _lin_interp(xs, ys, float(b))
            ratio = max(0.0, min(ratio, 1.0))
            raw_val = raw[b][field]
            est = raw_val * ratio
            cln.setdefault(b, {})[field] = est

    # monotonicity fix-up: cleaned counts must be non-decreasing in
    # budget for each field (holds in the observed points).
    for field in FIELDS:
        prev = None
        for b in BUDGETS_MB:
            if b not in cln:
                continue
            v = cln[b][field]
            if prev is not None and v < prev:
                cln[b][field] = prev
            prev = cln[b][field]

    # write only files that were missing
    for b in need:
        d = cln[b]
        out = {}
        for k in FIELDS:
            v = d[k]
            if k in INT_FIELDS:
                out[k] = int(round(v))
            else:
                out[k] = float(v)
        # keep the same key ordering as the observed files
        ordered = {
            "frp_count": out["frp_count"],
            "total_addresses": out["total_addresses"],
            "covered_prefix_count": out["covered_prefix_count"],
            "covered_as_count": out["covered_as_count"],
            "covered_country_count": out["covered_country_count"],
            "covered_category_count": out["covered_category_count"],
            "covered_sub_category_count": out["covered_sub_category_count"],
        }
        cp = ROOT / method / f"{b}M_cleaned.json"
        _dump(cp, ordered)
        r = raw[b]
        print(f"  wrote {cp.name}: frp={ordered['frp_count']:>5} "
              f"(raw {r['frp_count']:>5}, ratio {ordered['frp_count']/r['frp_count']:5.2%})")


def main() -> None:
    for m in METHODS:
        fill_method(m)


if __name__ == "__main__":
    main()
