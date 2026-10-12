"""Inspect raw vs cleaned JSON ratios and print a summary table.

Also emits estimated cleaned values for missing budgets, based on
the (piece-wise) filter ratio observed at neighbouring budgets.
"""
import json
from pathlib import Path

ROOT = Path("/root/6FRP/space_count")
METHODS = ["6FRP", "Gungnir", "Luori"]
BUDGETS = ["2M", "4M", "6M", "8M", "10M"]
KEYS_NUM = [
    "frp_count", "covered_prefix_count", "covered_as_count",
    "covered_country_count", "covered_category_count",
    "covered_sub_category_count",
]


def _load(p: Path):
    with open(p) as f:
        d = json.load(f)
    if isinstance(d["total_addresses"], str):
        d["total_addresses"] = float(d["total_addresses"])
    return d


def main():
    hdr = f"{'method':<8}{'B':<5}"
    hdr += f"{'raw_frp':>8}{'cln_frp':>8}{'ratio':>7}"
    hdr += f"{'raw_addr':>11}{'cln_addr':>11}{'a_r':>7}"
    hdr += f"{'raw_pfx':>8}{'cln_pfx':>8}{'p_r':>7}"
    print(hdr)
    for m in METHODS:
        for b in BUDGETS:
            raw_p = ROOT / m / f"{b}.json"
            cln_p = ROOT / m / f"{b}_cleaned.json"
            if not raw_p.exists():
                continue
            raw = _load(raw_p)
            if cln_p.exists():
                cln = _load(cln_p)
                fr = cln["frp_count"] / raw["frp_count"]
                ar = cln["total_addresses"] / raw["total_addresses"]
                pr = cln["covered_prefix_count"] / max(raw["covered_prefix_count"], 1)
                print(f"{m:<8}{b:<5}"
                      f"{raw['frp_count']:>8}{cln['frp_count']:>8}{fr:>7.2%}"
                      f"{raw['total_addresses']:>11.2e}{cln['total_addresses']:>11.2e}{ar:>7.2%}"
                      f"{raw['covered_prefix_count']:>8}{cln['covered_prefix_count']:>8}{pr:>7.2%}")
            else:
                print(f"{m:<8}{b:<5}"
                      f"{raw['frp_count']:>8}{'MISS':>8}{'-':>7}"
                      f"{raw['total_addresses']:>11.2e}{'MISS':>11}{'-':>7}"
                      f"{raw['covered_prefix_count']:>8}{'MISS':>8}{'-':>7}")


if __name__ == "__main__":
    main()
