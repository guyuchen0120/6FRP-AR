#!/usr/bin/env python3
"""
Plot NET-NEW comparison curves of Heimdall vs. Luori vs. Gungnir under
identical probing budgets (2M / 4M / 6M / 8M / 10M).

"Net-new" = FRPs whose prefixes are NOT covered by the aligned public seed
set S_0 (Luori's released FRP set aligned to the RouteViews view).

Reads:
    ./6FRP/{2M,4M,6M,8M,10M}_cleaned.json   (Heimdall training results)
    ./Luori/{2M,4M,6M,8M,10M}_cleaned.json
    ./Gungnir/{2M,4M,6M,8M,10M}_cleaned.json

Outputs:
    ./comparison_curves_netnew.png
    ./comparison_curves_netnew.pdf
    ./frp_count_netnew.pdf
    ./address_space_netnew.pdf
    ./as_coverage_netnew.pdf
    ./country_coverage_netnew.pdf
    ./prefix_coverage_netnew.pdf
    ./category_coverage_netnew.pdf
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
BUDGETS = ["2M", "4M", "6M", "8M", "10M"]
BUDGET_VALUES = [2, 4, 6, 8, 10]  # 单位: 百万探测包
METHODS = {
    "Luori":   {"dir": "Luori",   "color": "#1f77b4", "marker": "s", "ls": "--"},
    "Gungnir": {"dir": "Gungnir", "color": "#ff7f0e", "marker": "^", "ls": "-."},
    "Heimdall (Ours)": {"dir": "6FRP", "color": "#d62728", "marker": "o", "ls": "-"},
}

METRICS = [
    ("frp_count",              "Net-new confirmed FRPs",        False),
    ("total_addresses",        "Net-new address space",         True),
    ("covered_prefix_count",   "Net-new BGP prefixes",          False),
    ("covered_as_count",       "Net-new ASes",                  False),
    ("covered_country_count",  "Net-new countries",             False),
    ("covered_category_count", "Net-new AS categories",         False),
]

def load_series(method_dir: str) -> dict:
    series = {m[0]: [] for m in METRICS}
    for budget in BUDGETS:
        path = HERE / method_dir / f"{budget}_cleaned.json"
        with open(path) as f:
            data = json.load(f)
        for metric, _, _ in METRICS:
            v = data.get(metric, 0)
            if isinstance(v, str):
                v = float(v)
            series[metric].append(v)
    return series

def plot_single(ax, metric_key: str, ylabel: str, use_log: bool, all_data: dict):
    for name, info in METHODS.items():
        y = all_data[name][metric_key]
        ax.plot(
            BUDGET_VALUES, y,
            color=info["color"], marker=info["marker"], linestyle=info["ls"],
            linewidth=2.6, markersize=9, label=name,
        )
    ax.set_xlabel("Probing budget (million probes)", fontsize=17)
    ax.set_ylabel(ylabel, fontsize=17)
    ax.set_xticks(BUDGET_VALUES)
    ax.tick_params(axis="both", labelsize=15)
    ax.grid(True, linestyle=":", alpha=0.6)
    if use_log:
        ax.set_yscale("log")
    ax.legend(loc="best", fontsize=13, framealpha=0.9)

def main():
    all_data = {name: load_series(info["dir"]) for name, info in METHODS.items()}

    print("\n===== NET-NEW summary =====")
    for metric, ylabel, _ in METRICS:
        print(f"\n[{ylabel}]  (rows=method, cols=budget)")
        header = "method \\ budget | " + " | ".join(BUDGETS)
        print(header)
        print("-" * len(header))
        for name in METHODS:
            vals = all_data[name][metric]
            if metric == "total_addresses":
                cells = " | ".join(f"{v:.2e}" for v in vals)
            else:
                cells = " | ".join(f"{int(v)}" for v in vals)
            print(f"{name:<18} | {cells}")

    # ---- 合并大图（2 x 3 subplot） -----------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()
    for ax, (metric, ylabel, use_log) in zip(axes, METRICS):
        plot_single(ax, metric, ylabel, use_log, all_data)

    fig.suptitle(
        "Net-new discoveries (outside the aligned seed set $\\mathcal{S}_0$)"
        " under identical probing budgets",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_png = HERE / "comparison_curves_netnew.png"
    out_pdf = HERE / "comparison_curves_netnew.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")

    # ---- 6 张单图 ------------------------------------------------------------
    single_names = {
        "frp_count":              "frp_count_netnew",
        "total_addresses":        "address_space_netnew",
        "covered_prefix_count":   "prefix_coverage_netnew",
        "covered_as_count":       "as_coverage_netnew",
        "covered_country_count":  "country_coverage_netnew",
        "covered_category_count": "category_coverage_netnew",
    }
    # Flatter aspect ratio (5.6 x 2.6, ~2.15:1) with 17pt axis labels and
    # 15pt ticks; tight_layout(pad=0.2) keeps the caption close to the frame.
    for metric, ylabel, use_log in METRICS:
        fig, ax = plt.subplots(figsize=(5.6, 2.6))
        plot_single(ax, metric, ylabel, use_log, all_data)
        fig.tight_layout(pad=0.2)
        pdf = HERE / f"{single_names[metric]}.pdf"
        png = HERE / f"{single_names[metric]}.png"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {pdf}")

if __name__ == "__main__":
    main()
