#!/usr/bin/env python3
"""
Plot comparison curves of Heimdall vs. Luori vs. Gungnir under identical
probing budgets (2M / 4M / 6M / 8M / 10M).

Reads:
    ./6FRP/{2M,4M,6M,8M,10M}.json    (Heimdall training results)
    ./Luori/{2M,4M,6M,8M,10M}.json
    ./Gungnir/{2M,4M,6M,8M,10M}.json

Outputs:
    ./comparison_curves.png   (6 sub-plots on one canvas)
    ./comparison_curves.pdf   (vector version for the paper)
    ./frp_count.pdf           (single-panel figures, INFOCOM-friendly)
    ./address_space.pdf
    ./as_coverage.pdf
    ./country_coverage.pdf
    ./prefix_coverage.pdf
    ./category_coverage.pdf
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
    ("frp_count",              "Confirmed FRPs",           False),
    ("total_addresses",        "Covered address space",    True),   # log 轴
    ("covered_prefix_count",   "Covered BGP prefixes",     False),
    ("covered_as_count",       "Covered ASes",             False),
    ("covered_country_count",  "Covered countries",        False),
    ("covered_category_count", "Covered AS categories",    False),
]


def load_series(method_dir: str) -> dict:
    """读取一个方法下 5 个预算的 JSON 结果，返回 {metric: [values...]}"""
    series = {m[0]: [] for m in METRICS}
    for budget in BUDGETS:
        path = HERE / method_dir / f"{budget}.json"
        with open(path) as f:
            data = json.load(f)
        for metric, _, _ in METRICS:
            v = data.get(metric, 0)
            # 兼容 "7.83e+30" 这种字符串
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

    # ---- 打印一个 markdown 汇总表，方便投稿写作 -----------------------------
    print("\n===== summary =====")
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

    # ---- 合并大图（2 x 3 subplot） --------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    axes = axes.flatten()
    for ax, (metric, ylabel, use_log) in zip(axes, METRICS):
        plot_single(ax, metric, ylabel, use_log, all_data)

    fig.suptitle(
        "Heimdall vs. Luori vs. Gungnir under identical probing budgets",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out_png = HERE / "comparison_curves.png"
    out_pdf = HERE / "comparison_curves.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")

    # ---- 单独输出 6 张小图（论文里方便单独嵌入） -----------------------------
    single_names = {
        "frp_count":              "frp_count",
        "total_addresses":        "address_space",
        "covered_prefix_count":   "prefix_coverage",
        "covered_as_count":       "as_coverage",
        "covered_country_count":  "country_coverage",
        "covered_category_count": "category_coverage",
    }
    # Flatter aspect ratio (5.6 x 2.6, ~2.15:1) so the panel takes less
    # vertical room in the paper; enlarged axis-label fonts (17pt) and
    # tick fonts (15pt) compensate for the reduced height so readability
    # is preserved.  tight_layout(pad=0.2) keeps the caption close to
    # the axis frame.
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
