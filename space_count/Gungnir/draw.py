#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot metric curves from multiple JSON files.
Each JSON file may contain any of the following fields:
  frp_count, total_addresses, covered_prefix_count, covered_as_count,
  covered_country_count, covered_category_count, covered_sub_category_count
"""

import json
import os
import argparse
import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt

# ===================== Configurable ==============# ...省略其他代码... 


# ==== 修改后代码 ====
# All supported metrics and their English display names
ALL_KEYS = {
    "frp_count":                    "FRP Count",
    "total_addresses":              "Total Addresses",
    "covered_prefix_count":         "Covered Prefix Count",
    "covered_as_count":             "Covered AS Count",
    "covered_country_count":        "Covered Country Count",
    "covered_category_count":       "Covered Category Count",
    "covered_sub_category_count":   "Covered Sub-Category Count",
}

# Pick metrics to plot (edit as you like)
SELECTED_KEYS = [
    "frp_count",
    "covered_prefix_count",
    "covered_as_count",
    "covered_country_count",
    "covered_category_count",
    "covered_sub_category_count",
    "total_addresses",
]
# =================================================# ...省略其他代码... 


# ==== 修改后代码 ====


def load_data(json_files, labels=None):
    """Load metrics from the given JSON files in the given order."""
    if not json_files:
        raise SystemExit("No JSON files provided.")

    if labels and len(labels) != len(json_files):
        raise SystemExit("--labels count must match --inputs count.")

    x_labels = []
    series = {k: [] for k in ALL_KEYS}

    for idx, f in enumerate(json_files):
        if not os.path.isfile(f):
            raise SystemExit(f"File not found: {f}")
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        if labels:
            x_labels.append(labels[idx])
        else:
            x_labels.append(os.path.splitext(os.path.basename(f))[0])

        for k in ALL_KEYS:
            series[k].append(data.get(k, 0))

    return x_labels, series


def plot_separate(x_labels, series, keys, out_path):
    """One subplot per metric."""
    n = len(keys)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4), squeeze=False)
    fig.suptitle("Metric Trends", fontsize=16)

    for i, k in enumerate(keys):
        ax = axes[i // cols][i % cols]
        ax.plot(range(len(x_labels)), series[k],
                marker="o", linewidth=2, label=ALL_KEYS[k])
        ax.set_title(ALL_KEYS[k])
        ax.set_xlabel("Sample")
        ax.set_ylabel(ALL_KEYS[k])
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.legend(loc="best")

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")


def plot_combined(x_labels, series, keys, out_path):
    """All metrics on the same figure (log-scale Y to handle huge total_addresses)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for k in keys:
        ax.plot(range(len(x_labels)), series[k],
                marker="o", linewidth=2, label=ALL_KEYS[k])

    ax.set_title("Metric Trends (Combined)")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value (log scale)")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot metric curves from user-specified JSON files (order preserved)."
    )
    parser.add_argument(
        "-i", "--inputs", nargs="+", required=True,
        help="JSON files to plot, in the order you want them shown on the X axis."
    )
    parser.add_argument(
        "-l", "--labels", nargs="+", default=None,
        help="Optional X-axis labels, one per input file (same count as --inputs)."
    )
    parser.add_argument(
        "-k", "--keys", nargs="+", default=None,
        help=f"Metrics to plot. Choices: {list(ALL_KEYS.keys())}. "
             f"Default uses SELECTED_KEYS in the script."
    )
    parser.add_argument(
        "-o", "--out", default="metrics_curve.png",
        help="Output image path."
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Plot all metrics on a single figure (log-scaled Y)."
    )
    args = parser.parse_args()

    keys = args.keys if args.keys else SELECTED_KEYS
    unknown = [k for k in keys if k not in ALL_KEYS]
    if unknown:
        raise SystemExit(f"Unknown keys: {unknown}. Valid: {list(ALL_KEYS.keys())}")

    x_labels, series = load_data(args.inputs, args.labels)

    if args.combined:
        plot_combined(x_labels, series, keys, args.out)
    else:
        plot_separate(x_labels, series, keys, args.out)


if __name__ == "__main__":
    main()