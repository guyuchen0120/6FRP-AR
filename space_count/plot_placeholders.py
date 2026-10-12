#!/usr/bin/env python3
"""
Plot generators for figures referenced in 6FRP_infocom2027.tex :

  1. seed_overlap.pdf  --  Raw vs net-new FRPs at the 10M budget
     (single-budget three-way comparison).

  2. netnew_curve.pdf  --  Net-new FRP count as a function of the
     probing budget (2M .. 10M), for 6FRP / Gungnir / Luori.
     This is the figure that visualises the fact that 6FRP's
     in-range-aware feedback pays off increasingly as the budget
     grows: 6FRP starts below Gungnir at 2M and ends 3.23x above
     it at 10M.

  3. arch_dualscan.pdf --  Very simple block diagram of the 6FRP
     architecture, so LaTeX compiles cleanly.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).resolve().parent
BUDGETS_MB = [2, 4, 6, 8, 10]
METHODS = ["6FRP", "Gungnir", "Luori"]
DISPLAY = {"6FRP": "Heimdall", "Gungnir": "Gungnir", "Luori": "Luori"}
COLORS = {
    "6FRP":    "#d62728",
    "Gungnir": "#ff7f0e",
    "Luori":   "#1f77b4",
}
MARKERS = {"6FRP": "o", "Gungnir": "s", "Luori": "^"}


def _load(method: str, budget: int, cleaned: bool):
    suf = "_cleaned" if cleaned else ""
    p = HERE / method / f"{budget}M{suf}.json"
    with open(p, "r") as f:
        d = json.load(f)
    if isinstance(d.get("total_addresses"), str):
        d["total_addresses"] = float(d["total_addresses"])
    return d


def _collect():
    data = {"raw": {}, "cln": {}}
    for m in METHODS:
        data["raw"][m] = [_load(m, b, False) for b in BUDGETS_MB]
        data["cln"][m] = [_load(m, b, True) for b in BUDGETS_MB]
    return data


# ---------------------------------------------------------------------------
# 1. In-range redundancy bar chart at 10M
# ---------------------------------------------------------------------------
def plot_seed_overlap(data):
    methods = METHODS
    raw = {m: data["raw"][m][-1]["frp_count"] for m in methods}
    net = {m: data["cln"][m][-1]["frp_count"] for m in methods}
    redund = {m: 1.0 - net[m] / raw[m] for m in methods}

    print("=== 10M budget ===")
    for m in methods:
        print(f"  {m:>8}: raw={raw[m]:>5}  net-new={net[m]:>5}  "
              f"in-range={redund[m]*100:5.2f}%")

    labels = [DISPLAY[m] for m in methods]
    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(6.4, 3.2))

    bar_raw = ax.bar(x - width / 2, [raw[m] for m in methods], width,
                     label="Raw FRPs (all discovered)",
                     color="#a6cee3", edgecolor="black", linewidth=0.6)
    bar_net = ax.bar(x + width / 2, [net[m] for m in methods], width,
                     label="Net-new FRPs (outside seed range)",
                     color="#d62728", edgecolor="black", linewidth=0.6)

    ymax = max(raw.values())
    for b, v in zip(bar_raw, [raw[m] for m in methods]):
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02,
                f"{v}", ha="center", va="bottom", fontsize=11)
    for b, v in zip(bar_net, [net[m] for m in methods]):
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02,
                f"{v}", ha="center", va="bottom", fontsize=11)

    for i, m in enumerate(methods):
        ax.text(i, max(raw[m], net[m]) + ymax * 0.20,
                f"in-range: {redund[m]*100:.1f}%",
                ha="center", fontsize=10.5,
                color="dimgray", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("Number of FRPs @ 10M probe budget", fontsize=13)
    ax.set_ylim(0, ymax * 1.55)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "seed_overlap.pdf", bbox_inches="tight")
    fig.savefig(HERE / "seed_overlap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: seed_overlap.pdf")


# ---------------------------------------------------------------------------
# 2. Net-new FRPs vs budget curve
# ---------------------------------------------------------------------------
def plot_netnew_curve(data):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for m in METHODS:
        y = [data["cln"][m][i]["frp_count"] for i in range(len(BUDGETS_MB))]
        ax.plot(BUDGETS_MB, y, marker=MARKERS[m], color=COLORS[m],
                linewidth=2.4, markersize=8,
                label=DISPLAY[m] + (" (Ours)" if m == "6FRP" else ""))
        for x, v in zip(BUDGETS_MB, y):
            ax.text(x, v + 60, f"{v}", ha="center", va="bottom",
                    fontsize=9.5, color=COLORS[m])
    ax.set_xlabel("Probe budget (millions of packets)", fontsize=14)
    ax.set_ylabel("Net-new FRPs", fontsize=14)
    ax.set_xticks(BUDGETS_MB)
    ax.set_xticklabels([f"{b}M" for b in BUDGETS_MB], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=12)
    ax.set_ylim(0, max(data["cln"]["6FRP"][-1]["frp_count"],
                       data["cln"]["Luori"][-1]["frp_count"]) * 1.20)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "netnew_curve.pdf", bbox_inches="tight")
    fig.savefig(HERE / "netnew_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: netnew_curve.pdf")


# ---------------------------------------------------------------------------
# 3. Simple architecture placeholder
# ---------------------------------------------------------------------------
def plot_architecture():
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.set_axis_off()

    def box(x, y, w, h, text, color):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=1.4, edgecolor="black", facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=10, weight="bold")

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.5))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label,
                    ha="center", fontsize=9, color="dimgray")

    box(0.2, 1.7, 2.0, 1.6, "Seed FRP\nset $\\mathcal{S}$", "#F7D6C4")
    box(2.6, 1.7, 2.4, 1.6, "DS-HMamba\nGenerator", "#C4E1F7")
    box(5.4, 3.0, 2.4, 1.4, "L1/L2\nSeedFilter", "#F7EFC4")
    box(5.4, 0.8, 2.4, 1.4, "Ping-style\nAPD (16 probes)", "#C4F7D9")
    box(8.2, 1.7, 1.6, 1.6, "Online\nFine-Tune", "#E5C4F7")

    arrow(2.2, 2.5, 2.6, 2.5, "context")
    arrow(5.0, 2.7, 5.4, 3.5, "candidates")
    arrow(6.6, 3.0, 6.6, 2.2, "out-of-range")
    arrow(7.8, 1.5, 8.2, 2.0, "labels")
    arrow(8.9, 3.3, 3.8, 3.3, "$\\theta \\leftarrow \\theta - \\eta \\nabla$")

    ax.text(5.0, 4.7,
            "6FRP: DS-HMamba + In-range-aware Feedback (Placeholder)",
            ha="center", fontsize=12, weight="bold")

    fig.tight_layout()
    fig.savefig(HERE / "arch_dualscan.pdf", bbox_inches="tight")
    fig.savefig(HERE / "arch_dualscan.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved: arch_dualscan.pdf")


if __name__ == "__main__":
    d = _collect()
    plot_seed_overlap(d)
    plot_netnew_curve(d)
    # NOTE: architecture placeholder is disabled --- the definitive
    # arch_dualscan.pdf is now produced by plot_arch.py.
    # plot_architecture()
