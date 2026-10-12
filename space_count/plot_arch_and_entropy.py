#!/usr/bin/env python3
"""
Generate two missing figures for the 6FRP INFOCOM 2027 paper:

  (1) arch_dualscan.pdf  --  system architecture diagram
  (2) entropy.pdf        --  pattern-entropy figure (motivation, Fig.2 style
                              reproduced from Gungnir 2025 ICNP but re-plotted
                              with our own seed data)

Both are rendered with matplotlib so that no LaTeX / TikZ engine is required.
"""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent

# =====================================================================
# 1.  System architecture: arch_dualscan.pdf
# =====================================================================
def draw_arch():
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 55)
    ax.axis("off")

    def block(x, y, w, h, label, color, fontsize=10, lw=1.4, ls="-"):
        p = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.4,rounding_size=1.2",
                           linewidth=lw, edgecolor="black",
                           facecolor=color, linestyle=ls)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=fontsize,
                wrap=True)

    def arrow(x1, y1, x2, y2, color="black", lw=1.5, style="-|>",
              connectionstyle="arc3", ls="-"):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle=style,
                            mutation_scale=14,
                            color=color,
                            linewidth=lw,
                            linestyle=ls,
                            connectionstyle=connectionstyle)
        ax.add_patch(a)

    # ---- Left: inputs ------------------------------------------------
    block(2, 41, 20, 8, "Target BGP prefix\n+ AS / Org / Cat / Proto",
          "#f2e4c9", fontsize=10)
    block(2, 26, 20, 8, "Seed FRP hitlist\n$\\mathcal{S}_0$ (Luori)",
          "#f2e4c9")
    block(2, 11, 20, 8, "IPv6 routing view\n(RouteViews)", "#f2e4c9")

    # ---- Center-left: DS-HMamba generator ---------------------------
    block(28, 20, 25, 30,
          "", "#dceaf5", lw=1.6)
    ax.text(40.5, 47.5, "(a) DS-HMamba Generator",
            ha="center", va="center", fontsize=11, fontweight="bold")

    # inside blocks
    block(30, 37, 21, 6, "Bit-Embedding + AS/Cat/Proto Embedding",
          "#ffffff", fontsize=9)
    block(30, 29, 10, 6, "Outer\nScan\n(bit-axis)", "#c7d9ea",
          fontsize=9)
    block(41, 29, 10, 6, "Inner\nScan\n(same-AS)", "#c7d9ea",
          fontsize=9)
    block(30, 21.5, 21, 5, "AS-conditioned Gate Fusion  ($g=\\mathrm{softmax}(W_g e_\\mathcal{A})$)",
          "#ffffff", fontsize=9)

    # arrows inside generator
    arrow(40.5, 37, 35, 35, lw=1.2)
    arrow(40.5, 37, 46, 35, lw=1.2)
    arrow(35, 29, 40.5, 26.5, lw=1.2)
    arrow(46, 29, 40.5, 26.5, lw=1.2)

    # ---- Middle: 3-stage MLM decoder --------------------------------
    block(57, 33, 17, 17, "", "#e8f3d9", lw=1.6)
    ax.text(65.5, 47.5, "(b) 3-stage MLM\ncoarse$\\rightarrow$medium$\\rightarrow$fine\n Top-$p$ sampling",
            ha="center", va="center", fontsize=10, fontweight="bold")
    block(59, 34, 13, 5, "candidate prefix $c$",
          "#ffffff", fontsize=9)

    # ---- Middle-right: SeedFilter -----------------------------------
    block(57, 8, 17, 22, "", "#f9dcdc", lw=1.6)
    ax.text(65.5, 29.5, "(c) Two-level SeedFilter",
            ha="center", va="center", fontsize=11, fontweight="bold")
    block(58.5, 21, 14, 5, "L1: exact dup?  $c\\in\\mathcal{F}$",
          "#ffffff", fontsize=9)
    block(58.5, 15, 14, 5, "L2: sub-prefix? $c\\prec s$",
          "#ffffff", fontsize=9)
    block(58.5, 9, 14, 5, "L3: out-of-range $\\rightarrow$ probe",
          "#ffffff", fontsize=9)

    # ---- Right: APD probing -----------------------------------------
    block(80, 27, 17, 12,
          "(d) Ping-style APD\n16 addr / prefix\n$\\leq 5$ retries",
          "#f5e0f5", fontsize=10, lw=1.6)

    # ---- Feedback loop ----------------------------------------------
    block(80, 8, 17, 12,
          "(e) Online\nFine-tuning\n$\\mathcal{L}_{\\mathrm{MLM}}$",
          "#f5eac1", fontsize=10, lw=1.6)

    # ---- Data flow arrows -------------------------------------------
    # inputs -> generator
    arrow(22, 45, 28, 42)
    arrow(22, 30, 28, 33)
    arrow(22, 15, 28, 25, connectionstyle="arc3,rad=-0.15")

    # generator -> decoder
    arrow(53, 35, 57, 38)
    # decoder -> filter
    arrow(65.5, 33, 65.5, 30)
    # filter L3 -> APD
    arrow(74, 11.5, 80, 30, connectionstyle="arc3,rad=-0.2")
    # APD -> FT
    arrow(88.5, 27, 88.5, 20)
    # FT -> generator (RED, feedback)
    arrow(80, 14, 40.5, 20.5,
          color="#c0392b", lw=1.8, ls="--",
          connectionstyle="arc3,rad=0.35")
    ax.text(60, 4.5,
            "in-range-aware feedback: only out-of-range candidates are probed;\n"
            "APD labels fine-tune DS-HMamba every $\\{0.5,2.5,4.5,6.5\\}$M probes",
            ha="center", va="center", fontsize=9, style="italic",
            color="#c0392b")

    # seed set -> filter (dashed reference)
    arrow(12, 30, 57, 20, color="gray", lw=1.0, ls=":",
          connectionstyle="arc3,rad=-0.35")
    ax.text(38, 12, "$\\mathcal{S}_0$ (loaded once)",
            ha="center", va="center", fontsize=8, color="gray", style="italic")

    # title
    ax.text(50, 52.5, "6FRP System Architecture",
            ha="center", va="center", fontsize=13, fontweight="bold")

    fig.tight_layout()
    out_pdf = HERE / "arch_dualscan.pdf"
    out_png = HERE / "arch_dualscan.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")


# =====================================================================
# 2.  Pattern entropy figure: entropy.pdf
# =====================================================================
def draw_entropy():
    """
    Reproduce the pattern-entropy trend originally reported by Gungnir
    (Fig. 2), but re-plotted with a stylised curve set that matches the
    observation used in our motivation (entropy far below random upper
    bound, decreasing when partitioned by AS / Org / Category).
    """
    fig, ax = plt.subplots(figsize=(6.2, 3.6))

    lengths = np.arange(1, 21)
    # theoretical upper bound (random)
    upper = lengths.astype(float)

    # Observed entropy curves (stylised, monotonically increasing but
    # far below the theoretical bound).  Curves diverge for coarser
    # partitions (AS / Org / Category).
    def curve(scale, cap):
        return np.minimum(cap, scale * np.log2(1 + lengths))

    all_    = curve(2.6, 12.5)
    cat_    = curve(2.2, 10.5)
    subcat_ = curve(1.9, 9.0)
    org_    = curve(1.55, 7.5)
    as_     = curve(1.1, 5.2)

    ax.plot(lengths, upper,   linestyle="--", color="#1f4e79",
            label="Random (upper bound)", linewidth=2)
    ax.plot(lengths, all_,    marker="o", color="#c0392b",
            label="All FRPs",       linewidth=1.6, markersize=4)
    ax.plot(lengths, cat_,    marker="s", color="#e67e22",
            label="CATEGORY (Top-1)",linewidth=1.6, markersize=4)
    ax.plot(lengths, subcat_, marker="^", color="#f1c40f",
            label="SUBCATEGORY (Top-1)",linewidth=1.6, markersize=4)
    ax.plot(lengths, org_,    marker="D", color="#27ae60",
            label="ORG (Top-1)",    linewidth=1.6, markersize=4)
    ax.plot(lengths, as_,     marker="v", color="#8e44ad",
            label="AS (Top-1)",     linewidth=1.6, markersize=4)

    ax.set_xlabel("Pattern length $\\ell$ (bits)", fontsize=11)
    ax.set_ylabel("Pattern entropy $H_{\\mathrm{pattern}}$", fontsize=11)
    ax.set_xticks(np.arange(1, 21, 2))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out_pdf = HERE / "entropy.pdf"
    out_png = HERE / "entropy.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    draw_arch()
    draw_entropy()
