#!/usr/bin/env python3
"""
Minimal, publication-quality architecture diagram of Heimdall.

Design goals (v7, aligned with paper Sec. IV modules; clean feedback
routing so that arrow labels never overlap the Fine-tuner box):
  1. Single left-to-right pipeline on the main spine.  Data nodes
     (Seed Store, FRP set) use dashed borders; processing modules
     use solid borders.
  2. Every processing module in paper Sec. IV corresponds to a single
     visible box:
        - Module 1 (Sec. IV-C + IV-D): DS-HMamba Generator, drawn as
          ONE box with an internal split showing (i) the outer/inner
          dual scan (matching Sec. IV-C.1 and IV-C.2) and (ii) the
          MLM head, so the reader can see that the MLM decoder is
          the output head of DS-HMamba without needing a separate
          grouping frame.
        - Module 2 (Sec. IV-E): SeedFilter.
        - Module 3 (Sec. IV-F): Active Prober.
        - Module 4 (Sec. IV-G): Online Fine-tuner, drawn as an
          independent box sitting above the main spine.
  3. Feedback routing (red dashed):
        Prober top -> straight up -> right side of Fine-tuner;
        Fine-tuner left side -> straight left, then down -> Generator top.
     Labels are placed on the *vertical* segments so they never overlap
     the Fine-tuner box.
  4. Growth lane (green dashed): FRP set -> SeedFilter, bottom.
  5. Compact vertical layout (H = 3.8) for the two-column IEEE style.

Saves arch_dualscan.pdf and arch_dualscan.png next to this file.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent

# =============================================================================
# Canvas width is chosen so that the right-side padding matches the left:
#   left  padding = IN_X                     = 0.20
#   right edge of content = OUT_X + OUT_W    = 10.35 + 1.80 = 12.15
#   -> FIG_W = 12.15 + 0.20 = 12.35 (symmetric 0.20 gutter on both sides)
FIG_W, FIG_H = 12.35, 3.2
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_axis_off()


# ---------- primitives -------------------------------------------------------
def box(x, y, w, h, title, subtitle="",
        fc="#e8f0fe", ec="#345aa7", lw=1.7, ls="-",
        title_size=14, sub_size=11.5, radius=0.14):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        fc=fc, ec=ec, lw=lw, ls=ls))
    if subtitle:
        ax.text(x + w / 2, y + h / 2 + 0.17, title,
                ha="center", va="center",
                fontsize=title_size, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.24, subtitle,
                ha="center", va="center",
                fontsize=sub_size, style="italic", color="#555555")
    else:
        ax.text(x + w / 2, y + h / 2, title,
                ha="center", va="center",
                fontsize=title_size, fontweight="bold")


def h_arrow(x0, y, x1, color="black", lw=1.9, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle="-|>", mutation_scale=18,
        color=color, lw=lw, linestyle=ls))


def u_arrow_bottom(x0, y0, x1, y1, bot_y, color="#38761d", lw=2.0, ls="--"):
    """(x0,y0) down -> horizontal at bot_y -> up -> (x1,y1)."""
    ax.plot([x0, x0], [y0, bot_y], color=color, lw=lw, ls=ls,
            solid_capstyle="round")
    ax.plot([x0, x1], [bot_y, bot_y], color=color, lw=lw, ls=ls,
            solid_capstyle="round")
    ax.add_patch(FancyArrowPatch(
        (x1, bot_y), (x1, y1),
        arrowstyle="-|>", mutation_scale=17,
        color=color, lw=lw, linestyle=ls))


# =============================================================================
# LAYOUT
# =============================================================================
MID = 1.25            # y-center of main-spine boxes
BOX_H = 1.15
BOT_LANE = 0.35       # growth horizontal channel (bottom)

# Fine-tuner box vertical position: sits just above the main spine.
# Pulled DOWN (was 3.10) so the whole diagram is more compact.
FT_H = 0.75
FT_MID = 2.55                       # y-center of Fine-tuner
FT_Y = FT_MID - FT_H / 2            # bottom of Fine-tuner
FT_TOP = FT_Y + FT_H                # top of Fine-tuner

# Horizontal segments of the feedback lane sit at the vertical center of
# the Fine-tuner, so the arrows enter/exit its left and right sides
# cleanly, and labels sit on the vertical legs (never over the box).
FB_Y = FT_MID

# x-positions (left edge) and widths on the main spine
IN_X,  IN_W  = 0.20,  1.75    # Seed Store (data node)
GEN_X, GEN_W = 2.15,  3.55    # DS-HMamba Generator (with internal split)
FLT_X, FLT_W = 5.95,  1.95    # SeedFilter
APD_X, APD_W = 8.15,  1.95    # Active Prober
OUT_X, OUT_W = 10.35, 1.80    # FRP set (data node)

BOX_Y = MID - BOX_H / 2

# =============================================================================
# 1. Main-spine data / module boxes
# =============================================================================
# ---- Seed Store (data node, dashed border) ----
box(IN_X, BOX_Y, IN_W, BOX_H,
    "Seed Store",
    "prefixes / AS / $\\mathcal{S}_{0}$",
    fc="#fff2cc", ec="#a67b00", ls="--")

# ---- Module 1: DS-HMamba Generator (one box, internal split) --------------
ax.add_patch(FancyBboxPatch(
    (GEN_X, BOX_Y), GEN_W, BOX_H,
    boxstyle="round,pad=0.02,rounding_size=0.14",
    fc="#dbe5f9", ec="#345aa7", lw=2.0))
# title along the top of the container (no module numbering)
ax.text(GEN_X + GEN_W / 2, BOX_Y + BOX_H - 0.18,
        "DS-HMamba Generator",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color="#1c3a7a")

# internal vertical split between backbone and MLM head
# (shifted slightly to the LEFT so the left sub-part is narrower)
split_x = GEN_X + GEN_W * 0.55
ax.plot([split_x, split_x],
        [BOX_Y + 0.08, BOX_Y + BOX_H - 0.36],
        color="#345aa7", lw=1.0, ls=(0, (3, 2)))

# left sub-part: dual scan (outer + inner), matching Sec. IV-C
# (shifted slightly UPWARD relative to the previous layout)
ax.text((GEN_X + split_x) / 2, BOX_Y + 0.65,
        "Outer scan",
        ha="center", va="center",
        fontsize=11.5, fontweight="bold", color="#345aa7")
ax.text((GEN_X + split_x) / 2, BOX_Y + 0.41,
        "+ Inner scan",
        ha="center", va="center",
        fontsize=11.5, fontweight="bold", color="#345aa7")
ax.text((GEN_X + split_x) / 2, BOX_Y + 0.21,
        "(bit axis / same-AS axis)",
        ha="center", va="center",
        fontsize=9.0, style="italic", color="#555555")

# right sub-part: MLM head (= MLM Decoder, Sec. IV-D)
ax.text((split_x + GEN_X + GEN_W) / 2, BOX_Y + 0.50,
        "MLM head\n(decoder)",
        ha="center", va="center",
        fontsize=11.5, fontweight="bold", color="#3f803a")
ax.text((split_x + GEN_X + GEN_W) / 2, BOX_Y + 0.19,
        "coarse $\\to$ fine",
        ha="center", va="center",
        fontsize=9.5, style="italic", color="#555555")

# tiny arrow inside the box: backbone -> MLM head (dataflow within Module 1)
ax.add_patch(FancyArrowPatch(
    (split_x - 0.01, BOX_Y + 0.45), (split_x + 0.03, BOX_Y + 0.45),
    arrowstyle="-|>", mutation_scale=13, color="#345aa7", lw=1.2))

# ---- SeedFilter ----
box(FLT_X, BOX_Y, FLT_W, BOX_H,
    "SeedFilter",
    "dedup vs. $\\mathcal{S}_{0}$",
    fc="#f7dede", ec="#a83232")

# ---- Active Prober ----
box(APD_X, BOX_Y, APD_W, BOX_H,
    "Active Prober",
    "APD verification",
    fc="#eddef4", ec="#6a3b8f")

# ---- FRP set (data node, dashed border) ----
box(OUT_X, BOX_Y, OUT_W, BOX_H,
    "FRP set $\\mathcal{F}$",
    "discovered",
    fc="#d9ead3", ec="#38761d", ls="--")

# =============================================================================
# 2. Online Fine-tuner (independent box above the spine)
# =============================================================================
FT_W = 3.00
FT_X = (APD_X + APD_W / 2 + GEN_X + GEN_W * 0.75) / 2 - FT_W / 2
box(FT_X, FT_Y, FT_W, FT_H,
    "Online Fine-tuner",
    "in-range-aware, real-time",
    fc="#fde5d3", ec="#c14040", lw=2.0,
    title_size=12.5, sub_size=10.5)

# =============================================================================
# 3. Forward flow on the main spine
# =============================================================================
h_arrow(IN_X  + IN_W,  MID, GEN_X)
h_arrow(GEN_X + GEN_W, MID, FLT_X)
h_arrow(FLT_X + FLT_W, MID, APD_X)
h_arrow(APD_X + APD_W, MID, OUT_X)

# =============================================================================
# 4. Feedback lane (red dashed): Prober -> Fine-tuner -> Generator
#    Route:
#       (a) Prober top-center -> straight up -> right side of Fine-tuner
#       (b) Fine-tuner left side -> straight left -> down into Generator top
# =============================================================================
FB_COLOR = "#c14040"
FB_LW = 2.4

FT_L = FT_X                    # left edge of Fine-tuner box
FT_R = FT_X + FT_W             # right edge of Fine-tuner box
APD_TOP_MID = APD_X + APD_W / 2
GEN_TOP_MID = GEN_X + GEN_W * 0.72   # enter Generator above MLM-head side

# ---- (a) Prober -> Fine-tuner (right side) ----
# vertical leg upwards
ax.plot([APD_TOP_MID, APD_TOP_MID],
        [BOX_Y + BOX_H, FB_Y],
        color=FB_COLOR, lw=FB_LW, ls="--", solid_capstyle="round")
# horizontal leg leftwards into Fine-tuner right side
ax.add_patch(FancyArrowPatch(
    (APD_TOP_MID, FB_Y), (FT_R + 0.02, FB_Y),
    arrowstyle="-|>", mutation_scale=18,
    color=FB_COLOR, lw=FB_LW, linestyle="--"))

# ---- (b) Fine-tuner -> Generator (left side then down) ----
# horizontal leg leftwards from Fine-tuner left edge
ax.plot([FT_L - 0.02, GEN_TOP_MID],
        [FB_Y, FB_Y],
        color=FB_COLOR, lw=FB_LW, ls="--", solid_capstyle="round")
# vertical leg downwards into Generator top
ax.add_patch(FancyArrowPatch(
    (GEN_TOP_MID, FB_Y), (GEN_TOP_MID, BOX_Y + BOX_H),
    arrowstyle="-|>", mutation_scale=18,
    color=FB_COLOR, lw=FB_LW, linestyle="--"))

# =============================================================================
# 5. Growth lane: FRP set -> SeedFilter (green dashed)
# =============================================================================
u_arrow_bottom(OUT_X + OUT_W / 2, BOX_Y,
               FLT_X + FLT_W / 2, BOX_Y,
               bot_y=BOT_LANE, color="#38761d", lw=2.0, ls="--")
ax.text((OUT_X + OUT_W / 2 + FLT_X + FLT_W / 2) / 2, BOT_LANE - 0.10,
        "grow known FRP set $\\mathcal{F}$",
        fontsize=13, style="italic", color="#38761d",
        ha="center", va="top", fontweight="bold")

# =============================================================================
# 6. Legend
# =============================================================================
legend_handles = [
    Line2D([0], [0], color="black", lw=1.9, label="forward data flow"),
    Line2D([0], [0], color="#c14040", lw=2.4, ls="--",
           label="feedback (online fine-tune)"),
    Line2D([0], [0], color="#38761d", lw=2.0, ls="--",
           label="grow known FRP set"),
]
ax.legend(handles=legend_handles, loc="lower center",
          bbox_to_anchor=(0.5, -0.14), fontsize=12, framealpha=0.92,
          ncol=3)

plt.tight_layout()
fig.savefig(HERE / "arch_dualscan.pdf", bbox_inches="tight")
fig.savefig(HERE / "arch_dualscan.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("Saved arch_dualscan.pdf / arch_dualscan.png")
