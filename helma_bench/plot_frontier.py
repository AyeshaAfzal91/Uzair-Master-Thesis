"""
Plots from a bench_vision.py results CSV (repeats are averaged):

  frontier.png        energy per image vs ImageNet top-1, each model at its most
                      energy-efficient batch size, with the Pareto frontier
  throughput.png      images per second vs batch size
  energy.png          joules per image vs batch size
  power.png           average GPU power vs batch size (with the power limit)
  work_vs_energy.png  GFLOPs per image vs J per image (only for CSVs made by add_flops.py)

Usage:  python plot_frontier.py results/vision_fp32_<jobid>.csv
        python plot_frontier.py results/vision_fp32_<jobid>_flops.csv   (adds work_vs_energy.png)
Images are saved next to the CSV.

Needs timm's accuracy table once (login node):
  wget -P timm_results https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/results-imagenet.csv
"""
import math
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

csv = sys.argv[1]
out_dir = os.path.dirname(os.path.abspath(csv))
here = os.path.dirname(os.path.abspath(__file__))

# Short display names, one fixed colour + marker per model (same in every plot)
STYLE = {
    "resnet50.a1_in1k":                           ("ResNet-50",              "#2a78d6", "o"),
    "convnext_tiny.in12k_ft_in1k":                ("ConvNeXt-T",             "#eb6834", "s"),
    "tf_efficientnetv2_s.in21k_ft_in1k":          ("EfficientNetV2-S",       "#1baf7a", "D"),
    "swin_tiny_patch4_window7_224.ms_in1k":       ("Swin-T",                 "#eda100", "^"),
    "vit_small_patch16_224.augreg_in21k_ft_in1k": ("ViT-S/16",               "#e87ba4", "v"),
    "vit_base_patch16_224.augreg_in1k":           ("ViT-B/16 (IN-1k pretr.)",  "#e34948", "P"),
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": ("ViT-B/16 (IN-21k pretr.)", "#6250d6", "X"),
}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})

# ---- load and average the repeats
raw = pd.read_csv(csv)
df = raw[raw.status == "ok"]
df = df.groupby(["model", "img_size", "batch_size"], as_index=False)[
    [c for c in ["img_per_s", "ms_per_batch", "avg_power_W", "J_per_img", "achieved_TFLOPs"] if c in raw]].mean()
power_limit = raw.power_limit_W.iloc[0]
gpu = raw.gpu.iloc[0]
precision = "FP32 (TF32 math)" if raw.get("tf32", pd.Series([False])).iloc[0] else "FP32"
batch_sizes = sorted(df.batch_size.unique())


def name(m):
    return STYLE.get(m, (m, "#5f5e59", "o"))[0]


def color(m):
    return STYLE.get(m, (m, "#5f5e59", "o"))[1]


def marker(m):
    return STYLE.get(m, (m, "#5f5e59", "o"))[2]


# Models that have identical speed/energy to another model are drawn hollow + dashed,
# so both stay visible where the lines lie on top of each other.
HOLLOW = {"vit_base_patch16_224.augreg_in1k", "resnet50.a1_in1k"}


def line_style(m):
    if m in HOLLOW:
        return dict(linestyle=(0, (5, 3)), markerfacecolor="white", markeredgewidth=2)
    return dict(linestyle="-")


def legend_outside(ax, handles=None, labels=None):
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)


# ---- 1. frontier ------------------------------------------------------------
# timm's table has one row per (model, input size): match on both
acc = pd.read_csv(os.path.join(here, "timm_results", "results-imagenet.csv"))[["model", "img_size", "top1"]]
best = df.loc[df.groupby("model").J_per_img.idxmin()].merge(acc, on=["model", "img_size"])
best = best.sort_values("J_per_img")
# a model is on the frontier if no cheaper model is at least as accurate
best["on_frontier"] = best.top1 > best.top1.shift().cummax().fillna(-1)

print(best[["model", "img_size", "batch_size", "img_per_s", "J_per_img", "top1", "on_frontier"]]
      .to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
front = best[best.on_frontier]
ax.plot(front.J_per_img, front.top1, color="#9a9a96", linestyle="--", linewidth=1.5,
        zorder=1, label="Pareto frontier")
for _, r in best.iterrows():
    tag = "  (on frontier)" if r.on_frontier else ""
    hollow = r.model in HOLLOW
    ax.scatter(r.J_per_img, r.top1, s=90, marker=marker(r.model), zorder=3,
               facecolor="white" if hollow else color(r.model), edgecolor=color(r.model),
               linewidth=2 if hollow else 1, label=f"{name(r.model)}, bs {r.batch_size}{tag}")
ax.set_xlim(0, best.J_per_img.max() * 1.15)
ax.set_ylim(best.top1.min() - 1, best.top1.max() + 1)
ax.set_xlabel("Energy per image (J)   ← lower is better")
ax.set_ylabel("ImageNet top-1 accuracy (%)")
ax.set_title(f"Energy vs accuracy  ·  {gpu}, {precision}", loc="left")
legend_outside(ax)
fig.savefig(os.path.join(out_dir, "frontier.png"))
plt.close(fig)


# ---- 2-4. batch-size plots (one image each) ----------------------------------
def batch_plot(column, ylabel, title, filename, limit=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, g in df.groupby("model"):
        g = g.sort_values("batch_size")
        ax.plot(g.batch_size, g[column], color=color(m), marker=marker(m), markersize=7,
                linewidth=2, label=name(m), zorder=3 if m in HOLLOW else 2, **line_style(m))
    if limit is not None:
        ax.axhline(limit, color="#9a9a96", linestyle="--", linewidth=1.5, label=f"Power limit ({limit:.0f} W)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(batch_sizes))  # ticks exactly at the measured batch sizes
    ax.xaxis.set_major_formatter(ScalarFormatter())          # plain numbers, not 2^n
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  ·  {gpu}, {precision}", loc="left")
    # legend in the same top-to-bottom order as the lines at the largest batch size
    last = df[df.batch_size == df.groupby("model").batch_size.transform("max")].set_index("model")[column]
    handles, labels = ax.get_legend_handles_labels()
    rank = {name(m): -v for m, v in last.items()}
    order = sorted(range(len(labels)), key=lambda i: rank.get(labels[i], float("inf")))  # power-limit line last
    legend_outside(ax, [handles[i] for i in order], [labels[i] for i in order])
    fig.savefig(os.path.join(out_dir, filename))
    plt.close(fig)


batch_plot("img_per_s", "Throughput (images / s)", "Throughput", "throughput.png")
batch_plot("J_per_img", "Energy per image (J)", "Energy per image", "energy.png")
batch_plot("avg_power_W", "Average GPU power (W)", "GPU power", "power.png", limit=power_limit)

saved = "frontier.png, throughput.png, energy.png, power.png"


# ---- 5. work vs energy (only for CSVs made by add_flops.py) --------------------
# x = work per image (GFLOPs), y = energy per image (J), each model at its best batch size.
# If energy only depended on work, all models would lie on one straight line.
# Grey lines mark constant efficiency (GFLOP per joule): further down-right = more efficient.
def work_vs_energy_plot():
    b = best.copy()
    b["gflops_per_img"] = b.achieved_TFLOPs * 1000 / b.img_per_s
    b["gflop_per_J"] = b.gflops_per_img / b.J_per_img
    print("\nEfficiency at best batch size (GFLOP per joule):")
    print(b[["model", "gflops_per_img", "J_per_img", "gflop_per_J"]].round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    xmax, ymax = b.gflops_per_img.max() * 1.15, b.J_per_img.max() * 1.15
    for eff in (50, 100, 200, 300):          # GFLOP per joule
        x_end = min(xmax, ymax * eff)        # where the line leaves the plot
        ax.plot([0, x_end], [0, x_end / eff], color="#c9c8c2", linewidth=1, zorder=1)
        # label sits on the middle of its line, tilted along it, on a small white patch
        x_lab = 0.55 * x_end
        ax.text(x_lab, x_lab / eff, f"{eff} GFLOP/J", color="#7a7974", fontsize=9,
                ha="center", va="center", rotation=math.degrees(math.atan2(1 / eff, 1)),
                transform_rotates_text=True, rotation_mode="anchor",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5), zorder=2)
    for _, r in b.iterrows():
        hollow = r.model in HOLLOW
        ax.scatter(r.gflops_per_img, r.J_per_img, s=90, marker=marker(r.model), zorder=3,
                   facecolor="white" if hollow else color(r.model), edgecolor=color(r.model),
                   linewidth=2 if hollow else 1,
                   label=f"{name(r.model)}  ({r.gflop_per_J:.0f} GFLOP/J)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.grid(False)  # the grey efficiency lines are the guides here
    ax.set_xlabel("Work per image (GFLOPs)")
    ax.set_ylabel("Energy per image (J)")
    ax.set_title(f"Work vs energy, best batch size  ·  {gpu}, {precision}", loc="left")
    legend_outside(ax)
    fig.savefig(os.path.join(out_dir, "work_vs_energy.png"))
    plt.close(fig)


if "achieved_TFLOPs" in df:
    work_vs_energy_plot()
    saved += ", work_vs_energy.png"
print(f"saved {saved} in {out_dir}")
