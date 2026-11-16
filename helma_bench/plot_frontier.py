"""
Plots from a bench_vision.py results CSV:
  1. frontier.png    - energy per image vs ImageNet top-1 (timm's published FP32 accuracy),
                       each model at its most energy-efficient batch size
  2. batch_size.png  - throughput, energy per image and power vs batch size

Usage:  python plot_frontier.py results/vision_fp32_<jobid>.csv

Needs timm's accuracy table once (login node):
  wget -P timm_results https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/results/results-imagenet.csv
"""
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

csv = sys.argv[1]
out_dir = os.path.dirname(os.path.abspath(csv))
here = os.path.dirname(os.path.abspath(__file__))

# average the repeats
df = pd.read_csv(csv)
df = df[df.status == "ok"]
df = df.groupby(["model", "img_size", "batch_size"], as_index=False)[
    ["img_per_s", "ms_per_batch", "avg_power_W", "J_per_img", "sm_clock_MHz", "power_capped_frac"]].mean()

# timm's published accuracy. The table has one row per (model, input size),
# so match on both: e.g. resnet50.a1_in1k is 80.4 % at 224 px but 81.2 % at 288 px.
acc = pd.read_csv(os.path.join(here, "timm_results", "results-imagenet.csv"))[["model", "img_size", "top1"]]

# ---- 1. frontier: each model at the batch size with the lowest J/img
best = df.loc[df.groupby("model").J_per_img.idxmin()].merge(acc, on=["model", "img_size"])
best = best.sort_values("J_per_img")
best["on_frontier"] = best.top1 >= best.top1.cummax()  # nothing cheaper is more accurate
print(best[["model", "batch_size", "img_per_s", "J_per_img", "avg_power_W", "top1", "on_frontier"]]
      .to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
front = best[best.on_frontier]
ax.plot(front.J_per_img * 1000, front.top1, color="grey", zorder=1, label="Pareto frontier")
ax.scatter(best.J_per_img * 1000, best.top1, zorder=2)
for _, r in best.iterrows():
    ax.annotate(f"{r.model.split('.')[0]}\n({r.model.split('.')[1]}, bs {r.batch_size})",
                (r.J_per_img * 1000, r.top1), xytext=(5, 3), textcoords="offset points", fontsize=7)
ax.set_xlabel("Energy per image [mJ]  (lower is better)")
ax.set_ylabel("ImageNet top-1 [%]")
ax.set_title("Energy-to-accuracy, FP32, H100, best batch size per model")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "frontier.png"), dpi=200)

# ---- 2. batch-size behaviour
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for name, g in df.groupby("model"):
    label = name.replace("_patch16_224", "").replace("_patch4_window7_224", "")
    axes[0].plot(g.batch_size, g.img_per_s, marker="o", label=label)
    axes[1].plot(g.batch_size, g.J_per_img * 1000, marker="o")
    axes[2].plot(g.batch_size, g.avg_power_W, marker="o")
titles = ["Throughput [img/s]", "Energy per image [mJ]", "Average power [W]"]
for ax, t in zip(axes, titles):
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_title(t)
    ax.grid(alpha=0.3)
axes[0].set_yscale("log")
axes[1].set_yscale("log")
limit = pd.read_csv(csv).power_limit_W.iloc[0]
axes[2].axhline(limit, color="grey", linestyle="--", label=f"power limit {limit:.0f} W")
axes[2].legend(fontsize=8)
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "batch_size.png"), dpi=200)
print(f"saved frontier.png and batch_size.png in {out_dir}")
