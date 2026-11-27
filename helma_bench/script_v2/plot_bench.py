"""
Energy / throughput figures from bench_vision.py results (no GPU needed).

  python plot_bench.py ../results/v2_results/bench_<jobid>        (or a single CSV file)

Writes to ../results/v2_results/bench_<jobid>/plots/:
  throughput.png   images per second vs batch size
  energy.png       joules per image vs batch size
  power.png        average GPU power vs batch size, with the power limit
  sm_clock.png     SM clock vs batch size (drops when the power limit is reached)
  memory.png       peak GPU memory vs batch size; ✕ marks the batch size that ran out of memory
  summary.csv      per model: largest batch, energy at batch 1, lowest energy and its batch, peak throughput
Each point is the mean of the 3 repeats; the raw repeats are in bench.csv.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullLocator, ScalarFormatter  # noqa: E402

# one fixed name, colour and marker per model (same in every figure)
STYLE = {
    "resnet50.a1_in1k": ("ResNet-50", "#2a78d6", "o"),
    "convnext_tiny.in12k_ft_in1k": ("ConvNeXt-T", "#eb6834", "s"),
    "tf_efficientnetv2_s.in21k_ft_in1k": ("EfficientNetV2-S", "#1baf7a", "D"),
    "swin_tiny_patch4_window7_224.ms_in1k": ("Swin-T", "#eda100", "^"),
    "vit_small_patch16_224.augreg_in21k_ft_in1k": ("ViT-S/16", "#e87ba4", "v"),
    "vit_base_patch16_224.augreg_in1k": ("ViT-B/16 (IN-1k pretr.)", "#e34948", "P"),
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": ("ViT-B/16 (IN-21k pretr.)", "#6250d6", "X"),
}
# same network or same speed as another model -> dashed + hollow so both lines stay visible
HOLLOW = {"vit_base_patch16_224.augreg_in1k", "resnet50.a1_in1k"}
# Total memory of Helma's H100 as PyTorch counts it (total_memory / 1e9). NVIDIA sells it as "94 GB" (= 93 GiB).
GPU_MEMORY_GB = 99.96

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titlelocation": "left", "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})

path = sys.argv[1]
csv = os.path.join(path, "bench.csv") if os.path.isdir(path) else path
out = os.path.join(os.path.dirname(os.path.abspath(csv)), "plots")
os.makedirs(out, exist_ok=True)

raw = pd.read_csv(csv)
ok = raw[raw.status == "ok"]
cols = [c for c in ["img_per_s", "ms_per_batch", "avg_power_W", "J_per_img", "sm_clock_MHz", "peak_mem_GB"] if c in ok]
df = ok.groupby(["model", "batch_size"], as_index=False)[cols].mean()   # mean of the repeats
gpu, limit = raw.gpu.iloc[0], raw.power_limit_W.iloc[0]
precision = "FP32 (TF32 math)" if "tf32" in raw and raw.tf32.iloc[0] else "FP32"
batches = sorted(df.batch_size.unique())


def style(m):
    return STYLE.get(m, (m, "#7a7974", "o"))


def batch_plot(column, ylabel, title, fname, hline=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    end = df.loc[df.groupby("model").batch_size.idxmax()].set_index("model")[column]
    for m in sorted(df.model.unique(), key=lambda m: -end[m]):   # legend order = line order at the right end
        g = df[df.model == m].sort_values("batch_size")
        name, color, marker = style(m)
        kw = dict(linestyle=(0, (5, 3)), markerfacecolor="white", markeredgewidth=2) if m in HOLLOW else {}
        ax.plot(g.batch_size, g[column], color=color, marker=marker, markersize=6, linewidth=2, label=name,
                zorder=3 if m in HOLLOW else 2, **kw)
    if hline:
        ax.axhline(hline[0], color="#9a9a96", linestyle="--", linewidth=1.5, label=hline[1])
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(batches))
    ax.xaxis.set_major_formatter(ScalarFormatter())   # plain numbers, not 2^n
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}  ·  {gpu}, {precision}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.savefig(os.path.join(out, fname))
    plt.close(fig)


batch_plot("img_per_s", "Throughput (images / s)", "Throughput", "throughput.png")
batch_plot("J_per_img", "Energy per image (J)", "Energy per image", "energy.png")
batch_plot("avg_power_W", "Average GPU power (W)", "GPU power", "power.png", hline=(limit, f"Power limit ({limit:.0f} W)"))
batch_plot("sm_clock_MHz", "Mean SM clock (MHz)", "SM clock (lowered when the power limit is reached)", "sm_clock.png")


# ---- memory: where each model runs out of GPU memory
def memory_plot():
    """Peak memory per batch size. ✕ = the next batch size, which ran out of memory.
    Its height is an estimate: memory doubles when the batch doubles (true for every model here);
    it is drawn at least just above the limit, since the run did fail (cuDNN also needs scratch space)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    oom = raw[raw.status == "OOM"].groupby("model").batch_size.min()
    top = GPU_MEMORY_GB
    for m in sorted(df.model.unique(), key=lambda m: -oom.get(m, 1e9)):   # legend order = who fits most
        g = df[df.model == m].sort_values("batch_size")
        name, color, marker = style(m)
        kw = dict(linestyle=(0, (5, 3)), markerfacecolor="white", markeredgewidth=2) if m in HOLLOW else {}
        ax.plot(g.batch_size, g.peak_mem_GB, color=color, marker=marker, markersize=6, linewidth=2, label=name,
                zorder=3 if m in HOLLOW else 2, **kw)
        if m in oom:
            last = g.iloc[-1]
            est = max(2 * last.peak_mem_GB, GPU_MEMORY_GB * 1.04)   # it failed, so draw it above the limit
            ax.plot([last.batch_size, oom[m]], [last.peak_mem_GB, est], color=color, linestyle=":", linewidth=1.5)
            ax.plot(oom[m], est, marker="x", color=color, markersize=10, markeredgewidth=2.5, linestyle="none")
            top = max(top, est)
    ax.axhspan(GPU_MEMORY_GB, top * 1.3, color="#f1f0ec", zorder=0)
    ax.axhline(GPU_MEMORY_GB, color="#5f5e5a", linestyle="--", linewidth=1.5)
    ax.text(1, GPU_MEMORY_GB * 1.02, f"GPU memory limit {GPU_MEMORY_GB:.0f} GB  ·  above = does not fit",
            fontsize=9, color="#5f5e5a", va="bottom")
    ax.plot([], [], marker="x", color="#5f5e5a", linestyle=":", markersize=9, markeredgewidth=2.5,
            label="next batch size: out of memory\n(height = estimate, ≈ 2× the last one)")
    xs = sorted(set(df.batch_size) | set(oom))
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_yscale("log")   # memory doubles with batch: log axis shows every point, not just the last few
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(bottom=0.05, top=top * 1.3)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak GPU memory (GB, log scale)")
    ax.set_title(f"GPU memory vs batch size  ·  {gpu}, {precision}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    fig.savefig(os.path.join(out, "memory.png"))
    plt.close(fig)


if "peak_mem_GB" in df:
    memory_plot()

# ---- summary table
rows = []
for m in [m for m in STYLE if m in set(df.model)]:
    g = df[df.model == m]
    fail = raw[(raw.model == m) & (raw.status != "ok")]
    best = g.loc[g.J_per_img.idxmin()]
    rows.append({"model": style(m)[0], "largest_batch": int(g.batch_size.max()),
                 "next_batch_fails_with": fail.status.iloc[0] if len(fail) else "not reached",
                 "J_per_img_bs1": round(g[g.batch_size == 1].J_per_img.iloc[0], 4) if (g.batch_size == 1).any() else None,
                 "lowest_J_per_img": round(best.J_per_img, 4), "at_batch": int(best.batch_size),
                 "max_img_per_s": round(g.img_per_s.max(), 1)})
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(out, "summary.csv"), index=False)
print(summary.to_string(index=False))
print(f"\nsaved throughput.png, energy.png, power.png, sm_clock.png, memory.png, summary.csv in {out}")
