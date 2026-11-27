"""
Tables and figures from profile_vision.py results (CPU only, fine on the login node).

  python plot_profile.py ../results/v2_results/profile_<jobid>            (compares batch 1 and 256)
  python plot_profile.py ../results/v2_results/profile_<jobid> --batches 1 64

Writes to ../results/v2_results/profile_<jobid>/plots/

TABLES
  batch1_kernels.csv       every kernel of one forward pass at batch 1: name, type, layer, PyTorch op, time
  bottlenecks_bs<B>.csv    per layer role at batch B: time share, TFLOP/s, GB/s, FLOP/byte, bound
  max_batch.csv            largest batch that fits per model, its memory, and why the next one failed

FIGURES
  kernel_types_bs<B>.png   share of GPU time per kernel type (matmul, convolution, norm, ...)
  layers_bs<B>.png         share of GPU time per layer group (attention, MLP, convolution, ...)
  roofline_bs<B>.png       achieved TFLOP/s vs FLOPs per byte for each layer group -> compute or memory bound
  time_per_image.png       GPU time per image vs batch size, all models
  scaling_by_layer.png     time per image of each layer group vs batch size, one panel per model
  launch_overhead.png      share of time lost to CPU-side kernel launching vs batch size
  memory.png               peak GPU memory vs batch size, up to the largest batch that fits
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter  # noqa: E402

MODELS = {  # fixed order and display names
    "resnet50.a1_in1k": "ResNet-50",
    "convnext_tiny.in12k_ft_in1k": "ConvNeXt-T",
    "tf_efficientnetv2_s.in21k_ft_in1k": "EfficientNetV2-S",
    "swin_tiny_patch4_window7_224.ms_in1k": "Swin-T",
    "vit_small_patch16_224.augreg_in21k_ft_in1k": "ViT-S/16",
    "vit_base_patch16_224.augreg_in1k": "ViT-B/16 (IN-1k)",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k": "ViT-B/16 (IN-21k)",
}
GROUPS = {  # layer roles -> 8 groups for the figures
    "Attention: Q/K/V + output projection": ["attention.qkv", "attention.proj"],
    "Attention: core (QKᵀ, softmax, ×V)": ["attention.core"],
    "MLP (expand, GELU, contract)": ["mlp.fc1", "mlp.act", "mlp.fc2"],
    "Convolution (regular, patch embedding)": ["conv_kxk", "conv_1x1", "patch_embed"],
    "Convolution (depthwise)": ["conv_depthwise"],
    "Squeeze-excite": ["squeeze_excite"],
    "Norm": ["norm"],
    "Other (activations, adds, reshapes, pool, head)": ["activation", "other", "pool", "head", "linear"],
}
KINDS = ["matmul (GEMM)", "convolution", "attention (fused)", "norm", "softmax", "reduction / pool",
         "elementwise", "copy / layout", "other"]
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#8a8984"]
MODEL_COLOR = dict(zip(MODELS, PALETTE[:5] + ["#e34948", "#6250d6"]))
GROUP_COLOR = dict(zip(GROUPS, PALETTE))
KIND_COLOR = dict(zip(KINDS, PALETTE))
GROUP_MARKER = dict(zip(GROUPS, ["o", "s", "^", "D", "v", "P", "X", "*"]))
ROLE_TO_GROUP = {r: g for g, rs in GROUPS.items() for r in rs}
GREY = "#7a7974"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titlelocation": "left", "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def name(m):
    return MODELS.get(m, m)


def in_order(models):
    return [m for m in MODELS if m in set(models)]


def batch_axis(ax, batches):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(batches))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_xlabel("Batch size")


def legend_right(ax, handles=None, **kw):
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, **kw)


def save(fig, out, fname):
    fig.savefig(os.path.join(out, fname))
    plt.close(fig)


# =============================================================================
def stacked_shares(df, key, colors, title, out, fname):
    """Horizontal 100 % bars: one per model, segments = share of GPU time per category."""
    models = in_order(df.model.unique())
    share = df.pivot_table(index="model", columns=key, values="gpu_us", aggfunc="sum").reindex(models).fillna(0)
    share = share.div(share.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(models) + 1.6))
    left = np.zeros(len(models))
    for cat in colors:
        if cat not in share or share[cat].sum() == 0:
            continue
        v = share[cat].values
        ax.barh(range(len(models)), v, left=left, height=0.7, color=colors[cat], edgecolor="white",
                linewidth=1.5, label=cat)
        for i in range(len(models)):
            if v[i] >= 7:
                ax.text(left[i] + v[i] / 2, i, f"{v[i]:.0f}%", ha="center", va="center", fontsize=8, color="white")
        left += v
    ax.set_yticks(range(len(models)), [name(m) for m in models])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of GPU time (%)")
    ax.set_title(title)
    legend_right(ax, fontsize=9)
    save(fig, out, fname)


def roofline(roles, gpu, bs, out):
    g = roles[(roles.batch_size == bs) & (roles.gflops > 0) & (roles.mb > 0)]
    g = g.groupby(["model", "group"], as_index=False)[["gpu_us", "gflops", "mb"]].sum()
    if g.empty:
        return
    g["ai"] = g.gflops * 1e3 / g.mb                        # FLOPs per byte
    g["tflops"] = g.gflops / (g.gpu_us * 1e-6) / 1e3
    pc, pb = gpu["peak_tf32_TFLOPs"], gpu["peak_dram_GBps"]
    ridge = pc * 1e3 / pb
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.logspace(-1, 4, 300)
    ax.plot(x, np.minimum(pc, x * pb / 1e3), color=GREY, linewidth=1.8, zorder=1)
    ax.axvline(ridge, color="#c9c8c2", linestyle="--", linewidth=1, zorder=0)
    ax.text(ridge * 0.85, pc * 1.15, "memory-bound", ha="right", fontsize=9, color=GREY)
    ax.text(ridge * 1.15, pc * 1.15, "compute-bound", ha="left", fontsize=9, color=GREY)
    ax.text(9e3, pc * 0.8, f"peak TF32 {pc:.0f} TFLOP/s", ha="right", va="top", fontsize=9, color=GREY)
    for r in g.itertuples():
        ax.scatter(r.ai, r.tflops, s=80, marker=GROUP_MARKER[r.group], color=MODEL_COLOR.get(r.model, GREY),
                   edgecolor="white", linewidth=1, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(g.tflops.min() / 3, pc * 1.6)
    # label along the sloped memory roof (angle measured on screen, after the limits are fixed)
    x0, x1 = 0.3, 3.0
    p0, p1 = ax.transData.transform([(x0, x0 * pb / 1e3), (x1, x1 * pb / 1e3)])
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.annotate(f"memory roof {pb / 1e3:.2f} TB/s", (x0, x0 * pb / 1e3), xytext=(0, 6), textcoords="offset points",
                rotation=angle, rotation_mode="anchor", fontsize=9, color=GREY)
    ax.set_xlabel("FLOPs per byte moved (arithmetic intensity)")
    ax.set_ylabel("Achieved compute (TFLOP/s)")
    ax.set_title(f"Roofline per layer group  ·  batch {bs}")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=MODEL_COLOR[m], label=name(m))
               for m in in_order(g.model.unique())]
    handles += [plt.Line2D([], [], ls="", label="")]
    handles += [plt.Line2D([], [], marker=GROUP_MARKER[grp], ls="", ms=8, color=GREY, label=grp)
                for grp in GROUPS if grp in set(g.group)]
    legend_right(ax, handles=handles, fontsize=9, title="colour = model, shape = layer group", title_fontsize=9)
    save(fig, out, f"roofline_bs{bs}.png")


def lines_vs_batch(summary, col, ylabel, title, out, fname, logy=False, hline=None):
    s = summary[summary.status == "ok"]
    fig, ax = plt.subplots(figsize=(9, 5))
    end = s.loc[s.groupby("model").batch_size.idxmax()].set_index("model")[col]
    for m in sorted(in_order(s.model.unique()), key=lambda m: -end.get(m, 0)):   # legend order = line order
        d = s[s.model == m].sort_values("batch_size")
        ax.plot(d.batch_size, d[col], color=MODEL_COLOR[m], marker="o", ms=5, lw=2, label=name(m))
    if hline:
        ax.axhline(hline[0], color=GREY, ls="--", lw=1.2, label=hline[1])
    batch_axis(ax, sorted(s.batch_size.unique()))
    if logy:
        ax.set_yscale("log")
    else:
        ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    legend_right(ax, fontsize=9)
    save(fig, out, fname)


def scaling_by_layer(roles, out):
    """One panel per model: time per image of each layer group vs batch size."""
    g = roles.groupby(["model", "batch_size", "group"], as_index=False).gpu_us.sum()
    g["us_per_img"] = g.gpu_us / g.batch_size
    models = in_order(g.model.unique())
    batches = sorted(g.batch_size.unique())
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharey=True)
    for ax, m in zip(axes.flat, models):
        for grp in GROUPS:
            d = g[(g.model == m) & (g.group == grp)].sort_values("batch_size")
            if d.empty or d.gpu_us.sum() == 0:
                continue
            ax.plot(d.batch_size, d.us_per_img, color=GROUP_COLOR[grp], marker=GROUP_MARKER[grp], ms=4, lw=1.8)
        ax.set_title(name(m), fontsize=12)
        ax.set_yscale("log")
        batch_axis(ax, batches)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("Batch size", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("GPU time per image (µs)")
    legend_ax = axes.flat[len(models)] if len(models) < 8 else None
    for ax in axes.flat[len(models):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=GROUP_COLOR[grp], marker=GROUP_MARKER[grp], lw=1.8, label=grp)
               for grp in GROUPS if grp in set(g.group)]
    (legend_ax or axes.flat[-1]).legend(handles=handles, loc="center", fontsize=10,
                                        title="Layer group  (falling line = gets cheaper per image with batching)",
                                        title_fontsize=9)
    fig.suptitle("Time per image of each layer group vs batch size", x=0.01, ha="left", fontsize=14)
    fig.tight_layout()
    save(fig, out, "scaling_by_layer.png")


# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("results")
    p.add_argument("--batches", type=int, nargs="+", default=[1, 256], help="batch sizes to compare in detail")
    a = p.parse_args()
    out = os.path.join(a.results, "plots")
    os.makedirs(out, exist_ok=True)
    summary = pd.read_csv(os.path.join(a.results, "summary.csv"))
    roles = pd.read_csv(os.path.join(a.results, "roles.csv"))
    kernels = pd.read_csv(os.path.join(a.results, "kernels.csv"))
    gpu = json.load(open(os.path.join(a.results, "gpu.json")))
    roles["group"] = roles.role.map(ROLE_TO_GROUP).fillna(list(GROUPS)[-1])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    # ---- tables
    b1 = kernels[kernels.batch_size == 1].sort_values(["model", "gpu_us"], ascending=[True, False])
    b1.to_csv(os.path.join(out, "batch1_kernels.csv"), index=False)
    print("===== Batch 1: the kernels that take the most time =====")
    for m in in_order(b1.model.unique()):
        print(f"\n{name(m)}")
        print(b1[b1.model == m].head(6)[["kernel", "kind", "role", "calls", "gpu_us", "share_pct"]]
              .round(1).to_string(index=False))

    ok = summary[summary.status == "ok"]
    rows = []
    for m in in_order(summary.model.unique()):
        d, fail = ok[ok.model == m], summary[(summary.model == m) & (summary.status != "ok")]
        last = d.loc[d.batch_size.idxmax()]
        rows.append({"model": name(m), "largest_batch": int(last.batch_size), "peak_mem_GB": round(last.peak_mem_GB, 1),
                     "peak_MB_per_image": round(last.peak_mem_GB * 1e3 / last.batch_size, 2),
                     "next_batch": int(fail.batch_size.iloc[0]) if len(fail) else None,
                     "why_next_fails": fail.status.iloc[0] if len(fail) else "not reached"})
    mb = pd.DataFrame(rows)
    mb.to_csv(os.path.join(out, "max_batch.csv"), index=False)
    print(f"\n===== Largest batch that fits (GPU memory {gpu['memory_GB']:.0f} GB) =====")
    print(mb.to_string(index=False))

    for bs in a.batches:
        r = roles[roles.batch_size == bs]
        if r.empty:
            print(f"\n(no results at batch {bs})")
            continue
        cols = ["model", "role", "share_pct", "gpu_us", "gflops", "mb", "tflops", "gbps", "flop_per_byte",
                "roofline_efficiency_pct", "bound"]
        r[cols].to_csv(os.path.join(out, f"bottlenecks_bs{bs}.csv"), index=False)
        print(f"\n===== Batch {bs}: layer roles taking >= 5 % of the time =====")
        print(r[r.share_pct >= 5][cols].round(1).to_string(index=False))

        # ---- figures at this batch size
        stacked_shares(kernels[kernels.batch_size == bs], "kind", KIND_COLOR,
                       f"GPU time by kernel type  ·  batch {bs}", out, f"kernel_types_bs{bs}.png")
        stacked_shares(r, "group", GROUP_COLOR, f"GPU time by layer  ·  batch {bs}", out, f"layers_bs{bs}.png")
        roofline(roles, gpu, bs, out)

    # ---- figures over all batch sizes
    lines_vs_batch(summary, "us_per_img", "GPU time per image (µs)",
                   "Time per image vs batch size (CUDA graph)", out, "time_per_image.png", logy=True)
    scaling_by_layer(roles, out)
    lines_vs_batch(summary, "launch_overhead_pct", "Time lost to CPU kernel launching (%)",
                   "Launch overhead: 1 − CUDA-graph time / normal PyTorch time", out, "launch_overhead.png")
    lines_vs_batch(summary, "peak_mem_GB", "Peak GPU memory (GB)", "GPU memory vs batch size", out, "memory.png",
                   hline=(gpu["memory_GB"], f"GPU memory {gpu['memory_GB']:.0f} GB"))
    print(f"\ntables and figures in {out}")


if __name__ == "__main__":
    main()
