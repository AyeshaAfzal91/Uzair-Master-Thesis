"""
Kernel-level profile of the 7 thesis models on one H100 (FP32 weights, TF32 math = timm setting).

For every model and batch size 1, 2, 4, 8, ... (doubling until the GPU runs out of memory):
  - time per batch and per image (CUDA graph replay), peak GPU memory
  - time lost to CPU-side kernel launching (normal PyTorch vs CUDA graph)
  - every GPU kernel that runs: name, type, the layer it belongs to, the PyTorch op that launched it
  - per layer role: GPU time, FLOPs, bytes moved -> compute-bound, memory-bound or latency-bound

Output: helma_bench/results/v2_results/profile_<jobid>/
  summary.csv   one row per (model, batch)
  roles.csv     one row per (model, batch, layer role)
  kernels.csv   one row per (model, batch, kernel)
  gpu.json      GPU name and peak compute / bandwidth used for the bound verdict
  traces/       full kernel timelines for batch 1 and 256 (open in https://ui.perfetto.dev)

Run:  python profile_vision.py                       (all models, each in its own process)
      python profile_vision.py --models resnet50.a1_in1k --max-batch 4     (quick check)
Plots: python plot_profile.py ../results/v2_results/profile_<jobid>
"""
import argparse
import bisect
import gc
import glob
import json
import os
import re
import subprocess
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

# =============================================================================
# 1. Settings
# =============================================================================
MODELS = [
    "resnet50.a1_in1k",
    "convnext_tiny.in12k_ft_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in1k",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
]
MAX_BATCH = 65536        # doubling stops at the first out-of-memory long before this
PROFILE_ITERS = 3        # forward passes recorded by the profiler per batch size
TIMING_S = 2.0           # seconds of timing per batch size (eager and CUDA graph each)
KEEP_TRACES = (1, 256)   # batch sizes whose full timeline is kept on disk
HERE = os.path.dirname(os.path.abspath(__file__))                         # helma_bench/script_v2
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results", "v2_results"))  # helma_bench/results/v2_results

# FP32 weights and inputs, TF32 tensor-core math: identical to timm's benchmark/validation
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def load_model(name, device="cuda", pretrained=True):
    """timm model + the input size timm used for its published accuracy."""
    import timm
    from timm.data import resolve_data_config
    model = timm.create_model(name, pretrained=pretrained).eval().to(device)
    img = resolve_data_config({}, model=model, use_test_size=True)["input_size"][1]
    return model, img


# =============================================================================
# 2. Layer roles: which part of the network each module is
# =============================================================================
ROLES = ["patch_embed", "attention.qkv", "attention.core", "attention.proj", "mlp.fc1", "mlp.act", "mlp.fc2",
         "conv_kxk", "conv_1x1", "conv_depthwise", "squeeze_excite", "linear", "norm", "activation",
         "pool", "head", "other"]
ACTIVATIONS = (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Hardswish, nn.Hardsigmoid, nn.Tanh)
NORMS = (nn.modules.batchnorm._NormBase, nn.LayerNorm, nn.GroupNorm)


def module_roles(model):
    """
    List of {fqn, role, children} for the modules we track. First matching rule wins:
      classifier -> head | *patch_embed -> patch_embed | *Attention -> attention (+ qkv, proj inside)
      Mlp -> mlp (+ fc1, fc2 inside) | SqueezeExcite | Conv2d (k x k, 1 x 1 or depthwise) | Linear
      Norm | activation | pooling | anything else: look inside.
    Work outside all of these (residual adds, reshapes, window shifts) is counted as 'other'.
    """
    classifier = model.get_classifier()
    roles = []

    def add(fqn, role, children=()):
        roles.append({"fqn": fqn, "role": role, "children": list(children)})

    def visit(mod, fqn):
        cls = type(mod).__name__
        if mod is classifier and not isinstance(mod, nn.Identity):
            return add(fqn, "head")
        if cls.endswith("Head"):                       # classifier head containers: look inside
            for n, c in mod.named_children():
                visit(c, f"{fqn}.{n}")
            return
        if fqn.split(".")[-1] == "patch_embed":
            return add(fqn, "patch_embed")
        if cls.endswith("Attention"):
            kids = [f"{fqn}.{n}" for n in ("qkv", "proj") if isinstance(getattr(mod, n, None), nn.Linear)]
            for k in kids:
                add(k, "attention." + k.split(".")[-1])
            return add(fqn, "attention.core", kids)      # everything in attention except qkv / proj
        if cls in ("Mlp", "GluMlp", "SwiGLU"):
            kids = [f"{fqn}.{n}" for n in ("fc1", "fc2") if isinstance(getattr(mod, n, None), (nn.Linear, nn.Conv2d))]
            for k in kids:
                add(k, "mlp." + k.split(".")[-1])
            return add(fqn, "mlp.act", kids)             # everything in the MLP except fc1 / fc2
        if "SqueezeExcite" in cls or cls == "SEModule":
            return add(fqn, "squeeze_excite")
        if isinstance(mod, nn.Conv2d):
            if mod.groups > 1 and mod.groups == mod.in_channels:
                return add(fqn, "conv_depthwise")
            return add(fqn, "conv_1x1" if tuple(mod.kernel_size) == (1, 1) else "conv_kxk")
        if isinstance(mod, nn.Linear):
            return add(fqn, "linear")
        if isinstance(mod, NORMS) or re.match(r"^(Batch|Layer|Group|Rms)Norm", cls):
            return add(fqn, "norm")
        if isinstance(mod, ACTIVATIONS):
            return add(fqn, "activation")
        if "Pool" in cls:
            return add(fqn, "pool")
        for n, c in mod.named_children():
            visit(c, f"{fqn}.{n}" if fqn else n)

    visit(model, "")
    return roles


class RoleHooks:
    """Calls enter(role, fqn) / leave() around the forward pass of every tracked module."""

    def __init__(self, model, enter, leave):
        self.model, self.enter, self.leave, self.handles = model, enter, leave, []

    def __enter__(self):
        mods = dict(self.model.named_modules())
        for r in module_roles(self.model):
            def pre(module, inputs, role=r["role"], fqn=r["fqn"]):
                self.enter(role, fqn)               # hooks must return None (else they change data)

            def post(module, inputs, output):
                self.leave()
            m = mods[r["fqn"]]
            self.handles += [m.register_forward_pre_hook(pre), m.register_forward_hook(post)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


# =============================================================================
# 3. Work per role: FLOPs (counted) and bytes moved (estimated)
# =============================================================================
@torch.inference_mode()
def flops_per_image(model, img, device="cuda"):
    """FLOPs of one image per role (PyTorch FlopCounterMode; 1 multiply-add = 2 FLOPs)."""
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        model(torch.randn(1, 3, img, img, device=device))
    counts, root = fc.get_flop_counts(), type(model).__name__

    def of(fqn):
        return sum(counts.get(f"{root}.{fqn}", {}).values())
    per_role = dict.fromkeys(ROLES, 0)
    for r in module_roles(model):
        per_role[r["role"]] += of(r["fqn"]) - sum(of(c) for c in r["children"])
    per_role["other"] += fc.get_total_flops() - sum(per_role.values())
    return per_role


def _nbytes(obj):
    """Bytes of all tensors in obj (broadcast tensors count their real storage only)."""
    return sum(min(t.numel() * t.element_size(), t.untyped_storage().nbytes())
               for t in tree_leaves(obj) if isinstance(t, torch.Tensor))


class BytesPerRole(TorchDispatchMode):
    """
    Memory traffic per role in one forward pass, assuming every PyTorch op reads its inputs
    (incl. weights) and writes its outputs once. Views (reshape, transpose) and ops that just
    return their input (dropout in eval mode) move nothing. Standard roofline estimate:
    real DRAM traffic can be lower (data still in the 50 MB L2 cache) or higher (re-reads).
    """

    def __init__(self):
        super().__init__()
        self.stack, self.bytes = [], dict.fromkeys(ROLES, 0)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func.__name__                                   # e.g. "addmm.default", "add_.Tensor"
        if func.is_view or name.startswith(("empty", "new_empty")):
            return out
        ins = {(t.data_ptr(), tuple(t.shape)) for t in tree_leaves((args, kwargs)) if isinstance(t, torch.Tensor)}
        outs = [t for t in tree_leaves(out) if isinstance(t, torch.Tensor)]
        inplace = name.split(".")[0].endswith("_")
        if not inplace and outs and all((t.data_ptr(), tuple(t.shape)) in ins for t in outs):
            return out                                         # returned its input: no data moved
        self.bytes[self.stack[-1] if self.stack else "other"] += _nbytes((args, kwargs)) + _nbytes(out)
        return out


@torch.inference_mode()
def bytes_per_role(model, x):
    counter = BytesPerRole()
    with RoleHooks(model, lambda role, fqn: counter.stack.append(role), lambda: counter.stack.pop()), counter:
        model(x)
    return counter.bytes


# =============================================================================
# 4. Kernel types (from the kernel name)
# =============================================================================
KERNEL_KINDS = [  # first match wins, matched against the lower-case kernel name
    ("attention (fused)", r"fmha|flash_|flash::|attention|mem_eff|efficient_attention"),
    ("convolution", r"conv(?!ert)|fprop|dgrad|wgrad|winograd|implicit_gemm|depthwise|dwconv"),
    ("matmul (GEMM)", r"gemm|nvjet|cutlass|xmma|cublas|matmul|bmm|_mma"),
    ("softmax", r"softmax"),
    ("norm", r"norm|welford|bn_fw|batchnorm"),
    ("reduction / pool", r"reduce|meanops|sum_kernel|argmax|max_kernel|pool"),
    ("copy / layout", r"copy|transpose|permute|catarray|cat_|nchwtonhwc|nhwctonchw|tensortransform|"
                      r"roll_|index_|gather|scatter|contiguous|convert|memcpy|memset|fill_"),
    ("elementwise", r"elementwise|vectorized|unrolled|gelu|relu|silu|sigmoid|hardswish|tanh|add|mul|div|where"),
]


def kernel_kind(name):
    n = name.lower()
    return next((kind for kind, pat in KERNEL_KINDS if re.search(pat, n)), "other")


def short_kernel_name(name, max_len=140):
    """Drop 'void ' and the parameter list, keep template arguments (they name the op)."""
    n = (name[5:] if name.startswith("void ") else name).replace("(anonymous namespace)::", "")
    depth = 0
    for i, ch in enumerate(n):
        depth += (ch == "<") - (ch == ">")
        if ch == "(" and depth == 0:
            n = n[:i]
            break
    return n if len(n) <= max_len else n[:max_len] + "..."


# =============================================================================
# 5. Profiler trace -> one row per kernel with its layer role
# =============================================================================
def record_trace(model, x, path):
    """PyTorch profiler (CUPTI) trace of PROFILE_ITERS forward passes, every layer labelled."""
    from torch.profiler import ProfilerActivity, profile, record_function
    open_ranges = []

    def enter(role, fqn):
        rf = record_function(f"vt|{role}|{fqn}")
        rf.__enter__()
        open_ranges.append(rf)

    def leave():
        open_ranges.pop().__exit__(None, None, None)

    with RoleHooks(model, enter, leave):
        model(x)                                       # warm, not recorded
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(PROFILE_ITERS):
                with record_function("vt_iter"):
                    model(x)
            torch.cuda.synchronize()
    prof.export_chrome_trace(path)


class IntervalIndex:
    """Innermost CPU range containing a timestamp (ranges on one thread are strictly nested)."""

    def __init__(self, items):  # (start, end, thread, payload)
        self.by_tid = {}
        for s, e, tid, p in items:
            self.by_tid.setdefault(tid, []).append((s, e, p))
        for v in self.by_tid.values():
            v.sort(key=lambda x: (x[0], -x[1]))   # same start: outer range first
        self.starts = {tid: [x[0] for x in v] for tid, v in self.by_tid.items()}

    def innermost(self, t, tid):
        v = self.by_tid.get(tid, [])
        i = bisect.bisect_right(self.starts.get(tid, []), t) - 1
        while i >= 0:                              # latest-starting range that is still open at t
            if v[i][1] >= t:
                return v[i][2]
            i -= 1
        return None


def kernels_from_trace(path):
    """
    Every GPU kernel / memcpy of the recorded iterations, attributed to the innermost layer
    that launched it: kernel --(CUDA correlation id)--> CPU launch call --(time)--> layer range.
    Returns (DataFrame, number of iterations).
    """
    with open(path) as f:
        data = json.load(f)
    ev = [e for e in (data["traceEvents"] if isinstance(data, dict) else data)
          if e.get("ph") == "X" and "ts" in e and "dur" in e]

    def ranges(cat, keep):
        return [(float(e["ts"]), float(e["ts"]) + float(e["dur"]), e.get("tid"), e["name"])
                for e in ev if e.get("cat") == cat and keep(e["name"])]

    iters = sorted((s, e) for s, e, _, _ in ranges("user_annotation", lambda n: n == "vt_iter"))
    layers = IntervalIndex(ranges("user_annotation", lambda n: n.startswith("vt|")))
    ops = IntervalIndex(ranges("cpu_op", lambda n: True))
    launch = {e["args"]["correlation"]: (float(e["ts"]), e.get("tid")) for e in ev
              if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}

    rows = []
    for e in ev:
        if e.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        args = e.get("args", {})
        if args.get("correlation") not in launch:
            continue
        t, tid = launch[args["correlation"]]
        it = next((i for i, (s, end) in enumerate(iters) if s <= t <= end), None)
        if it is None:
            continue                               # not part of a recorded forward pass
        label = layers.innermost(t, tid)
        role, module = label.split("|")[1:3] if label else ("other", "")
        rows.append({"iter": it, "role": role, "module": module, "aten_op": ops.innermost(t, tid),
                     "kind": kernel_kind(e["name"]) if e["cat"] == "kernel" else "copy / layout",
                     "kernel": short_kernel_name(e["name"]), "start_us": float(e["ts"]), "dur_us": float(e["dur"]),
                     "grid": str(args.get("grid", "")), "block": str(args.get("block", ""))})
    return pd.DataFrame(rows), len(iters)


# =============================================================================
# 6. Timing and GPU peaks
# =============================================================================
def time_eager(model, x):
    """ms per forward pass in normal PyTorch (Python launches every kernel)."""
    model(x)
    torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < TIMING_S or n < 5:
        model(x)
        n += 1
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def time_graph(model, x):
    """ms per forward pass replayed as a CUDA graph (no CPU launch cost)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        model(x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        model(x)
    g.replay()
    torch.cuda.synchronize()
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < TIMING_S or n < 5:
        g.replay()
        n += 1
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def gpu_info():
    """GPU name, memory, and theoretical peaks of THIS GPU for the bound verdict."""
    import pynvml
    pynvml.nvmlInit()
    props = torch.cuda.get_device_properties(0)
    def clean(u):  # NVML returns str or bytes depending on version; NVML ignores CUDA_VISIBLE_DEVICES
        return (u.decode() if isinstance(u, bytes) else str(u)).replace("GPU-", "").lower()
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    h = next(h for h in handles if clean(pynvml.nvmlDeviceGetUUID(h)) == clean(props.uuid))
    sm_mhz = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)
    mem_mhz = pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)
    bus = pynvml.nvmlDeviceGetMemoryBusWidth(h)
    return {"gpu": props.name, "memory_GB": props.total_memory / 1e9, "sm_count": props.multi_processor_count,
            "sm_clock_max_MHz": sm_mhz,
            # Hopper: 2048 dense TF32 tensor-core FLOPs per SM per clock (NVIDIA H100 whitepaper)
            "peak_tf32_TFLOPs": props.multi_processor_count * sm_mhz * 1e6 * 2048 / 1e12,
            # HBM transfers twice per clock: bytes/s = bus width / 8 * 2 * clock
            "peak_dram_GBps": bus / 8 * 2 * mem_mhz * 1e6 / 1e9,
            "torch": torch.__version__, "cuda": torch.version.cuda}


def bound(gflops, mb, us, gpu):
    """
    Roofline verdict. Best possible time = max(FLOPs / peak compute, bytes / peak bandwidth);
    the larger term is the limit. Efficiency = best possible / measured. Below 30 % neither unit
    is busy -> latency-bound (too little work per kernel, launch gaps, dependencies).
    """
    t_c = gflops * 1e9 / (gpu["peak_tf32_TFLOPs"] * 1e12)
    t_m = mb * 1e6 / (gpu["peak_dram_GBps"] * 1e9)
    limit = "compute" if t_c >= t_m else "memory"
    eff = max(t_c, t_m) / (us * 1e-6)
    return 100 * eff, (f"{limit}-bound" if eff >= 0.3 else "latency-bound")


# =============================================================================
# 7. One model: batch 1, 2, 4, ... until out of memory
# =============================================================================
def run_model(name, out, max_batch):
    gpu = gpu_info()
    with open(os.path.join(out, "gpu.json"), "w") as f:
        json.dump(gpu, f, indent=2)
    model, img = load_model(name)
    flops = flops_per_image(model, img)
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    safe = name.replace("/", "_")
    summary, roles, kernels = [], [], []
    bs = 1
    while bs <= max_batch:
        trace = os.path.join(out, "traces", f"{safe}_bs{bs}.json")
        x = None
        try:
            with torch.inference_mode():
                x = torch.randn(bs, 3, img, img, device="cuda")
                torch.cuda.reset_peak_memory_stats()
                eager_ms = time_eager(model, x)
                peak_gb = torch.cuda.max_memory_allocated() / 1e9
                graph_ms = time_graph(model, x)
                record_trace(model, x, trace)
            mb = bytes_per_role(model, x)
        except RuntimeError as e:                    # out of memory (or a kernel refusing the size)
            reason = "OOM" if "out of memory" in str(e).lower() else f"{type(e).__name__}: {str(e)[:200]}"
            summary.append({"model": name, "batch_size": bs, "status": reason})
            print(f"  bs {bs:6d} | {reason} -> stop", flush=True)
            break
        finally:
            x = None
            gc.collect()
            torch.cuda.empty_cache()

        k, n_iter = kernels_from_trace(trace)
        if k.empty or n_iter == 0:
            sys.exit(f"no GPU kernels found in {trace}: the profiler (CUPTI) did not record the GPU")
        if bs not in KEEP_TRACES:
            os.remove(trace)
        k_us = k.dur_us.sum() / n_iter                   # kernel time per forward pass

        r = k.groupby("role", as_index=False).agg(gpu_us=("dur_us", "sum"), kernels=("dur_us", "size"))
        r[["gpu_us", "kernels"]] /= n_iter
        r["share_pct"] = 100 * r.gpu_us / k_us
        r["gflops"] = r.role.map(lambda x: flops.get(x, 0) * bs / 1e9)
        r["mb"] = r.role.map(lambda x: mb.get(x, 0) / 1e6)
        r["tflops"] = r.gflops / (r.gpu_us * 1e-6) / 1e3
        r["gbps"] = r.mb / 1e3 / (r.gpu_us * 1e-6)
        r["flop_per_byte"] = r.gflops * 1e3 / r.mb.where(r.mb > 0)
        verdicts = [bound(g, m, u, gpu) for g, m, u in zip(r.gflops, r.mb, r.gpu_us)]
        r["roofline_efficiency_pct"] = [v[0] for v in verdicts]
        r["bound"] = [v[1] for v in verdicts]
        r.insert(0, "batch_size", bs)
        r.insert(0, "model", name)
        roles.append(r)

        kk = k.groupby(["kernel", "kind", "role"], as_index=False).agg(
            calls=("dur_us", "size"), gpu_us=("dur_us", "sum"), aten_op=("aten_op", "first"),
            grid=("grid", "first"), block=("block", "first"))
        kk[["calls", "gpu_us"]] /= n_iter
        kk["share_pct"] = 100 * kk.gpu_us / k_us
        kk.insert(0, "batch_size", bs)
        kk.insert(0, "model", name)
        kernels.append(kk.sort_values("gpu_us", ascending=False))

        attn = k[(k.role == "attention.core") & (k.kind == "attention (fused)")]
        summary.append({
            "model": name, "batch_size": bs, "status": "ok", "img_size": img,
            "graph_ms": graph_ms, "eager_ms": eager_ms, "us_per_img": graph_ms * 1e3 / bs,
            "img_per_s": bs / (graph_ms * 1e-3), "kernel_ms": k_us / 1e3,
            "launch_overhead_pct": 100 * max(0.0, 1 - graph_ms / eager_ms),
            "kernels_per_forward": len(k) / n_iter, "peak_mem_GB": peak_gb,
            "gflops_per_img": sum(flops.values()) / 1e9,
            "attention": ("fused kernel" if len(attn) else "separate matmul + softmax kernels")
            if (k.role == "attention.core").any() else "none"})
        top = r.sort_values("gpu_us", ascending=False).head(3)
        print(f"  bs {bs:6d} | {graph_ms:9.3f} ms | {graph_ms * 1e3 / bs:9.2f} us/img | launch overhead "
              f"{summary[-1]['launch_overhead_pct']:5.1f}% | {peak_gb:6.2f} GB | "
              + ", ".join(f"{t.role} {t.share_pct:.0f}% ({t.bound})" for t in top.itertuples()), flush=True)
        for rows, tag in ((summary, "summary"), (roles, "roles"), (kernels, "kernels")):
            df = pd.DataFrame(rows) if tag == "summary" else pd.concat(rows)
            df.to_csv(os.path.join(out, f"{safe}_{tag}.csv"), index=False)
        bs *= 2
    pd.DataFrame(summary).to_csv(os.path.join(out, f"{safe}_summary.csv"), index=False)


# =============================================================================
# 8. All models, each in its own process (a crash or OOM can never affect the next model)
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="run one model (used internally)")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--max-batch", type=int, default=MAX_BATCH)
    p.add_argument("--out", default=os.path.join(RESULTS, f"profile_{os.environ.get('SLURM_JOB_ID', time.strftime('%Y%m%d-%H%M'))}"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.model:
        run_model(a.model, a.out, a.max_batch)
        return
    failed = []  # the main process never touches the GPU, so it holds no GPU memory
    for m in a.models:
        print(f"\n===== {m} =====", flush=True)
        if subprocess.call([sys.executable, __file__, "--model", m, "--out", a.out, "--max-batch", str(a.max_batch)]):
            failed.append(m)
    for tag in ("summary", "roles", "kernels"):
        files = sorted(glob.glob(os.path.join(a.out, f"*_{tag}.csv")))
        if files:
            pd.concat([pd.read_csv(f) for f in files]).to_csv(os.path.join(a.out, f"{tag}.csv"), index=False)
    print(f"\nresults: {a.out}" + (f"\nFAILED: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
