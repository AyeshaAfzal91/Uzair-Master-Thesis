"""
Energy / throughput benchmark of the 7 thesis models on one H100.
FP32 weights and inputs, TF32 math (same setting as timm's benchmark and accuracy table).

For every model and batch size 1, 2, 4, 8, ... (doubling until the GPU runs out of memory):
  throughput (img/s), time per batch (ms), average power (W), energy per image (J/img),
  SM clock, temperature, how often the power limit held the GPU back, peak GPU memory,
  GFLOPs per image and achieved TFLOP/s.

Energy comes from the GPU's built-in energy counter (NVML), read before and after each
measurement window. A background thread samples power/clock/temp every 100 ms for context.

Per batch size: CUDA graph capture -> 15 s warmup -> 3 measurement windows of >= 30 s.

Output: helma_bench/results/v2_results/bench_<jobid>/bench.csv  (one row per model, batch size and repeat)
Run:    python bench_vision.py                                    (all models)
        python bench_vision.py --models resnet50.a1_in1k --max-batch 4     (quick check)
Plots:  python plot_bench.py ../results/v2_results/bench_<jobid>
"""
import argparse
import gc
import glob
import os
import subprocess
import sys
import threading
import time

import pandas as pd
import pynvml
import timm
import torch
from timm.data import resolve_data_config

# ---------------------------------------------------------------- settings
MODELS = [  # proposal, section "A - Vision"
    "resnet50.a1_in1k",
    "convnext_tiny.in12k_ft_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in1k",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
]
MAX_BATCH = 65536  # batch doubles from 1 until out of memory (reached long before this)
WARMUP_S = 15      # run this long before measuring, so clocks/power/temp settle
MEASURE_S = 30     # each measurement window lasts at least this long
REPEATS = 3        # measurement windows per (model, batch size)
SAMPLE_S = 0.1     # power/clock sampling period
HERE = os.path.dirname(os.path.abspath(__file__))                         # helma_bench/script_v2
RESULTS = os.path.normpath(os.path.join(HERE, "..", "results", "v2_results"))  # helma_bench/results/v2_results

# Standard timm benchmark.py float32 settings (TF32 tensor-core math)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------- GPU monitoring
def nvml_handle():
    """NVML handle of the GPU PyTorch is using (matched by UUID)."""
    pynvml.nvmlInit()
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).replace("GPU-", "")
    return pynvml.nvmlDeviceGetHandleByUUID(uuid)


class Sampler(threading.Thread):
    """Samples power, SM clock, temperature and power-cap throttling in the background."""

    def __init__(self, handle):
        super().__init__(daemon=True)
        self.h = handle
        self.samples = []
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            reasons = pynvml.nvmlDeviceGetCurrentClocksEventReasons(self.h)
            self.samples.append({
                "power_W": pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000,
                "sm_MHz": pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM),
                "temp_C": pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU),
                "power_capped": bool(reasons & pynvml.nvmlClocksEventReasonSwPowerCap),
            })
            time.sleep(SAMPLE_S)

    def stop(self):
        self.stop_flag.set()
        self.join()
        return pd.DataFrame(self.samples)


def energy_J(handle):
    """Total energy the GPU has used since the driver loaded, in joules."""
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000


def idle_power_W(handle, seconds=10):
    e0, t0 = energy_J(handle), time.time()
    time.sleep(seconds)
    return (energy_J(handle) - e0) / (time.time() - t0)


@torch.inference_mode()
def gflops_per_image(model, img_size):
    """Operations to classify one image (PyTorch FlopCounterMode, 1 multiply-add = 2 FLOPs)."""
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        model(torch.randn(1, 3, img_size, img_size, device="cuda"))
    return fc.get_total_flops() / 1e9


# ---------------------------------------------------------------- benchmark
def capture_cuda_graph(model, x):
    """Record one forward pass as a CUDA graph, so replaying it has no Python/launch overhead."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            model(x)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        model(x)
    torch.cuda.synchronize()
    return graph


def warmup(graph):
    """Replay for WARMUP_S seconds; return the measured seconds per iteration."""
    iters, t0 = 0, time.time()
    while time.time() - t0 < WARMUP_S:
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        iters += 10
    return (time.time() - t0) / iters


def measure(graph, handle, n_iters):
    """One measurement window of n_iters replays."""
    sampler = Sampler(handle)
    sampler.start()
    torch.cuda.synchronize()
    e0, t0 = energy_J(handle), time.time()
    for _ in range(n_iters):
        graph.replay()
    torch.cuda.synchronize()
    seconds, joules = time.time() - t0, energy_J(handle) - e0
    return seconds, joules, sampler.stop()


def run_model(name, out_dir, max_batch):
    handle = nvml_handle()
    gpu = pynvml.nvmlDeviceGetName(handle)
    limit_W = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
    idle_W = idle_power_W(handle)

    model = timm.create_model(name, pretrained=True).cuda().eval()
    img_size = resolve_data_config({}, model=model, use_test_size=True)["input_size"][1]  # timm's eval size
    params_M = sum(p.numel() for p in model.parameters()) / 1e6
    gflops = gflops_per_image(model, img_size)
    print(f"{name}  ({img_size}px, {params_M:.1f}M params, {gflops:.2f} GFLOPs/img) | GPU {gpu} | "
          f"power limit {limit_W:.0f} W | idle {idle_W:.0f} W", flush=True)

    out_csv = os.path.join(out_dir, name.replace("/", "_") + ".csv")
    rows = []
    bs = 1
    while bs <= max_batch:
        info = {"model": name, "img_size": img_size, "params_M": params_M, "batch_size": bs,
                "gpu": gpu, "power_limit_W": limit_W, "idle_power_W": idle_W, "tf32": True,
                "gflops_per_img": gflops}
        x = graph = None
        try:
            with torch.inference_mode():
                x = torch.randn(bs, 3, img_size, img_size, device="cuda")
                torch.cuda.reset_peak_memory_stats()
                graph = capture_cuda_graph(model, x)
                sec_per_iter = warmup(graph)
                n_iters = max(20, int(MEASURE_S / sec_per_iter))

                for r in range(REPEATS):
                    seconds, joules, s = measure(graph, handle, n_iters)
                    images = n_iters * bs
                    row = {**info, "status": "ok", "repeat": r, "iters": n_iters,
                           "img_per_s": images / seconds,
                           "ms_per_batch": seconds / n_iters * 1000,
                           "avg_power_W": joules / seconds,
                           "J_per_img": joules / images,
                           "achieved_TFLOPs": gflops * images / seconds / 1000,
                           "sm_clock_MHz": s.sm_MHz.mean(),
                           "max_temp_C": s.temp_C.max(),
                           "power_capped_frac": s.power_capped.mean(),
                           "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
                    rows.append(row)
                    print(f"  bs {bs:6d} | {row['img_per_s']:9.1f} img/s | {row['ms_per_batch']:9.2f} ms/batch"
                          f" | {row['avg_power_W']:5.0f} W | {row['J_per_img'] * 1000:8.3f} mJ/img"
                          f" | {row['sm_clock_MHz']:5.0f} MHz | capped {row['power_capped_frac']:4.0%}"
                          f" | {row['peak_mem_GB']:5.1f} GB", flush=True)
        except RuntimeError as e:  # out of memory (or a kernel refusing the size): stop this model here
            reason = "OOM" if "out of memory" in str(e).lower() else f"{type(e).__name__}: {str(e)[:200]}"
            rows.append({**info, "status": reason})
            print(f"  bs {bs:6d} | {reason} -> largest batch for this model: {bs // 2}", flush=True)
            break
        finally:
            x = graph = None
            gc.collect()
            torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(out_csv, index=False)  # saved after every batch size
        bs *= 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="run one model (used internally: one process per model)")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--max-batch", type=int, default=MAX_BATCH)
    p.add_argument("--out", default=os.path.join(RESULTS, f"bench_{os.environ.get('SLURM_JOB_ID', time.strftime('%Y%m%d-%H%M'))}"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.model:
        run_model(a.model, a.out, a.max_batch)
        return
    # each model in its own process: a crash or OOM can never affect the next model
    failed = [m for m in a.models if subprocess.call(
        [sys.executable, __file__, "--model", m, "--out", a.out, "--max-batch", str(a.max_batch)])]
    files = [f for f in sorted(glob.glob(os.path.join(a.out, "*.csv"))) if os.path.basename(f) != "bench.csv"]
    if files:
        pd.concat([pd.read_csv(f) for f in files]).to_csv(os.path.join(a.out, "bench.csv"), index=False)
    print(f"\nresults: {a.out}/bench.csv" + (f"\nFAILED: {failed}" if failed else ""))


if __name__ == "__main__":
    main()
