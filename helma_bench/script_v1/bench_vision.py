import gc
import os
import threading
import time

import pandas as pd
import pynvml
import timm
import torch
from timm.data import resolve_data_config

# ---------------------------------------------------------------- settings
MODELS = [
    "resnet50.a1_in1k",
    "convnext_tiny.in12k_ft_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in1k",
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224.augreg_in1k",
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
]
BATCH_SIZES = [1, 16, 64, 128, 256, 512]
WARMUP_S = 15      # run this long before measuring, so clocks/power/temp settle
MEASURE_S = 30     # each measurement window lasts at least this long
REPEATS = 3        # measurement windows per (model, batch size)
SAMPLE_S = 0.1     # power/clock sampling period

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
JOB = os.environ.get("SLURM_JOB_ID", time.strftime("%Y%m%d-%H%M"))
OUT_CSV = os.path.join(OUT_DIR, f"vision_fp32_{JOB}.csv")

# Standard timm benchmark.py float32 settings
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    handle = nvml_handle()
    gpu = pynvml.nvmlDeviceGetName(handle)
    limit_W = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
    idle_W = idle_power_W(handle)
    print(f"GPU: {gpu} | power limit {limit_W:.0f} W | idle {idle_W:.0f} W | writing {OUT_CSV}")

    rows = []
    for name in MODELS:
        model = timm.create_model(name, pretrained=True).cuda().eval()
        img_size = resolve_data_config({}, model=model, use_test_size=True)["input_size"][1]
        params_M = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\n{name}  ({img_size}px, {params_M:.1f}M params)")

        for bs in BATCH_SIZES:
            info = {"model": name, "img_size": img_size, "params_M": params_M, "batch_size": bs,
                    "gpu": gpu, "power_limit_W": limit_W, "idle_power_W": idle_W}
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
                               "sm_clock_MHz": s.sm_MHz.mean(),
                               "max_temp_C": s.temp_C.max(),
                               "power_capped_frac": s.power_capped.mean(),
                               "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9}
                        rows.append(row)
                        print(f"  bs {bs:4d} | {row['img_per_s']:9.1f} img/s | {row['ms_per_batch']:8.2f} ms/batch"
                              f" | {row['avg_power_W']:5.0f} W | {row['J_per_img'] * 1000:8.2f} mJ/img"
                              f" | {row['sm_clock_MHz']:5.0f} MHz | capped {row['power_capped_frac']:.0%}")
            except torch.OutOfMemoryError:
                rows.append({**info, "status": "OOM"})
                print(f"  bs {bs:4d} | OOM, skipping larger batches")
                pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
                break
            finally:
                x = graph = None
                gc.collect()
                torch.cuda.empty_cache()

            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nDone: {OUT_CSV}")


if __name__ == "__main__":
    main()
