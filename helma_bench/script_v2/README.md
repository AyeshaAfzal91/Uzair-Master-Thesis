# Vision benchmark on Helma H100 — helma_bench/script_v2/

Two measurements, each with its own plot script and sbatch file. Both use the 7 proposal models,
FP32 weights with TF32 math (same as timm), and batch sizes 1, 2, 4, 8, ... doubling until the GPU
runs out of memory. Each model runs in its own process, so one failure never affects the others.

| Step | Run (GPU) | Plot (no GPU) | Answers |
|---|---|---|---|
| 1. Energy & throughput | `run_bench.sbatch` → `bench_vision.py` | `plot_bench.py` | img/s, J/img, power, SM clock, power cap, memory, largest batch |
| 2. Kernel profile | `run_profile.sbatch` → `profile_vision.py` | `plot_profile.py` | which kernels run, for which layer, how long, compute- or memory-bound, scaling |

Folders:
```
helma_bench/
├── script_v2/            these 7 files
├── apptainer/vision_env.sif   container (used by the sbatch files)
└── results/v2_results/
    ├── logs/             <job-name>-<jobid>.out (output) and .err (errors)
    ├── bench_<jobid>/    bench.csv + plots/
    └── profile_<jobid>/  summary/roles/kernels.csv, traces/ + plots/
```
Model weights are cached in `Uzair-Master-Thesis/model/` (TORCH_HOME / HF_HOME).

## Run

```bash
cd /hnvme/workspace/ihpc180h-thesis_bench/Uzair-Master-Thesis/helma_bench/script_v2
mkdir -p ../results/v2_results/logs    # once: Slurm cannot create the log folder itself

# 10-minute checks first
sbatch run_bench.sbatch   --models resnet50.a1_in1k --max-batch 4
sbatch run_profile.sbatch --models resnet50.a1_in1k --max-batch 4

# full runs (run them one after the other, not at the same time)
sbatch run_bench.sbatch           # ~4-8 h
sbatch run_profile.sbatch         # ~1-3 h

# plots (login node)
SIF=../apptainer/vision_env.sif
apptainer exec --bind /hnvme $SIF python plot_bench.py   ../results/v2_results/bench_<jobid>
apptainer exec --bind /hnvme $SIF python plot_profile.py ../results/v2_results/profile_<jobid>
```

## 1. Energy & throughput — results/v2_results/bench_<jobid>/

`bench.csv`: one row per model, batch size and repeat — `img_per_s`, `ms_per_batch`, `avg_power_W`,
`J_per_img`, `sm_clock_MHz`, `max_temp_C`, `power_capped_frac`, `peak_mem_GB`, `gflops_per_img`,
`achieved_TFLOPs`. The row with status `OOM` is the first batch size that does not fit.

`plots/`: `throughput.png`, `energy.png`, `power.png`, `sm_clock.png`, `summary.csv`
(largest batch, energy at batch 1, lowest energy per image and at which batch, peak throughput).

Method: CUDA graph → 15 s warmup → 3 windows of ≥ 30 s. Energy from the GPU's energy counter.

## 2. Kernel profile — results/v2_results/profile_<jobid>/

`summary.csv`, `roles.csv`, `kernels.csv`, `gpu.json`, `traces/` (batch 1 and 256 timelines,
open in https://ui.perfetto.dev to see every kernel PyTorch dispatches).

`plots/`:

| File | Question |
|---|---|
| `batch1_kernels.csv` | which kernels run at batch 1, for which layer, launched by which PyTorch op, how long |
| `kernel_types_bs1.png`, `_bs256.png` | time per kernel type (matmul, convolution, attention, norm, ...) |
| `layers_bs1.png`, `_bs256.png` | time per layer (attention Q/K/V, attention core, MLP, conv, depthwise conv, norm) |
| `roofline_bs1.png`, `_bs256.png`, `bottlenecks_*.csv` | compute-bound, memory-bound or latency-bound per layer |
| `time_per_image.png`, `scaling_by_layer.png` | how the cost per image changes with batch size |
| `launch_overhead.png` | where the CPU, not the GPU, is the bottleneck |
| `memory.png`, `max_batch.csv` | largest batch that fits in 94 GB |

Bytes moved per layer are an estimate (each op reads its inputs and writes its outputs once); FLOPs are counted.
