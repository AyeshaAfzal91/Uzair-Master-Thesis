"""
Adds two columns to a bench_vision.py results CSV (no GPU, no re-benchmarking):

  gflops_per_img   billions of floating-point operations to classify one image.
                   Fixed per (model, input size): it does not depend on the GPU or the run.
  achieved_TFLOPs  trillions of operations the GPU actually completed per second
                   = gflops_per_img * img_per_s / 1000  -> how efficiently the model uses the GPU

Counting convention: one multiply-add = 2 FLOPs (PyTorch FlopCounterMode).
timm's own tables report GMACs, which are half of these GFLOPs.

Usage:  python add_flops.py results/vision_fp32_<jobid>.csv
Writes  results/vision_fp32_<jobid>_flops.csv   (the original file is not changed)
"""
import sys

import pandas as pd
import timm
import torch
from torch.utils.flop_counter import FlopCounterMode

csv = sys.argv[1]
df = pd.read_csv(csv)

gflops = {}
for (model_name, img_size), _ in df.groupby(["model", "img_size"]):
    # Operation count depends only on the architecture, so random weights are fine (no download)
    model = timm.create_model(model_name, pretrained=False).eval()
    x = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    gflops[(model_name, img_size)] = counter.get_total_flops() / 1e9
    print(f"{model_name:45s} {img_size}px  {gflops[(model_name, img_size)]:6.2f} GFLOPs/image")

df["gflops_per_img"] = [gflops[(m, s)] for m, s in zip(df.model, df.img_size)]
df["achieved_TFLOPs"] = df.gflops_per_img * df.img_per_s / 1000

out = csv.replace(".csv", "_flops.csv")
df.to_csv(out, index=False)
print(f"\nwrote {out}")
