#!/bin/bash -l
#SBATCH --job-name=vision-fp32
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=vision-fp32-%j.out
unset SLURM_EXPORT_ENV

ROOT=/hnvme/workspace/ihpc180h-thesis_bench/Uzair-Master-Thesis
cd $ROOT/helma_bench

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export TORCH_HOME=$ROOT/model
export HF_HOME=$ROOT/model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi
apptainer exec --nv --bind /hnvme apptainer/vision_env.sif python bench_vision.py
