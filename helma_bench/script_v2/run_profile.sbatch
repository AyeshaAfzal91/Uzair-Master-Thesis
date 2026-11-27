#!/bin/bash -l
#SBATCH --job-name=vision-profile
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=06:00:00
#SBATCH --export=NONE
#SBATCH --output=/hnvme/workspace/ihpc180h-thesis_bench/Uzair-Master-Thesis/helma_bench/results/v2_results/logs/%x-%j.out
#SBATCH --error=/hnvme/workspace/ihpc180h-thesis_bench/Uzair-Master-Thesis/helma_bench/results/v2_results/logs/%x-%j.err
# Kernel profile of the 7 models, batch 1, 2, 4, ... until out of memory.
#   sbatch run_profile.sbatch                                                   (full run)
#   sbatch run_profile.sbatch --models resnet50.a1_in1k --max-batch 4          (10-minute check)
unset SLURM_EXPORT_ENV

ROOT=/hnvme/workspace/ihpc180h-thesis_bench/Uzair-Master-Thesis
cd $ROOT/helma_bench/script_v2

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export TORCH_HOME=$ROOT/model
export HF_HOME=$ROOT/model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi
apptainer exec --nv --bind /hnvme $ROOT/helma_bench/apptainer/vision_env.sif python profile_vision.py "$@"
