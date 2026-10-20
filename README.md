## Administrative Tasks

### GitHub
**Repository:** https://github.com/AyeshaAfzal91/Uzair-Master-Thesis
- Document a **summary of changes and steps taken** for each **task** in the corresponding **README files**.
- Always commit the latest results to GitHub.

### Student Email and Accounts/username

| Account | Address |
|---------|---------|
| FAU | TODO |
| Gmail | TODO |
| GitHub | TODO |
| Matrix | TODO |

### Status Overview

| Item | Status |
|------|--------|
| Thesis listed on the website | TODO |
| GitHub access | TODO |
| Student added to the mailing list (hpcplus@fau.de) | TODO |
| Weekly meetings (Fridays at 10 a.m.): calendar invitation set up | TODO |
| Matrix room for chat (create an account at matrix.org) | TODO |
| Access to HPC systems (Fritz/Alex/Helma/testcluster) at NHR@FAU (Portal: https://portal.hpc.fau.de; login: `ssh -Y ihpcXXXh@fritz.nhr.fau.de`) | TODO |

### Thesis Timeline

- **Thesis requirements:** Two presentations and a comprehensive written thesis.
- **Thesis duration:** November–April (flexible; to be confirmed).
- **Thesis registration:** November (flexible; to be confirmed).
- **First presentation:** November/December (flexible; to be confirmed).
- **Second presentation and thesis submission:** By the end of April (flexible; to be confirmed).


---

## Thesis Context 

### Hardware
- H100/H200 Helma or Testcluster GPUs?

### Applications (compute-bound and/or memory-bound)

#### Three LLMs (focus on dense)
1. **Mamba2 hybrid** (state space models)
   - https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
2. **MLA** (multi-head-latent attention from deepseek)
   - TransMLA this model: https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Depth-Base
3. **GQA** (group query attention)
   - https://huggingface.co/nvidia/Llama-3.1-Minitron-4B-Depth-Base

#### AI Benchmarks from Computer Vision
- **FLUX1 dev:** https://huggingface.co/black-forest-labs/FLUX.1-dev
- **ResNet-50:** https://huggingface.co/microsoft/resnet-50

### Reference Papers (running knobs: freq scaling and power capping)
- **LLM:** https://arxiv.org/abs/2605.11999
  - Hardware: NVIDIA H200
  - LLM models: GQA, MLA, Gated DeltaNet, and Mamba2
- **AI:** https://arxiv.org/abs/2603.16164
  - Hardware: NVIDIA H100, NVIDIA H200, and AMD MI300X GPUs
  - AI model: a benchmarking framework with popular deep learning applications from computer vision (image classification and generation) and large language models (continued pre-training and inference) implementing modern methods.

### Experimental Setting (General)
- Always use the updated software version available.  
