## Nov 27

### DONE
- Measurements are there.

### TODO
- Understand the non-attention softmax for Swin-T
- ViT-S appears to be more energy-efficient than EfficientNetV2-S in terms of energy per image. Why is this the case? How does their performance compare?
- How do different types of kernels (GEMM, attention, and others) differ in terms of performance characteristics? Examine the underlying PyTorch implementations and determine whether they are compute-bound or memory-bound using roofline analysis in profilers such as NVIDIA Nsight Compute. Investigate the performance bottlenecks. How are convolution layers (Conv1, Conv2, and Conv3×3) implemented?
- What assumptions are being made? To what extent do they hold, and how have they been verified?

## Nov 16
### DONE
- Added 7 models from timm's library and proposal
- Benchmark script created with same config as timm
- Benchmark ran on single node H100
- Result graphs for throughput, energy, power and frontier generated
### TODO
- Understand and find out for batch 1, which kernels used for compute, how much time and does it scales with batch size
- types of kernels used in different models, convulution, reduction, track which kernels are dispatched by pytorch realtime
- step size difference must be cosistent
- use raw data and fix batch size for frontier
- find out key differences between convulution and transformer computes, memory bound stages, is attention cheaper, look at different cnns and difference, vision tower
- until what batch size can we push these models
- error and out files
- folder organisation
- study lengths of compute and differences, where the time and bottlenecks, compute or memory bound, batch size scaling

## Nov 8
* DONE
- Account setup, helma access completed
- Connection established, temp workspace created for 90 days (/hnvme/workspace/ihpc180h-thesis_bench)
- Wrote some basic script for Resnet50
- TO DO LATER: Access to change TDP and Clock speed
* TODO
- Add multiple models and basic graphs
- 

Nov 2
- All administrative work has been completed.
- The topic has been proposed and is pending review by Uzair.
- Start with Helma 100, incorporating any proposed AI benchmark domain.
- Further progress and discussion will take place next week.
