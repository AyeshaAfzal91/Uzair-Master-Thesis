import torch
import pynvml
import time
import gc
import pandas as pd
from transformers import AutoModelForCausalLM

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

MODELS = {
    "Mamba2-130m": "state-spaces/mamba2-130m",
    "Pythia-70m": "EleutherAI/pythia-70m",
    "Pythia-160m": "EleutherAI/pythia-160m"
}

SEQ_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768]
BATCH_SIZES = [1, 2, 4, 8, 16, 32]

TARGET_DURATION_SEC = 2.0  

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

def benchmark_workload():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    results = []

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*50}\n[{model_name}] Loading in bfloat16...\n{'='*50}")
        try:
            if "mamba" in model_name.lower():
                model = MambaLMHeadModel.from_pretrained(
                    model_id, 
                    device="cuda", 
                    dtype=torch.bfloat16
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    torch_dtype=torch.bfloat16, 
                    trust_remote_code=True
                ).to("cuda")
                
            model.eval()
            
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        for bsz in BATCH_SIZES:
            for seq_len in SEQ_LENGTHS:
                print(f"  -> Batch: {bsz:2d} | Seq: {seq_len:5d} | ", end="", flush=True)
                
                # Dummy input (Single-stage inference mapping to prefill phase)
                input_ids = torch.randint(0, 1000, (bsz, seq_len), dtype=torch.long, device="cuda")
                
                try:
                    with torch.no_grad():
                        # TIME-BASED WARMUP 
                        # Force the RTX 3070 out of idle state (P8) into P0 performance state
                        warmup_end = time.time() + 1.0 
                        while time.time() < warmup_end:
                            _ = model(input_ids)
                        torch.cuda.synchronize()

                        # ADAPTIVE ITERATION CALIBRATION
                        start_calib = torch.cuda.Event(enable_timing=True)
                        end_calib = torch.cuda.Event(enable_timing=True)
                        
                        start_calib.record()
                        _ = model(input_ids)
                        end_calib.record()
                        torch.cuda.synchronize()
                        
                        single_pass_ms = start_calib.elapsed_time(end_calib)
                        # Ensure we queue enough passes, min 3 iterations
                        num_iters = max(3, int((TARGET_DURATION_SEC * 1000) / single_pass_ms))

                        # EXPLICIT ASYNC POLLING AROUND THE FORWARD PASS
                    
                        power_mw = []
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        
                        start_event.record()
                        
                        # PyTorch execution is async: queue all iterations to the GPU stream instantly
                        for _ in range(num_iters):
                            _ = model(input_ids)
                            
                        end_event.record()
                        
                        # CPU-side explicit polling loop: 
                        # The CPU spins and polls NVML while trapped waiting for the GPU end event.
                        while not end_event.query():
                            power_mw.append(pynvml.nvmlDeviceGetPowerUsage(handle))
                            time.sleep(0.005) # Oversample at 5ms intervals
                            
                        torch.cuda.synchronize()
                        
                        # METRICS CALCULATION
        
                        if len(power_mw) == 0:
                            power_mw.append(pynvml.nvmlDeviceGetPowerUsage(handle))
                            
                        avg_power_w = (sum(power_mw) / len(power_mw)) / 1000.0
                        total_time_s = start_event.elapsed_time(end_event) / 1000.0
                        
                        total_tokens = bsz * seq_len * num_iters
                        throughput = total_tokens / total_time_s
                        
                        # Energy (Joules) = Power (Watts) * Time (Seconds)
                        total_energy_j = avg_power_w * total_time_s
                        energy_per_token_mj = (total_energy_j / total_tokens) * 1000.0

                        print(f"✅ {throughput:7.0f} tok/s | {energy_per_token_mj:7.4f} mJ/tok | {len(power_mw):3d} polls")
                        
                        results.append({
                            "Model": model_name,
                            "Batch_Size": bsz,
                            "Seq_Len": seq_len,
                            "Throughput_tok_s": throughput,
                            "Energy_per_token_mJ": energy_per_token_mj,
                            "Avg_Power_W": avg_power_w,
                            "OOM": False
                        })

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print("❌ OOM (Memory Exhausted)")
                        results.append({
                            "Model": model_name,
                            "Batch_Size": bsz,
                            "Seq_Len": seq_len,
                            "Throughput_tok_s": None,
                            "Energy_per_token_mJ": None,
                            "Avg_Power_W": None,
                            "OOM": True
                        })
                        flush_memory()
                        # If a model OOMs at 8K, it will definitely OOM at 16K. 
                        # Break to the next batch size to save execution time.
                        break
                    else:
                        raise e
                
                # Cleanup between configurations
                del input_ids
                flush_memory()

        # Aggressive cleanup between models
        del model
        flush_memory()

    # Save to CSV for the plotting script
    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nBenchmark complete! Results saved to benchmark_results.csv")
    pynvml.nvmlShutdown()

if __name__ == "__main__":
    benchmark_workload()