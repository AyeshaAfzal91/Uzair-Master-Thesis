import torch
import timm
import time
import pandas as pd
import subprocess
import threading
import gc

class HardwareMonitor:
    def __init__(self):
        self.running = False
        self.power, self.gpu_util, self.mem_util = [], [], []
    
    def _poll(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"], 
                    text=True
                )
                p, g, m = map(float, out.strip().split(', '))
                self.power.append(p)
                self.gpu_util.append(g)
                self.mem_util.append(m)
            except:
                pass
            time.sleep(0.05)
    
    def start(self):
        self.running = True
        self.power, self.gpu_util, self.mem_util = [], [], []
        self.thread = threading.Thread(target=self._poll)
        self.thread.start()
        
    def stop(self):
        self.running = False
        self.thread.join()
        if not self.power:
            return 0.0, 0.0, 0.0
        avg = lambda lst: sum(lst) / len(lst)
        return avg(self.power), avg(self.gpu_util), avg(self.mem_util)

print(f"Hardware: {torch.cuda.get_device_name(0)}")
print("Initializing Comprehensive Sweep...\n")

model_name = 'resnet50'
base_model = timm.create_model(model_name, pretrained=False).cuda().eval()

precisions = {"FP32": torch.float32, "BF16": torch.bfloat16}
resolutions = [224, 384]
batch_sizes = [1, 16, 64, 256, 512] 
iters = 100

results = []
monitor = HardwareMonitor()

for prec_name, dtype in precisions.items():
    model = base_model.to(dtype=dtype)
    
    for res in resolutions:
        for bs in batch_sizes:
            print(f"Running: {prec_name} | {model_name} | BS: {bs} | Res: {res}x{res}")
            
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

            try:
                static_input = torch.randn(bs, 3, res, res, device='cuda', dtype=dtype)
                
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        _ = model(static_input)
                torch.cuda.current_stream().wait_stream(s)
                
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    static_output = model(static_input)
                    
                torch.cuda.synchronize()
                monitor.start()
                start = time.time()
                
                for _ in range(iters):
                    g.replay()
                    
                torch.cuda.synchronize()
                end = time.time()
                avg_watts, avg_gpu_util, avg_mem_util = monitor.stop()
                
                del g, static_input, static_output
                
                total_time = end - start
                ms_per_batch = (total_time / iters) * 1000
                images_per_sec = (bs * iters) / total_time
                total_joules = avg_watts * total_time
                joules_per_image = total_joules / (bs * iters)
                peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
                
                results.append({
                    "Precision": prec_name, "Res": res, "Batch": bs, 
                    "Img/sec": round(images_per_sec, 0), "ms/batch": round(ms_per_batch, 2),
                    "Joules/Img": round(joules_per_image, 4), "Avg Watts": round(avg_watts, 1),
                    "GPU Util%": round(avg_gpu_util, 1), "Mem Util%": round(avg_mem_util, 1),
                    "VRAM(GB)": round(peak_vram_gb, 2)
                })
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("  -> CUDA OOM: Not enough VRAM. Logging and skipping configuration.")
                    if monitor.running:
                        monitor.stop()
                    results.append({
                        "Precision": prec_name, "Res": res, "Batch": bs, 
                        "Img/sec": "OOM", "ms/batch": "OOM", "Joules/Img": "OOM",
                        "Avg Watts": "OOM", "GPU Util%": "OOM", "Mem Util%": "OOM", "VRAM(GB)": "OOM"
                    })
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise e

df = pd.DataFrame(results)
csv_name = "comprehensive_benchmark_results.csv"
df.to_csv(csv_name, index=False)

print("\n--- Comprehensive Benchmark Results ---")
print(df.to_string(index=False))
print(f"\nResults saved to {csv_name}")
