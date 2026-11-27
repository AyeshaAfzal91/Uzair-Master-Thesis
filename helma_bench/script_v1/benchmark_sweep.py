import torch
import timm
import time
import pandas as pd
import subprocess
import threading

class PowerMonitor:
    def __init__(self):
        self.running = False
        self.measurements = []
    
    def _poll(self):
        while self.running:
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], 
                    text=True
                )
                self.measurements.append(float(out.strip()))
            except:
                pass
            time.sleep(0.05)
    
    def start(self):
        self.running = True
        self.measurements = []
        self.thread = threading.Thread(target=self._poll)
        self.thread.start()
        
    def stop(self):
        self.running = False
        self.thread.join()
        if not self.measurements:
            return 0.0
        return sum(self.measurements) / len(self.measurements)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Precision: BF16 (Bfloat16)\n")

model_name = 'resnet50'
# Cast model natively to BF16
model = timm.create_model(model_name, pretrained=False).cuda().bfloat16().eval()

batch_sizes = [1, 16, 64, 128]
resolutions = [224, 384]

results = []
monitor = PowerMonitor()

for res in resolutions:
    for bs in batch_sizes:
        print(f"Benchmarking {model_name} | Batch: {bs} | Res: {res}x{res}...")
        
        # Cast inputs natively to BF16
        static_input = torch.randn(bs, 3, res, res, device='cuda', dtype=torch.bfloat16)
        
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
        iters = 100
        
        monitor.start()
        start = time.time()
        
        for _ in range(iters):
            g.replay()
            
        torch.cuda.synchronize()
        end = time.time()
        avg_watts = monitor.stop()
        
        total_time = end - start
        ms_per_batch = (total_time / iters) * 1000
        images_per_sec = (bs * iters) / total_time
        total_joules = avg_watts * total_time
        joules_per_image = total_joules / (bs * iters)
        
        results.append({
            "Batch": bs, 
            "Res": res, 
            "Img/sec": round(images_per_sec, 2),
            "Avg Watts": round(avg_watts, 2),
            "Total Joules": round(total_joules, 2),
            "Joules/Img": round(joules_per_image, 4)
        })

df = pd.DataFrame(results)
print("\n--- Energy & Throughput Results (BF16) ---")
print(df.to_string(index=False))
