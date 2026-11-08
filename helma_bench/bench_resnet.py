import os
import torch
import timm
import time

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

print("Loading ResNet-50...")
model = timm.create_model('resnet50', pretrained=True)
model = model.cuda()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224).cuda()

print("Running warmup...")
with torch.no_grad():
    for _ in range(5):
        model(dummy_input)

print("Running test inference...")
start = time.time()
with torch.no_grad():
    output = model(dummy_input)
end = time.time()

print(f"Success! Model output shape: {output.shape}")
print(f"Single image inference time: {(end-start)*1000:.2f} milliseconds")
