# /workspace/final_pipeline.py — Part 1
import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 1. 모델 준비
model_fp32 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model_fp32.eval()

# 2. ONNX 변환
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model_fp32,
    dummy_input,
    "resnet18.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes={
        "input":  {0: "batch"},
        "output": {0: "batch"}
    },
    dynamo=False   # 최신 PyTorch에서 레거시(트레이싱) 방식 고정
)
print("[Part 1] resnet18.onnx 변환 완료")

# 3. 검증용 이미지 전처리 준비
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img = Image.open("image/test.jpg").convert("RGB")
input_tensor = preprocess(img).unsqueeze(0)
x_np = input_tensor.numpy().astype(np.float32)

with torch.no_grad():
    output_pytorch = model_fp32(input_tensor)
pred_pytorch = np.argmax(output_pytorch.numpy(), axis=1)[0]
print(f"[Part 1] PyTorch 예측 클래스: {pred_pytorch}")

# Part 2-1
import torch.quantization as tq

model_dynamic = tq.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)
print("[Part 2-1] Dynamic Quantization 완료")

# Part 2-2
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
import os

class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, image_folder):
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
        ])
        self.image_paths = [os.path.join(image_folder, f)
            for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))]
        self.datas = []
        for p in self.image_paths:
            img_c = Image.open(p).convert("RGB")
            x = self.preprocess(img_c).unsqueeze(0).numpy().astype(np.float32)
            self.datas.append({"input": x})
        self.enum_data = iter(self.datas)

    def get_next(self):
        return next(self.enum_data, None)

reader = ImageCalibrationReader("image")
quantize_static(
    model_input="resnet18.onnx",
    model_output="resnet18_int8_static.onnx",
    calibration_data_reader=reader,
    weight_type=QuantType.QInt8,
)
print("[Part 2-2] Static Quantization 완료")

# Part 2-3
torch.save(model_fp32.state_dict(), "final_resnet18_fp32.pth")
torch.save(model_dynamic, "final_resnet18_dynamic.pth")
print("[Part 2-3] 모델 저장 완료")

# Part 3
import onnxruntime as ort

sess_fp32 = ort.InferenceSession("resnet18.onnx", providers=["CPUExecutionProvider"])
sess_int8 = ort.InferenceSession("resnet18_int8_static.onnx", providers=["CPUExecutionProvider"])

def measure(run_fn, n_warmup=10, n_runs=50):
    for _ in range(n_warmup):
        run_fn()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return np.mean(times), np.std(times)

mean_pt, std_pt = measure(lambda: model_fp32(input_tensor))
mean_dyn, std_dyn = measure(lambda: model_dynamic(input_tensor))
mean_onnx, std_onnx = measure(lambda: sess_fp32.run(None, {"input": x_np}))
mean_int8, std_int8 = measure(lambda: sess_int8.run(None, {"input": x_np}))

print(f"[Part 3] PyTorch FP32      : {mean_pt:.3f}ms (±{std_pt:.3f})")
print(f"[Part 3] PyTorch Dynamic   : {mean_dyn:.3f}ms (±{std_dyn:.3f})")
print(f"[Part 3] ONNX Runtime FP32 : {mean_onnx:.3f}ms (±{std_onnx:.3f})")
print(f"[Part 3] ONNX Runtime INT8 : {mean_int8:.3f}ms (±{std_int8:.3f})")

# Part 4
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

# 간단한 MNIST용 CNN (실습용, 빠르게 준비 가능)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

mnist_model = SimpleCNN()
mnist_model.eval()  # 실습 목적상 사전 학습 없이 랜덤 가중치로 진행 (정확도 수치보다 "절차" 확인이 목표)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def evaluate_accuracy(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total * 100, correct, total

# FP32와 양자화 버전 정확도 비교
mnist_dynamic = tq.quantize_dynamic(mnist_model, {torch.nn.Linear}, dtype=torch.qint8)

acc_fp32, c1, t1 = evaluate_accuracy(mnist_model, test_loader)
acc_dyn, c2, t2 = evaluate_accuracy(mnist_dynamic, test_loader)

print(f"[Part 4] FP32 정확도:    {acc_fp32:.2f}% ({c1}/{t1})")
print(f"[Part 4] 양자화 정확도:  {acc_dyn:.2f}% ({c2}/{t2})")
print(f"[Part 4] 정확도 차이:    {acc_fp32 - acc_dyn:.2f}%p")

# Part 5
import pandas as pd

def get_file_size_kb(path):
    return os.path.getsize(path) / 1024 if os.path.exists(path) else None

rows = [
    {"방법": "PyTorch FP32",      "추론시간(ms)": round(mean_pt, 2),   "예측클래스": int(pred_pytorch), "용량(KB)": get_file_size_kb("final_resnet18_fp32.pth")},
    {"방법": "PyTorch Dynamic",   "추론시간(ms)": round(mean_dyn, 2),  "예측클래스": "-",               "용량(KB)": get_file_size_kb("final_resnet18_dynamic.pth")},
    {"방법": "ONNX Runtime FP32", "추론시간(ms)": round(mean_onnx, 2), "예측클래스": "-",               "용량(KB)": get_file_size_kb("resnet18.onnx")},
    {"방법": "ONNX Runtime INT8", "추론시간(ms)": round(mean_int8, 2), "예측클래스": "-",               "용량(KB)": get_file_size_kb("resnet18_int8_static.onnx")},
]

df = pd.DataFrame(rows)
print(df.to_string(index=False))

# 결과를 파일로도 저장 (호스트에서 바로 확인 가능)
df.to_csv("final_comparison_result.csv", index=False)
print("\n결과가 final_comparison_result.csv로 저장되었습니다.")

# 출력 결과
# root@fec2ee4b9389:/workspace# python final_pipeline.py
# Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
# 100%|█████████████████████████████████████████████████| 44.7M/44.7M [00:05<00:00, 9.03MB/s]
# [Part 1] resnet18.onnx 변환 완료
# [Part 1] PyTorch 예측 클래스: 285
# [Part 2-1] Dynamic Quantization 완료
# WARNING:root:Please consider to run pre-processing before quantization. Refer to example: https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
# WARNING:root:Please consider pre-processing before quantization. See https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md 
# [Part 2-2] Static Quantization 완료
# [Part 2-3] 모델 저장 완료
# [Part 3] PyTorch FP32      : 58.391ms (±12.234)
# [Part 3] PyTorch Dynamic   : 54.831ms (±7.805)
# [Part 3] ONNX Runtime FP32 : 39.430ms (±13.917)
# [Part 3] ONNX Runtime INT8 : 43.718ms (±17.420)
# Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# Failed to download (trying next):
# HTTP Error 404: Not Found

# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to ./data/MNIST/raw/train-images-idx3-ubyte.gz
# 100%|███████████████████████████████████████| 9912422/9912422 [00:05<00:00, 1679205.99it/s]
# Extracting ./data/MNIST/raw/train-images-idx3-ubyte.gz to ./data/MNIST/raw

# Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
# Failed to download (trying next):
# HTTP Error 404: Not Found

# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to ./data/MNIST/raw/train-labels-idx1-ubyte.gz
# 100%|████████████████████████████████████████████| 28881/28881 [00:00<00:00, 150349.32it/s]
# Extracting ./data/MNIST/raw/train-labels-idx1-ubyte.gz to ./data/MNIST/raw

# Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
# Failed to download (trying next):
# HTTP Error 404: Not Found

# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw/t10k-images-idx3-ubyte.gz
# 100%|███████████████████████████████████████| 1648877/1648877 [00:01<00:00, 1576058.17it/s]
# Extracting ./data/MNIST/raw/t10k-images-idx3-ubyte.gz to ./data/MNIST/raw

# Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
# Failed to download (trying next):
# HTTP Error 404: Not Found

# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
# Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz
# 100%|█████████████████████████████████████████████| 4542/4542 [00:00<00:00, 1695792.13it/s]
# Extracting ./data/MNIST/raw/t10k-labels-idx1-ubyte.gz to ./data/MNIST/raw

# [Part 4] FP32 정확도:    10.00% (1000/10000)
# [Part 4] 양자화 정확도:  10.00% (1000/10000)
# [Part 4] 정확도 차이:    0.00%p
#                방법  추론시간(ms) 예측클래스       용량(KB)
#      PyTorch FP32     58.39   285 45738.912109
#   PyTorch Dynamic     54.83     - 44253.218750
# ONNX Runtime FP32     39.43     - 45652.890625
# ONNX Runtime INT8     43.72     - 11471.514648

# 결과가 final_comparison_result.csv로 저장되었습니다.