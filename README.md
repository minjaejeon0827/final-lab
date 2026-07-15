# 캡스톤 실습

# 캡스톤 실습: 딥러닝 모델 변환과 양자화 전체 파이프라인

## 실습 목표

이번 실습은 Chapter 1(문제 인식)부터 Chapter 6(측정 방법론)까지 배운 내용 중, **GPU 없이도 확인 가능한 모든 개념**을 하나로 엮습니다. Chapter 1~2에서 만들었던 Docker 환경(`pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime`)을 그대로 재사용해서, OS와 상관없이 동일한 조건에서 진행합니다.

```
포함되는 것                            제외되는 것
─────────────────────────────────   ─────────────────
✓ PyTorch → ONNX 변환 (Ch.2)         ✗ TensorRT 엔진 빌드 (Ch.3, GPU 필요)
✓ ONNX Runtime CPU 추론 (Ch.2)       ✗ GPU 추론 비교 (Ch.2 선택절)
✓ Dynamic/Static PTQ (Ch.4)         ✗ TensorRT INT8 (Ch.5, GPU 필요)
✓ 워밍업 + 반복 측정 (Ch.6)
✓ 데이터셋 전체 정확도 검증 (Ch.7)
✓ 모델 저장 및 용량 비교 (Ch.4)
```

**최종 산출물**: 하나의 스크립트로 "PyTorch → ONNX 변환 → PTQ 양자화 → 속도/정확도/용량 3중 비교표"까지 전부 완성합니다.

---

## 준비: Docker 컨테이너 실행

### Step 1: 실습 폴더 준비

Chapter 2에서 배운 대로, 바탕화면에 실습 폴더를 만듭니다.

**Windows (WSL2 Ubuntu 터미널)**

```bash
mkdir -p /mnt/c/Users/$(whoami)/Desktop/final-lab
cd /mnt/c/Users/$(whoami)/Desktop/final-lab
mkdir -p image
# 강아지, 고양이 등 자유롭게 이미지 1장을 image/test.jpg로 저장
```

**macOS / Linux**

```bash
mkdir -p ~/Desktop/final-lab
cd ~/Desktop/final-lab
mkdir -p image
```

### Step 2: 컨테이너 실행

이번 실습은 GPU가 필요 없으므로, `--gpus all` 옵션 없이 컨테이너를 실행합니다.

```bash
docker run -dit \
  --name final-lab-container \
  -v $(pwd):/workspace \
  pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime bash
```

> 💡 GPU가 있는 컴퓨터라도, 이번 실습은 CPU 전용이므로 `--gpus all`을 굳이 붙일 필요가 없습니다. Chapter 3~5의 TensorRT/GPU 실습과 이 컨테이너는 별개로 관리하는 것을 권장합니다.
> 

### Step 3: 필요한 패키지 설치

```bash
pip install onnx onnxruntime pillow pandas
```

```bash
python -c "import onnx, onnxruntime, pandas; print('Successfully installed!')"
```

```
Successfully installed!
```

이제부터 나오는 모든 코드는 이 컨테이너 안에서 실행합니다.

---

## Part 1: PyTorch → ONNX 변환 (Chapter 2 복습)

test.jpg

컨테이너 안에서 `final_pipeline.py` 파일을 만듭니다.

```python
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
```

```bash
python final_pipeline.py
```

```
[Part 1] resnet18.onnx 변환 완료
[Part 1] PyTorch 예측 클래스: 259
```

> 💡 실행 결과 `resnet18.onnx`는 `/workspace`에 저장되므로, Bind Mount 덕분에 호스트의 `final-lab` 폴더에도 그대로 남습니다. Chapter 2에서 배운 `dynamic_axes`, `opset_version` 개념이 여기서 그대로 재사용되고 있습니다.
> 

---

## Part 2: 양자화 3종 세트 만들기 (Chapter 4 복습)

같은 `final_pipeline.py` 파일에 이어서 작성합니다.

### Part 2-1: Dynamic Quantization

```python
# Part 2-1
import torch.quantization as tq

model_dynamic = tq.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)
print("[Part 2-1] Dynamic Quantization 완료")
```

### Part 2-2: Static Quantization (ONNX Runtime)

```python
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
```

> 💡 `image` 폴더는 `/workspace/image`, 즉 호스트의 `final-lab/image` 폴더와 연결되어 있으므로 여기 저장한 `test.jpg`를 그대로 캘리브레이션 데이터로 씁니다.
> 

### Part 2-3: 세 모델 파일로 저장하기

```python
# Part 2-3
torch.save(model_fp32.state_dict(), "final_resnet18_fp32.pth")
torch.save(model_dynamic, "final_resnet18_dynamic.pth")
print("[Part 2-3] 모델 저장 완료")
```

```bash
python final_pipeline.py
```

```
[Part 2-1] Dynamic Quantization 완료
[Part 2-2] Static Quantization 완료
[Part 2-3] 모델 저장 완료
```

```bash
# 컨테이너 안에서 파일이 잘 생성됐는지 확인
ls -lh *.onnx *.pth
```

> 💡 Chapter 4에서 배운 대로, 양자화 모델(`model_dynamic`)은 `state_dict()`가 아니라 모델 객체 전체를 저장했습니다.
> 

---

## Part 3: 워밍업 + 반복 측정으로 속도 비교 (Chapter 6 복습)

같은 파일에 이어서 작성합니다.

```python
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
```

```bash
python final_pipeline.py
```

```
[Part 3] PyTorch FP32      : 13.842ms (±0.891)
[Part 3] PyTorch Dynamic   : 13.685ms (±0.823)
[Part 3] ONNX Runtime FP32 : 8.213ms (±0.312)
[Part 3] ONNX Runtime INT8 : 11.204ms (±0.658)
```

> 💡 Chapter 4에서 배운 대로, ONNX INT8이 FP32보다 느리게 나올 수 있습니다. 이건 실패가 아니라 "CPU 환경에서 quant/dequant 비용이 이득보다 클 수 있다"는 정상적인 결과입니다.
> 

---

## Part 4: 데이터셋 전체로 정확도 검증 (Chapter 7 복습)

같은 파일에 이어서 작성합니다. MNIST 데이터셋을 처음 다운로드할 때 인터넷 연결이 필요하므로, 컨테이너가 외부 네트워크에 접근 가능한지 미리 확인해두면 좋습니다.

```python
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
```

```bash
python final_pipeline.py
```

```
[Part 4] FP32 정확도:    9.8% (980/10000)
[Part 4] 양자화 정확도:  9.6% (960/10000)
[Part 4] 정확도 차이:    0.2%p
```

> ⚠️ 사전 학습을 생략했기 때문에 정확도 자체는 낮게 나옵니다(랜덤 가중치이므로 10% 근처). **이 실습의 목표는 실제 정확도 수치가 아니라, "이미지 1장이 아니라 만 장 단위로 검증하는 절차"를 스스로 완성해보는 것**입니다.
> 
> 
> `torchvision.datasets.MNIST(..., download=True)`는 처음 실행할 때 인터넷에서 데이터를 받아옵니다. 컨테이너 안에서 다운로드가 안 된다면, 컨테이너에 외부 네트워크 접근이 막혀 있지 않은지 확인하세요(일반적으로 `docker run`에 별도 네트워크 제한 옵션을 주지 않았다면 문제없이 접근됩니다). 데이터는 `/workspace/data`(호스트의 `final-lab/data`)에 저장되므로, 컨테이너를 재시작해도 다시 받을 필요가 없습니다.
> 

---

## Part 5: 최종 비교표 만들기

같은 파일에 이어서 작성합니다.

```python
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
```

```bash
python final_pipeline.py
```

```
              방법  추론시간(ms) 예측클래스     용량(KB)
     PyTorch FP32       13.84      259    46837.2
  PyTorch Dynamic       13.69        -    44921.5
 ONNX Runtime FP32       8.21        -    46742.8
 ONNX Runtime INT8      11.20        -    11890.3

결과가 final_comparison_result.csv로 저장되었습니다.
```

> 💡 `final_comparison_result.csv`는 `/workspace`에 저장되므로, Bind Mount 덕분에 컨테이너를 나가서 호스트의 `final-lab` 폴더에서 바로 열어볼 수 있습니다. 여기서 ONNX INT8의 용량이 확실히 작아진 것에 주목하세요. Static Quantization은 **가중치 전체**를 INT8로 바꾸기 때문에, Linear 레이어에만 적용된 Dynamic Quantization(용량 거의 그대로)과 달리 용량 감소가 뚜렷하게 나타납니다.
> 

---

## 컨테이너 정리

```bash
exit
```

```bash
# 호스트에서 실행
docker stop final-lab-container
docker rm final-lab-container
```

결과 파일(`resnet18.onnx`, `resnet18_int8_static.onnx`, `.pth` 파일들, `final_comparison_result.csv`)은 모두 호스트의 `final-lab` 폴더에 남아있으므로 컨테이너를 지워도 안전합니다.

---

## 결과 공유 및 정리

### 토의 질문

```
1. 속도 순위(빠른 순)와 용량 순위(작은 순)가 서로 일치했나요, 달랐나요?
   → 왜 그런 차이가 생겼는지 Chapter 3~4 개념으로 설명해보세요.

2. ONNX Runtime INT8이 FP32보다 느렸다면, 그 이유를 옆 사람에게 설명해보세요.
   (Chapter 4.4절의 네 가지 조건 중 어떤 것이 해당하나요?)

3. Dynamic Quantization의 용량이 거의 줄지 않은 이유는 무엇인가요?
   (Chapter 4.5절 내용과 연결)

4. 이번 실습에서 Docker를 사용한 것이, 만약 팀원이 다른 OS(Windows/macOS/Linux)를
   쓰고 있었다면 어떤 도움이 됐을지 이야기해보세요.
```

### 전체 체크리스트

- [ ]  컨테이너 안에서 pip install이 정상적으로 됐나요?
- [ ]  Part 1: ONNX 변환 후 PyTorch와 예측 클래스가 일치했나요?
- [ ]  Part 2: Dynamic, Static 두 가지 양자화를 모두 완료했나요?
- [ ]  Part 3: 워밍업+반복 측정으로 4가지 방식의 속도를 비교했나요?
- [ ]  Part 4: 데이터셋 전체로 정확도 평가 절차를 완성했나요?
- [ ]  Part 5: 속도·용량·정확도를 하나의 표로 종합하고, CSV로 저장했나요?
- [ ]  컨테이너를 나간 뒤에도 호스트 폴더에서 결과 파일을 확인할 수 있었나요?

---

## 이 실습이 완성하는 것

이번 45분 실습은 GPU나 TensorRT 없이, **Docker 컨테이너 하나**로 이 파트에서 배운 핵심 흐름 전체(환경 준비(Ch.1) → 변환(Ch.2) → 양자화(Ch.4) → 올바른 측정(Ch.6) → 데이터셋 전체 검증(Ch.7))를 완성했습니다.

특히 이번 실습에서 다시 확인한 것은, Chapter 1에서 배운 "Docker로 환경을 통일한다"는 원칙이 실습 마지막까지 계속 유지되고 있었다는 점입니다. OS가 무엇이든, 이 컨테이너 안에서는 똑같은 결과가 재현됩니다. 그리고 TensorRT나 GPU 환경이 갖춰지면, 오늘 만든 `resnet18.onnx` 파일에 Chapter 3의 `trtexec` 한 줄만 추가하면 나머지 절차(FP16/INT8 엔진, GPU 가속)로 바로 확장할 수 있습니다.
