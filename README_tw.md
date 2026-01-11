[English](./README.md) | **[中文](./README_tw.md)**

# Capybara

<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href="https://github.com/DocsaidLab/Capybara/releases"><img src="https://img.shields.io/github/v/release/DocsaidLab/Capybara?color=ffa"></a>
    <a href="https://pypi.org/project/capybara-docsaid/"><img src="https://img.shields.io/pypi/v/capybara-docsaid.svg"></a>
    <a href="https://pypi.org/project/capybara-docsaid/"><img src="https://img.shields.io/pypi/dm/capybara-docsaid?color=9cf"></a>
</p>

![title](https://raw.githubusercontent.com/DocsaidLab/Capybara/refs/heads/main/docs/title.webp)

---

## 介紹

Capybara 的設計目標聚焦三個方向：

1. **預設安裝輕量化**：`pip install capybara-docsaid` 僅安裝核心 utils/structures/vision，不強迫安裝重型推論依賴。
2. **推論後端改為 opt-in extras**：需要 ONNX Runtime / OpenVINO / TorchScript 時再用 extras 安裝。
3. **降低風險**：導入 ruff/pyright/pytest 品質門檻，並以核心程式碼 **90%** 行覆蓋率為維護目標。

你會得到：

- **影像工具**（`capybara.vision`）：讀寫、轉色、縮放/旋轉/補邊/裁切，以及影片抽幀工具。
- **幾何結構**（`capybara.structures`）：`Box/Boxes`、`Polygon/Polygons`、`Keypoints`，以及 IoU 等輔助函數。
- **推論封裝（可選）**：`capybara.onnxengine` / `capybara.openvinoengine` / `capybara.torchengine`。
- **功能 extras（可選）**：`visualization`（繪圖工具）、`ipcam`（簡易 Web demo）、`system`（系統資訊工具）。
- **小工具**（`capybara.utils`）：`PowerDict`、`Timer`、`make_batch`、`download_from_google` 等常用 helper。

## 快速開始

### 安裝與驗證

```bash
pip install capybara-docsaid
python -c "import capybara; print(capybara.__version__)"
```

## 技術文件

若想進一步瞭解安裝與使用方式，請參閱 [**Capybara Documents**](https://docsaid.org/docs/capybara)。

該文件提供本專案的詳細說明與常見問題解答。

## 安裝

### 核心安裝（輕量）

```bash
pip install capybara-docsaid
```

### 啟用推論後端（可選）

```bash
# ONNXRuntime（CPU）
pip install "capybara-docsaid[onnxruntime]"

# ONNXRuntime（GPU）
pip install "capybara-docsaid[onnxruntime-gpu]"

# OpenVINO runtime
pip install "capybara-docsaid[openvino]"

# TorchScript runtime
pip install "capybara-docsaid[torchscript]"

# 全部一起裝
pip install "capybara-docsaid[all]"
```

### 選用功能 extras（可選）

```bash
# 視覺化（matplotlib/pillow）
pip install "capybara-docsaid[visualization]"

# IPCam app（flask）
pip install "capybara-docsaid[ipcam]"

# 系統資訊（psutil）
pip install "capybara-docsaid[system]"
```

### 挑選多個功能

假設你想使用 openvino 推論，並搭配 ipcam 相關的功能，可以這樣安裝：

```bash
# 選用 OpenVINO 和 IPCam
pip install "capybara-docsaid[openvino,ipcam]"
```

### 從 Git 安裝

```bash
pip install git+https://github.com/DocsaidLab/Capybara.git
```

## 系統相依套件（依功能需求安裝）

有些功能需要 OS 層級的 codec / image IO / PDF 工具（依功能需求安裝）：

- `PyTurboJPEG`（JPEG 讀寫加速）：需要 TurboJPEG library。
- `pillow-heif`（HEIC/HEIF 支援）：需要 libheif。
- `pdf2image`（PDF 轉圖）：需要 Poppler。
- 影片抽幀：建議安裝 `ffmpeg`（讓 OpenCV 影片讀取更穩定）。

### Ubuntu

```bash
sudo apt install ffmpeg libturbojpeg libheif-dev poppler-utils
```

### macOS

```bash
brew install jpeg-turbo ffmpeg libheif poppler
```

### GPU 注意事項（ONNXRuntime CUDA）

若使用 `onnxruntime-gpu`，請依 ORT 的版本安裝相容的 CUDA/cuDNN：

- 請參考 [**onnxruntime 官方網站**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

## 使用方式

### 影像資料格式約定

- Capybara 的影像以 `numpy.ndarray` 表示，預設遵循 OpenCV 慣例：**BGR**、shape 通常為 `(H, W, 3)`。
- 若你希望以 RGB 工作，可用 `imread(..., color_base="RGB")` 或 `imcvtcolor(img, "BGR2RGB")` 轉換。

### 影像 I/O

```python
from capybara import imread, imwrite

img = imread("your_image.jpg")
if img is None:
    raise RuntimeError("Failed to read image.")

imwrite(img, "out.jpg")
```

補充：

- `imread` 讀不到圖時會回傳 `None`（路徑不存在則會直接丟 `FileExistsError`）。
- `imread` 也支援 `.heic`（需 `pillow-heif` + OS 層級 libheif）。

### Resize / pad

`imresize` 支援在 `size` 中用 `None` 表示「維持長寬比自動推算另一邊」。

```python
import numpy as np
from capybara import BORDER, imresize, pad

img = np.zeros((480, 640, 3), dtype=np.uint8)
img = imresize(img, (320, None))  # (height, width)
img = pad(img, pad_size=(8, 8), pad_mode=BORDER.REPLICATE)
```

### 轉色（Color Conversion）

```python
import numpy as np
from capybara import imcvtcolor

img = np.zeros((240, 320, 3), dtype=np.uint8)  # BGR
gray = imcvtcolor(img, "BGR2GRAY")             # grayscale
rgb = imcvtcolor(img, "BGR2RGB")               # RGB
```

### 旋轉 / 透視校正

```python
import numpy as np
from capybara import Polygon, imrotate, imwarp_quadrangle

img = np.zeros((240, 320, 3), dtype=np.uint8)
rot = imrotate(img, angle=15, expand=True)  # 角度定義與 OpenCV 相同：正值為逆時針

poly = Polygon([[10, 10], [200, 20], [190, 120], [20, 110]])
patch = imwarp_quadrangle(img, poly)        # 4 點透視校正
```

### 裁切（Box / Boxes）

```python
import numpy as np
from capybara import Box, Boxes, imcropbox, imcropboxes

img = np.zeros((240, 320, 3), dtype=np.uint8)
crop1 = imcropbox(img, Box([10, 20, 110, 120]), use_pad=True)
crop_list = imcropboxes(
    img,
    Boxes([[0, 0, 10, 10], [100, 100, 400, 300]]),
    use_pad=True,
)
```

### 二值化 + 形態學（Morphology）

形態學操作位於 `capybara.vision.morphology`（不在頂層 `capybara` namespace）。

```python
import numpy as np
from capybara import imbinarize
from capybara.vision.morphology import imopen

img = np.zeros((240, 320, 3), dtype=np.uint8)
mask = imbinarize(img)        # OTSU + binary
mask = imopen(mask, ksize=3)  # 開運算去除雜點
```

### Boxes / IoU

```python
import numpy as np
from capybara import Box, Boxes, pairwise_iou

boxes_a = Boxes([[10, 10, 20, 20], [30, 30, 60, 60]])
boxes_b = Boxes(np.array([[12, 12, 18, 18]], dtype=np.float32))
print(pairwise_iou(boxes_a, boxes_b))

box = Box([0.1, 0.2, 0.9, 0.8], is_normalized=True).convert("XYWH")
print(box.numpy())
```

### Polygons（多邊形）/ IoU

```python
from capybara import Polygon, polygon_iou

p1 = Polygon([[0, 0], [10, 0], [10, 10], [0, 10]])
p2 = Polygon([[5, 5], [15, 5], [15, 15], [5, 15]])
print(polygon_iou(p1, p2))
```

### Base64（影像 / ndarray）

```python
import numpy as np
from capybara import img_to_b64str, npy_to_b64str
from capybara.vision.improc import b64str_to_img, b64str_to_npy

img = np.zeros((32, 32, 3), dtype=np.uint8)
b64_img = img_to_b64str(img)          # JPEG bytes -> base64 string
if b64_img is None:
    raise RuntimeError("Failed to encode image into base64.")
img2 = b64str_to_img(b64_img)         # base64 string -> numpy image

vec = np.arange(8, dtype=np.float32)
b64_vec = npy_to_b64str(vec)
vec2 = b64str_to_npy(b64_vec, dtype="float32")
```

### PDF 轉影像

```python
from capybara.vision.improc import pdf2imgs

pages = pdf2imgs("file.pdf")  # list[np.ndarray], each page is BGR image
if pages is None:
    raise RuntimeError("Failed to decode PDF.")
print(len(pages))
```

### 視覺化（可選）

需要先安裝：`pip install "capybara-docsaid[visualization]"`。

```python
import numpy as np
from capybara import Box
from capybara.vision.visualization.draw import draw_box

img = np.zeros((240, 320, 3), dtype=np.uint8)
img = draw_box(img, Box([10, 20, 100, 120]))
```

### IPCam（可選）

`IpcamCapture` 本身不依賴 Flask；若要使用 `WebDemo` 才需要安裝 `ipcam` extra。

```python
from capybara.vision.ipcam.camera import IpcamCapture

cap = IpcamCapture(url=0, color_base="BGR")  # 或填入 RTSP/HTTP URL
frame = next(cap)
```

Web demo（需要先安裝：`pip install "capybara-docsaid[ipcam]"`）：

```python
from capybara.vision.ipcam.app import WebDemo

WebDemo("rtsp://<ipcam-url>").run(port=5001)
```

### 系統資訊（可選）

需要先安裝：`pip install "capybara-docsaid[system]"`。

```python
from capybara.utils.system_info import get_system_info

print(get_system_info())
```

### 影片抽幀

```python
from capybara import video2frames_v2

frames = video2frames_v2("demo.mp4", frame_per_sec=2, max_size=1280)
print(len(frames))
```

## 推論後端（Inference Backends）

推論後端為可選功能；請先用 extras 安裝後再 import 對應 engine 模組。

### Runtime / Backend 搭配表

注意：TorchScript runtime 在程式內以 `Runtime.pt` 命名（對應安裝 extra：`torchscript`）。

| Runtime (`capybara.runtime.Runtime`) | Backend 名稱   | Provider / device                                                                                           |
| ------------------------------------ | -------------- | ----------------------------------------------------------------------------------------------------------- |
| `onnx`                               | `cpu`          | `["CPUExecutionProvider"]`                                                                                  |
| `onnx`                               | `cuda`         | `["CUDAExecutionProvider"(device_id), "CPUExecutionProvider"]`                                              |
| `onnx`                               | `tensorrt`     | `["TensorrtExecutionProvider"(device_id), "CUDAExecutionProvider"(device_id), "CPUExecutionProvider"]`      |
| `onnx`                               | `tensorrt_rtx` | `["NvTensorRTRTXExecutionProvider"(device_id), "CUDAExecutionProvider"(device_id), "CPUExecutionProvider"]` |
| `openvino`                           | `cpu`          | `device="CPU"`                                                                                              |
| `openvino`                           | `gpu`          | `device="GPU"`                                                                                              |
| `openvino`                           | `npu`          | `device="NPU"`                                                                                              |
| `pt`                                 | `cpu`          | `torch.device("cpu")`                                                                                       |
| `pt`                                 | `cuda`         | `torch.device("cuda")`                                                                                      |

### Runtime registry（auto 後端選擇）

```python
from capybara.runtime import Runtime

print(Runtime.onnx.auto_backend_name())      # 優先順序：cuda -> tensorrt_rtx -> tensorrt -> cpu
print(Runtime.openvino.auto_backend_name())  # 優先順序：gpu -> npu -> cpu
print(Runtime.pt.auto_backend_name())        # 優先順序：cuda -> cpu
```

### ONNX Runtime（`capybara.onnxengine`）

```python
import numpy as np
from capybara.onnxengine import EngineConfig, ONNXEngine

engine = ONNXEngine(
    "model.onnx",
    backend="cpu",
    config=EngineConfig(enable_io_binding=False),
)
outputs = engine.run({"input": np.ones((1, 3, 224, 224), dtype=np.float32)})
print(outputs.keys())
print(engine.summary())
```

### OpenVINO（`capybara.openvinoengine`）

```python
import numpy as np
from capybara.openvinoengine import OpenVINOConfig, OpenVINODevice, OpenVINOEngine

engine = OpenVINOEngine(
    "model.xml",
    device=OpenVINODevice.cpu,
    config=OpenVINOConfig(num_requests=2),
)
outputs = engine.run({"input": np.ones((1, 3), dtype=np.float32)})
print(outputs.keys())
```

### TorchScript（`capybara.torchengine`）

```python
import numpy as np
from capybara.torchengine import TorchEngine

engine = TorchEngine("model.pt", device="cpu")
outputs = engine.run({"image": np.zeros((1, 3, 224, 224), dtype=np.float32)})
print(outputs.keys())
```

### Benchmark（依硬體而異）

所有 engines 都提供 `benchmark(...)`，用於快速量測吞吐/延遲。

```python
import numpy as np
from capybara.onnxengine import ONNXEngine

engine = ONNXEngine("model.onnx", backend="cpu")
dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
print(engine.benchmark({"input": dummy}, repeat=50, warmup=5))
```

### 進階：自訂參數（可選）

`EngineConfig` / `OpenVINOConfig` / `TorchEngineConfig` 會原樣傳遞到底層 runtime。

```python
from capybara.onnxengine import EngineConfig, ONNXEngine

engine = ONNXEngine(
    "model.onnx",
    backend="cuda",
    config=EngineConfig(
        provider_options={
            "CUDAExecutionProvider": {
                "enable_cuda_graph": True,
            },
        },
    ),
)
```

## 品質門檻（Quality Gates / 開發者）

本專案在合併前會強制通過：

```bash
ruff check .
ruff format --check .
pyright
python -m pytest --cov=capybara --cov-config=.coveragerc --cov-report=term
```

備註：

- 覆蓋率門檻為 **90% 覆蓋率**（規則定義於 `.coveragerc`）。
- 重型/環境相依模組不納入預設 coverage gate，以維持 CI 可重現與可維護。

## Docker（可選）

```bash
git clone https://github.com/DocsaidLab/Capybara.git
cd Capybara
bash docker/build.bash
```

執行：

```bash
docker run --rm -it capybara_docsaid bash
```

若你需要在容器內使用 GPU，請使用 NVIDIA container runtime（例如 `--gpus all`）。

## 測試（本地）

```bash
python -m pytest -vv
```

## 授權

Apache-2.0，見 `LICENSE`。

## 引用

```bibtex
@misc{lin2025capybara,
  author       = {Kun-Hsiang Lin*, Ze Yuan*},
  title        = {Capybara: An Integrated Python Package for Image Processing and Deep Learning.},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\\url{https://github.com/DocsaidLab/Capybara}},
  note         = {* equal contribution}
}
```
