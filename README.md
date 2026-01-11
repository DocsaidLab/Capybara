**[English](./README.md)** | [Chinese](./README_tw.md)

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

## Introduction

Capybara is designed with three goals:

1. **Lightweight default install**: `pip install capybara-docsaid` installs only the core `utils/structures/vision` modules, without forcing heavy inference dependencies.
2. **Inference backends as opt-in extras**: install ONNX Runtime / OpenVINO / TorchScript only when you need them via extras.
3. **Lower risk**: enforce quality gates with ruff/pyright/pytest and target **90%** line coverage for the core codebase.

What you get:

- **Image tools** (`capybara.vision`): I/O, color conversion, resize/rotate/pad/crop, and video frame extraction.
- **Geometry structures** (`capybara.structures`): `Box/Boxes`, `Polygon/Polygons`, `Keypoints`, plus helper functions like IoU.
- **Inference wrappers (optional)**: `capybara.onnxengine` / `capybara.openvinoengine` / `capybara.torchengine`.
- **Feature extras (optional)**: `visualization` (drawing tools), `ipcam` (simple web demo), `system` (system info tools).
- **Utilities** (`capybara.utils`): `PowerDict`, `Timer`, `make_batch`, `download_from_google`, and other common helpers.

## Quick Start

### Install and verify

```bash
pip install capybara-docsaid
python -c "import capybara; print(capybara.__version__)"
```

## Documentation

To learn more about installation and usage, see [**Capybara Documents**](https://docsaid.org/docs/capybara).

The documentation includes detailed guides and common FAQs for this project.

## Installation

### Core install (lightweight)

```bash
pip install capybara-docsaid
```

### Enable inference backends (optional)

```bash
# ONNX Runtime (CPU)
pip install "capybara-docsaid[onnxruntime]"

# ONNX Runtime (GPU)
pip install "capybara-docsaid[onnxruntime-gpu]"

# OpenVINO runtime
pip install "capybara-docsaid[openvino]"

# TorchScript runtime
pip install "capybara-docsaid[torchscript]"

# Install everything
pip install "capybara-docsaid[all]"
```

### Feature extras (optional)

```bash
# Visualization (matplotlib/pillow)
pip install "capybara-docsaid[visualization]"

# IPCam app (flask)
pip install "capybara-docsaid[ipcam]"

# System info (psutil)
pip install "capybara-docsaid[system]"
```

### Combine multiple extras

If you want OpenVINO inference and the IPCam features, install:

```bash
# OpenVINO + IPCam
pip install "capybara-docsaid[openvino,ipcam]"
```

### Install from Git

```bash
pip install git+https://github.com/DocsaidLab/Capybara.git
```

## System Dependencies (Install as needed)

Some features require OS-level codecs / image I/O / PDF tools (install as needed):

- `PyTurboJPEG` (faster JPEG I/O): requires the TurboJPEG library.
- `pillow-heif` (HEIC/HEIF support): requires libheif.
- `pdf2image` (PDF to images): requires Poppler.
- Video frame extraction: installing `ffmpeg` is recommended (more stable OpenCV video decoding).

### Ubuntu

```bash
sudo apt install ffmpeg libturbojpeg libheif-dev poppler-utils
```

### macOS

```bash
brew install jpeg-turbo ffmpeg libheif poppler
```

### GPU Notes (ONNX Runtime CUDA)

If you're using `onnxruntime-gpu`, install the compatible CUDA/cuDNN version for your ORT version:

- See [**the ONNX Runtime documentation**](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)

## Usage

### Image data conventions

- Capybara images are represented as `numpy.ndarray`. By default, they follow OpenCV conventions: **BGR**, and shape is typically `(H, W, 3)`.
- If you prefer working in RGB, use `imread(..., color_base="RGB")` or convert with `imcvtcolor(img, "BGR2RGB")`.

### Image I/O

```python
from capybara import imread, imwrite

img = imread("your_image.jpg")
if img is None:
    raise RuntimeError("Failed to read image.")

imwrite(img, "out.jpg")
```

Notes:

- `imread` returns `None` when it fails to decode an image (if the path doesn't exist, it raises `FileExistsError`).
- `imread` also supports `.heic` (requires `pillow-heif` + OS-level libheif).

### Resize / pad

With `imresize`, you can pass `None` in `size` to keep the aspect ratio and have the other dimension inferred automatically.

```python
import numpy as np
from capybara import BORDER, imresize, pad

img = np.zeros((480, 640, 3), dtype=np.uint8)
img = imresize(img, (320, None))  # (height, width)
img = pad(img, pad_size=(8, 8), pad_mode=BORDER.REPLICATE)
```

### Color conversion

```python
import numpy as np
from capybara import imcvtcolor

img = np.zeros((240, 320, 3), dtype=np.uint8)  # BGR
gray = imcvtcolor(img, "BGR2GRAY")             # grayscale
rgb = imcvtcolor(img, "BGR2RGB")               # RGB
```

### Rotation / perspective correction

```python
import numpy as np
from capybara import Polygon, imrotate, imwarp_quadrangle

img = np.zeros((240, 320, 3), dtype=np.uint8)
rot = imrotate(img, angle=15, expand=True)  # Angle definition matches OpenCV: positive values rotate counterclockwise

poly = Polygon([[10, 10], [200, 20], [190, 120], [20, 110]])
patch = imwarp_quadrangle(img, poly)        # 4-point perspective warp
```

### Cropping (Box / Boxes)

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

### Binarization + morphology

Morphology operators live in `capybara.vision.morphology` (not in the top-level `capybara` namespace).

```python
import numpy as np
from capybara import imbinarize
from capybara.vision.morphology import imopen

img = np.zeros((240, 320, 3), dtype=np.uint8)
mask = imbinarize(img)        # OTSU + binary
mask = imopen(mask, ksize=3)  # Opening to remove small noise
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

### Polygons / IoU

```python
from capybara import Polygon, polygon_iou

p1 = Polygon([[0, 0], [10, 0], [10, 10], [0, 10]])
p2 = Polygon([[5, 5], [15, 5], [15, 15], [5, 15]])
print(polygon_iou(p1, p2))
```

### Base64 (image / ndarray)

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

### PDF to images

```python
from capybara.vision.improc import pdf2imgs

pages = pdf2imgs("file.pdf")  # list[np.ndarray], each page is BGR image
if pages is None:
    raise RuntimeError("Failed to decode PDF.")
print(len(pages))
```

### Visualization (optional)

Install first: `pip install "capybara-docsaid[visualization]"`.

```python
import numpy as np
from capybara import Box
from capybara.vision.visualization.draw import draw_box

img = np.zeros((240, 320, 3), dtype=np.uint8)
img = draw_box(img, Box([10, 20, 100, 120]))
```

### IPCam (optional)

`IpcamCapture` itself does not depend on Flask; you only need the `ipcam` extra to use `WebDemo`.

```python
from capybara.vision.ipcam.camera import IpcamCapture

cap = IpcamCapture(url=0, color_base="BGR")  # or provide an RTSP/HTTP URL
frame = next(cap)
```

Web demo (install first: `pip install "capybara-docsaid[ipcam]"`):

```python
from capybara.vision.ipcam.app import WebDemo

WebDemo("rtsp://<ipcam-url>").run(port=5001)
```

### System info (optional)

Install first: `pip install "capybara-docsaid[system]"`.

```python
from capybara.utils.system_info import get_system_info

print(get_system_info())
```

### Video frame extraction

```python
from capybara import video2frames_v2

frames = video2frames_v2("demo.mp4", frame_per_sec=2, max_size=1280)
print(len(frames))
```

## Inference Backends

Inference backends are optional; install the corresponding extras before importing the relevant engine modules.

### Runtime / backend matrix

Note: TorchScript runtime is named `Runtime.pt` in code (corresponding extra: `torchscript`).

| Runtime (`capybara.runtime.Runtime`) | Backend name    | Provider / device                                                                                           |
| ------------------------------------ | --------------- | ----------------------------------------------------------------------------------------------------------- |
| `onnx`                               | `cpu`           | `["CPUExecutionProvider"]`                                                                                  |
| `onnx`                               | `cuda`          | `["CUDAExecutionProvider"(device_id), "CPUExecutionProvider"]`                                              |
| `onnx`                               | `tensorrt`      | `["TensorrtExecutionProvider"(device_id), "CUDAExecutionProvider"(device_id), "CPUExecutionProvider"]`      |
| `onnx`                               | `tensorrt_rtx`  | `["NvTensorRTRTXExecutionProvider"(device_id), "CUDAExecutionProvider"(device_id), "CPUExecutionProvider"]` |
| `openvino`                           | `cpu`           | `device="CPU"`                                                                                              |
| `openvino`                           | `gpu`           | `device="GPU"`                                                                                              |
| `openvino`                           | `npu`           | `device="NPU"`                                                                                              |
| `pt`                                 | `cpu`           | `torch.device("cpu")`                                                                                       |
| `pt`                                 | `cuda`          | `torch.device("cuda")`                                                                                      |

### Runtime registry (auto backend selection)

```python
from capybara.runtime import Runtime

print(Runtime.onnx.auto_backend_name())      # Priority: cuda -> tensorrt_rtx -> tensorrt -> cpu
print(Runtime.openvino.auto_backend_name())  # Priority: gpu -> npu -> cpu
print(Runtime.pt.auto_backend_name())        # Priority: cuda -> cpu
```

### ONNX Runtime (`capybara.onnxengine`)

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

### OpenVINO (`capybara.openvinoengine`)

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

### TorchScript (`capybara.torchengine`)

```python
import numpy as np
from capybara.torchengine import TorchEngine

engine = TorchEngine("model.pt", device="cpu")
outputs = engine.run({"image": np.zeros((1, 3, 224, 224), dtype=np.float32)})
print(outputs.keys())
```

### Benchmark (depends on hardware)

All engines provide `benchmark(...)` for quick throughput/latency measurements.

```python
import numpy as np
from capybara.onnxengine import ONNXEngine

engine = ONNXEngine("model.onnx", backend="cpu")
dummy = np.zeros((1, 3, 224, 224), dtype=np.float32)
print(engine.benchmark({"input": dummy}, repeat=50, warmup=5))
```

### Advanced: Custom options (optional)

`EngineConfig` / `OpenVINOConfig` / `TorchEngineConfig` are passed through to the underlying runtime as-is.

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

## Quality Gates (Contributors)

Before merging, this project requires:

```bash
ruff check .
ruff format --check .
pyright
python -m pytest --cov=capybara --cov-config=.coveragerc --cov-report=term
```

Notes:

- Coverage gate is **90% line coverage** (rules defined in `.coveragerc`).
- Heavy / environment-dependent modules are excluded from the default coverage gate to keep CI reproducible and maintainable.

## Docker (optional)

```bash
git clone https://github.com/DocsaidLab/Capybara.git
cd Capybara
bash docker/build.bash
```

Run:

```bash
docker run --rm -it capybara_docsaid bash
```

If you need GPU access inside the container, use the NVIDIA container runtime (e.g. `--gpus all`).

## Testing (local)

```bash
python -m pytest -vv
```

## License

Apache-2.0, see `LICENSE`.

## Citation

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
