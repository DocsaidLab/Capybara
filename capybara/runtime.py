from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, ClassVar, cast

__all__ = ["Backend", "ProviderSpec", "Runtime"]


def _normalize_key(value: str | Runtime) -> str:
    if isinstance(value, Runtime):
        return value.name
    return str(value).strip().lower()


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    include_device: bool = False


@dataclass(frozen=True)
class Backend:
    name: str
    runtime_key: str
    providers: tuple[ProviderSpec, ...] = ()
    device: str | None = None
    description: str | None = None

    _REGISTRY: ClassVar[dict[str, dict[str, Backend]]] = {}
    cpu: ClassVar[Backend]
    cuda: ClassVar[Backend]
    tensorrt: ClassVar[Backend]
    tensorrt_rtx: ClassVar[Backend]
    ov_cpu: ClassVar[Backend]
    ov_gpu: ClassVar[Backend]
    ov_npu: ClassVar[Backend]

    def __post_init__(self) -> None:
        normalized_runtime = self.runtime_key.strip().lower()
        normalized_name = self.name.strip().lower()
        object.__setattr__(self, "runtime_key", normalized_runtime)
        object.__setattr__(self, "name", normalized_name)
        namespace = self._REGISTRY.setdefault(normalized_runtime, {})
        if normalized_name in namespace:
            raise ValueError(
                f"Backend '{self.name}' already registered for runtime '{self.runtime_key}'."
            )
        namespace[normalized_name] = self

    @property
    def runtime(self) -> str:
        return self.runtime_key

    @classmethod
    def available(cls, runtime: str | Runtime) -> tuple[Backend, ...]:
        runtime_key = _normalize_key(runtime)
        namespace = cls._REGISTRY.get(runtime_key, {})
        return tuple(namespace.values())

    @classmethod
    def from_any(
        cls,
        value: Any,
        *,
        runtime: str | Runtime | None = None,
    ) -> Backend:
        runtime_key = cls._resolve_runtime_key(runtime)
        namespace = cls._REGISTRY.get(runtime_key, {})
        if isinstance(value, Backend):
            if value.runtime_key != runtime_key:
                raise ValueError(
                    f"Backend '{value.name}' is not registered for runtime '{runtime_key}'."
                )
            return value
        normalized = str(value).strip().lower()
        try:
            return namespace[normalized]
        except KeyError:  # pragma: no cover - defensive guard
            options = ", ".join(namespace.keys()) or "<none>"
            raise ValueError(
                f"Unsupported backend '{value}' for runtime '{runtime_key}'. "
                f"Pick from [{options}]"
            ) from None

    @classmethod
    def _resolve_runtime_key(cls, runtime: str | Runtime | None) -> str:
        if runtime is None:
            if len(cls._REGISTRY) == 1:
                return next(iter(cls._REGISTRY))
            raise ValueError(
                "runtime must be specified when resolving backends."
            )
        if isinstance(runtime, Runtime):
            return runtime.name
        return str(runtime).strip().lower()


@dataclass(frozen=True)
class Runtime:
    name: str
    backend_names: tuple[str, ...]
    default_backend_name: str
    description: str | None = None

    _REGISTRY: ClassVar[dict[str, Runtime]] = {}
    onnx: ClassVar[Runtime]
    openvino: ClassVar[Runtime]
    pt: ClassVar[Runtime]

    def __post_init__(self) -> None:
        normalized = self.name.strip().lower()
        object.__setattr__(self, "name", normalized)
        normalized_backends = tuple(
            name.strip().lower() for name in self.backend_names
        )
        object.__setattr__(self, "backend_names", normalized_backends)
        default_backend = self.default_backend_name.strip().lower()
        object.__setattr__(self, "default_backend_name", default_backend)

        if normalized in self._REGISTRY:
            raise ValueError(f"Runtime '{self.name}' is already registered.")
        self._REGISTRY[normalized] = self

        available = {backend.name for backend in Backend.available(normalized)}
        requested = set(normalized_backends)
        missing = requested - available
        if missing:
            raise ValueError(
                f"Runtime '{self.name}' references unknown backend(s): {sorted(missing)}."
            )
        if default_backend not in requested:
            raise ValueError(
                f"Default backend '{self.default_backend_name}' is not tracked "
                f"for runtime '{self.name}'."
            )

    @classmethod
    def from_any(cls, value: Any) -> Runtime:
        if isinstance(value, Runtime):
            return value
        normalized = str(value).strip().lower()
        try:
            return cls._REGISTRY[normalized]
        except KeyError:  # pragma: no cover - defensive
            options = ", ".join(cls._REGISTRY) or "<none>"
            raise ValueError(
                f"Unsupported runtime '{value}'. Pick from [{options}]"
            ) from None

    def available_backends(self) -> tuple[Backend, ...]:
        namespace = {
            backend.name: backend for backend in Backend.available(self)
        }
        order = []
        for name in self.backend_names:
            order.append(namespace[name])
        return tuple(order)

    def normalize_backend(self, backend: Backend | str | None) -> Backend:
        if backend is None:
            return Backend.from_any(self.default_backend_name, runtime=self)
        resolved = Backend.from_any(backend, runtime=self)
        return resolved

    def auto_backend_name(self) -> str:
        if self.name == "onnx":
            providers = _get_available_onnx_providers()
            for backend_name, provider_name in _ONNX_AUTO_PRIORITY:
                if provider_name in providers:
                    return backend_name
            return self.default_backend_name
        if self.name == "pt":
            has_torch, has_cuda = _get_torch_capabilities()
            if has_torch and has_cuda:
                return "cuda"
            return self.default_backend_name
        if self.name == "openvino":
            devices = _get_openvino_devices()
            for backend_name, device_prefix in _OPENVINO_AUTO_PRIORITY:
                if any(str(dev).startswith(device_prefix) for dev in devices):
                    return backend_name
            return self.default_backend_name
        return self.default_backend_name

    def __str__(self) -> str:  # pragma: no cover
        return self.name


_ONNX_BACKENDS: tuple[Backend, ...] = (
    Backend(
        name="cpu",
        runtime_key="onnx",
        providers=(ProviderSpec("CPUExecutionProvider"),),
        description="Pure CPU execution provider.",
    ),
    Backend(
        name="cuda",
        runtime_key="onnx",
        providers=(
            ProviderSpec("CUDAExecutionProvider", include_device=True),
            ProviderSpec("CPUExecutionProvider"),
        ),
        description="CUDA with CPU fallback.",
    ),
    Backend(
        name="tensorrt",
        runtime_key="onnx",
        providers=(
            ProviderSpec("TensorrtExecutionProvider", include_device=True),
            ProviderSpec("CUDAExecutionProvider", include_device=True),
            ProviderSpec("CPUExecutionProvider"),
        ),
        description="TensorRT backed by CUDA and CPU providers.",
    ),
    Backend(
        name="tensorrt_rtx",
        runtime_key="onnx",
        providers=(
            ProviderSpec("NvTensorRTRTXExecutionProvider", include_device=True),
            ProviderSpec("CUDAExecutionProvider", include_device=True),
            ProviderSpec("CPUExecutionProvider"),
        ),
        description="TensorRT RTX provider chain.",
    ),
)

_OPENVINO_BACKENDS: tuple[Backend, ...] = (
    Backend(
        name="cpu",
        runtime_key="openvino",
        device="CPU",
        description="Intel CPU device for OpenVINO.",
    ),
    Backend(
        name="gpu",
        runtime_key="openvino",
        device="GPU",
        description="Intel GPU device for OpenVINO.",
    ),
    Backend(
        name="npu",
        runtime_key="openvino",
        device="NPU",
        description="Intel NPU device for OpenVINO.",
    ),
)

_PT_BACKENDS: tuple[Backend, ...] = (
    Backend(
        name="cpu",
        runtime_key="pt",
        device="cpu",
        description="PyTorch CPU execution.",
    ),
    Backend(
        name="cuda",
        runtime_key="pt",
        device="cuda",
        description="PyTorch CUDA execution.",
    ),
)

Runtime.onnx = Runtime(
    name="onnx",
    backend_names=tuple(backend.name for backend in _ONNX_BACKENDS),
    default_backend_name="cpu",
    description="ONNXRuntime execution backend.",
)

Runtime.openvino = Runtime(
    name="openvino",
    backend_names=tuple(backend.name for backend in _OPENVINO_BACKENDS),
    default_backend_name="cpu",
    description="Intel OpenVINO runtime.",
)

Runtime.pt = Runtime(
    name="pt",
    backend_names=tuple(backend.name for backend in _PT_BACKENDS),
    default_backend_name="cpu",
    description="TorchScript/PT runtime.",
)

# Convenience handles for legacy-style access.
Backend.cpu = Backend.from_any("cpu", runtime="onnx")
Backend.cuda = Backend.from_any("cuda", runtime="onnx")
Backend.tensorrt = Backend.from_any("tensorrt", runtime="onnx")
Backend.tensorrt_rtx = Backend.from_any("tensorrt_rtx", runtime="onnx")
Backend.ov_cpu = Backend.from_any("cpu", runtime="openvino")
Backend.ov_gpu = Backend.from_any("gpu", runtime="openvino")
Backend.ov_npu = Backend.from_any("npu", runtime="openvino")


_ONNX_AUTO_PRIORITY: tuple[tuple[str, str], ...] = (
    # Prefer pure CUDA when available to avoid TensorRT dependency issues.
    ("cuda", "CUDAExecutionProvider"),
    ("tensorrt_rtx", "NvTensorRTRTXExecutionProvider"),
    ("tensorrt", "TensorrtExecutionProvider"),
)

_OPENVINO_AUTO_PRIORITY: tuple[tuple[str, str], ...] = (
    # Prefer GPU when available, then NPU; fall back to CPU/default otherwise.
    ("gpu", "GPU"),
    ("npu", "NPU"),
)


def _get_available_onnx_providers() -> set[str]:
    try:  # pragma: no cover - optional dependency
        import onnxruntime as ort  # type: ignore
    except Exception:
        return set()
    try:
        return set(ort.get_available_providers())
    except Exception:  # pragma: no cover - runtime query failure
        return set()


def _get_torch_capabilities() -> tuple[bool, bool]:
    try:  # pragma: no cover - optional dependency
        import torch  # type: ignore
    except Exception:
        return False, False
    try:
        return True, bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - runtime query failure
        return True, False


def _get_openvino_devices() -> set[str]:
    try:  # pragma: no cover - optional dependency
        ov = cast(Any, import_module("openvino.runtime"))
    except Exception:
        return set()
    try:
        core = ov.Core()
        return {str(dev) for dev in getattr(core, "available_devices", [])}
    except Exception:  # pragma: no cover - runtime query failure
        return set()
