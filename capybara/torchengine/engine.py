from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["TorchEngine", "TorchEngineConfig"]


def _lazy_import_torch():
    try:  # pragma: no cover - optional dependency
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - surfaced in init
        raise ImportError(
            "PyTorch is required to use Runtime.pt. Please install torch>=2.0."
        ) from exc
    return torch


def _validate_benchmark_params(*, repeat: Any, warmup: Any) -> tuple[int, int]:
    repeat_i = int(repeat)
    warmup_i = int(warmup)
    if repeat_i < 1:
        raise ValueError("repeat must be >= 1.")
    if warmup_i < 0:
        raise ValueError("warmup must be >= 0.")
    return repeat_i, warmup_i


@dataclass(slots=True)
class TorchEngineConfig:
    dtype: str | Any | None = None
    cuda_sync: bool = True


class TorchEngine:
    """Thin wrapper around torch.jit.ScriptModule for AngiDetection."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | Any = "cuda",
        output_names: Sequence[str] | None = None,
        config: TorchEngineConfig | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self._torch = _lazy_import_torch()
        self._cfg = config or TorchEngineConfig()
        self.device = self._normalize_device(device)
        self.output_names = tuple(output_names or ())
        self.dtype = self._normalize_dtype(self._cfg.dtype)
        self._model = self._load_model()

    def __call__(self, **inputs: Any) -> dict[str, np.ndarray]:
        if len(inputs) == 1 and isinstance(
            next(iter(inputs.values())), Mapping
        ):
            feed_dict = dict(next(iter(inputs.values())))
        else:
            feed_dict = inputs
        return self.run(feed_dict)

    def run(self, feed: Mapping[str, Any]) -> dict[str, np.ndarray]:
        prepared = self._prepare_feed(feed)
        outputs = self._forward(prepared)
        return self._format_outputs(outputs)

    def benchmark(
        self,
        inputs: Mapping[str, Any],
        *,
        repeat: int = 100,
        warmup: int = 10,
        cuda_sync: bool | None = None,
    ) -> dict[str, Any]:
        repeat, warmup = _validate_benchmark_params(
            repeat=repeat, warmup=warmup
        )
        prepared = self._prepare_feed(inputs)
        sync = self._should_sync(cuda_sync)
        with self._torch.inference_mode():
            for _ in range(warmup):
                self._forward(prepared)
                if sync:
                    self._sync()

        latencies: list[float] = []
        t0 = time.perf_counter()
        with self._torch.inference_mode():
            for _ in range(repeat):
                if sync:
                    self._sync()
                start = time.perf_counter()
                self._forward(prepared)
                if sync:
                    self._sync()
                latencies.append((time.perf_counter() - start) * 1e3)
        total = time.perf_counter() - t0
        arr = np.asarray(latencies, dtype=np.float64)
        return {
            "repeat": repeat,
            "warmup": warmup,
            "throughput_fps": repeat / total if total else None,
            "latency_ms": {
                "mean": float(arr.mean()) if arr.size else None,
                "median": float(np.median(arr)) if arr.size else None,
                "p90": float(np.percentile(arr, 90)) if arr.size else None,
                "p95": float(np.percentile(arr, 95)) if arr.size else None,
                "min": float(arr.min()) if arr.size else None,
                "max": float(arr.max()) if arr.size else None,
            },
        }

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_path,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "outputs": list(self.output_names),
        }

    # Internal helpers -----------------------------------------------------
    def _load_model(self):
        torch = self._torch
        model = torch.jit.load(self.model_path, map_location=self.device)
        model.eval()
        with torch.no_grad():
            model.to(self.device)
            if self.dtype == torch.float16:
                model.half()
            elif self.dtype == torch.float32:
                model.float()
            else:
                model.to(dtype=self.dtype)
        return model

    def _prepare_feed(self, feed: Mapping[str, Any]) -> list[Any]:
        if not isinstance(feed, Mapping):
            raise TypeError("TorchEngine feed must be a mapping.")
        prepared: list[Any] = []
        for value in feed.values():
            tensor = self._as_tensor(value)
            tensor = tensor.to(self.device)
            tensor = tensor.to(self.dtype)
            prepared.append(tensor)
        return prepared

    def _forward(self, inputs: Sequence[Any]) -> Any:
        if len(inputs) == 1:
            return self._model(inputs[0])
        return self._model(*inputs)

    def _format_outputs(self, outputs: Any) -> dict[str, np.ndarray]:
        torch = self._torch
        if isinstance(outputs, Mapping):
            return {
                str(key): self._tensor_to_numpy(value)
                for key, value in outputs.items()
            }
        if isinstance(outputs, (list, tuple)):
            names = self._normalize_output_names(len(outputs))
            return {
                names[idx]: self._tensor_to_numpy(value)
                for idx, value in enumerate(outputs)
            }
        if torch.is_tensor(outputs):
            name = self.output_names[0] if self.output_names else "output"
            return {name: self._tensor_to_numpy(outputs)}
        raise TypeError(
            "Unsupported TorchScript output. Expected tensor/dict/sequence."
        )

    def _normalize_output_names(self, count: int) -> tuple[str, ...]:
        if self.output_names:
            if len(self.output_names) != count:
                raise ValueError(
                    f"output_names has {len(self.output_names)} entries but "
                    f"model produced {count} outputs."
                )
            return self.output_names
        return tuple(f"output_{idx}" for idx in range(count))

    def _tensor_to_numpy(self, tensor: Any) -> np.ndarray:
        torch = self._torch
        if not torch.is_tensor(tensor):
            raise TypeError("Model outputs must be torch.Tensor instances.")
        array = tensor.detach().to("cpu")
        if array.dtype != torch.float32:
            array = array.to(torch.float32)
        return array.contiguous().numpy()

    def _as_tensor(self, value: Any):
        torch = self._torch
        if torch.is_tensor(value):
            return value
        arr = np.asarray(value, dtype=np.float32)
        return torch.from_numpy(arr)

    def _normalize_device(self, device: Any):
        torch = self._torch
        torch_device_type = getattr(torch, "device", None)
        if isinstance(torch_device_type, type) and isinstance(
            device, torch_device_type
        ):
            return device
        return torch.device(device)

    def _normalize_dtype(self, dtype: Any | None):
        torch = self._torch
        if dtype is None or (
            isinstance(dtype, str) and dtype.strip().lower() == "auto"
        ):
            token = Path(self.model_path).name.lower()
            if "fp16" in token and self._device_is_cuda():
                return torch.float16
            return torch.float32
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            normalized = dtype.strip().lower()
            if normalized in {"fp16", "float16", "half"}:
                return torch.float16
            if normalized in {"fp32", "float32"}:
                return torch.float32
        raise ValueError(f"Unsupported dtype specification '{dtype}'.")

    def _device_is_cuda(self) -> bool:
        return getattr(self.device, "type", "") == "cuda"

    def _should_sync(self, override: bool | None) -> bool:
        if override is not None:
            return bool(override and self._device_is_cuda())
        return bool(self._cfg.cuda_sync and self._device_is_cuda())

    def _sync(self) -> None:
        if self._device_is_cuda():
            self._torch.cuda.synchronize(self.device)
