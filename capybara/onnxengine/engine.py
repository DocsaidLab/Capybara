from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    _BFLOAT16 = np.dtype("bfloat16")
except TypeError:  # pragma: no cover - older numpy
    _BFLOAT16 = np.float16

try:  # pragma: no cover - optional dependency handled at runtime
    import onnxruntime as ort  # type: ignore
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "onnxruntime is required. Install 'onnxruntime' or 'onnxruntime-gpu'."
    ) from exc

from ..runtime import Backend, ProviderSpec


@dataclass(slots=True)
class EngineConfig:
    """High level configuration knobs for the ONNX engine."""

    graph_optimization: str | ort.GraphOptimizationLevel = "all"
    execution_mode: str | ort.ExecutionMode | None = None
    intra_op_num_threads: int | None = None
    inter_op_num_threads: int | None = None
    log_severity_level: int | None = None
    session_config_entries: dict[str, str] | None = None
    provider_options: dict[str, dict[str, Any]] | None = None
    fallback_to_cpu: bool = True
    enable_io_binding: bool = False
    run_config_entries: dict[str, str] | None = None
    enable_profiling: bool = False


@dataclass(slots=True)
class _InputSpec:
    name: str
    dtype: np.dtype[Any] | None
    dtype_str: str
    shape: tuple[int | None, ...]


_GRAPH_OPT_MAP = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}

_EXECUTION_MODE_MAP = {
    "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
    "parallel": ort.ExecutionMode.ORT_PARALLEL,
}

_TYPE_MAP = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(bfloat16)": _BFLOAT16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int16)": np.int16,
    "tensor(int8)": np.int8,
    "tensor(uint64)": np.uint64,
    "tensor(uint32)": np.uint32,
    "tensor(uint16)": np.uint16,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


# NOTE: Benchmarks are intentionally strict about negative/zero loops to avoid
# reporting misleading throughput/latency statistics.
def _validate_benchmark_params(*, repeat: Any, warmup: Any) -> tuple[int, int]:
    repeat_i = int(repeat)
    warmup_i = int(warmup)
    if repeat_i < 1:
        raise ValueError("repeat must be >= 1.")
    if warmup_i < 0:
        raise ValueError("warmup must be >= 0.")
    return repeat_i, warmup_i


# Provider-specific defaults that enable CUDA graph and TensorRT caching tweaks.
_DEFAULT_PROVIDER_OPTIONS: dict[str, dict[str, Any]] = {
    "CUDAExecutionProvider": {
        "cudnn_conv_algo_search": "HEURISTIC",
        "cudnn_conv_use_max_workspace": "1",
        "do_copy_in_default_stream": True,
        "arena_extend_strategy": "kSameAsRequested",
        "tunable_op_enable": False,
        "enable_cuda_graph": False,  # 試驗性功能, 推論效果不穩定
    },
    "TensorrtExecutionProvider": {
        "trt_max_workspace_size": 4 * 1024 * 1024 * 1024,
        "trt_int8_enable": False,
        "trt_fp16_enable": False,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "trt_engine_cache",
        "trt_timing_cache_enable": True,
    },
}


class ONNXEngine:
    """A thin wrapper around onnxruntime.InferenceSession with ergonomic defaults."""

    def __init__(
        self,
        model_path: str | Path,
        gpu_id: int = 0,
        backend: str | Backend = Backend.cpu,
        session_option: Mapping[str, Any] | None = None,
        provider_option: Mapping[str, Any] | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.backend = Backend.from_any(backend, runtime="onnx")
        self.device_id = int(gpu_id)
        self._session_overrides = dict(session_option or {})
        self._provider_override = dict(provider_option or {})
        self._cfg = config or EngineConfig()

        self._session = self._create_session()
        self.providers = self._session.get_providers()
        self.provider_options = self._session.get_provider_options()
        self.metadata = self._extract_metadata()

        self._output_names = [node.name for node in self._session.get_outputs()]
        self._input_specs = self._inspect_inputs()
        self._binding = (
            self._session.io_binding() if self._cfg.enable_io_binding else None
        )
        self._run_options = self._build_run_options()

    def __call__(self, **inputs: np.ndarray) -> dict[str, np.ndarray]:
        if len(inputs) == 1 and isinstance(
            next(iter(inputs.values())), Mapping
        ):
            feed_dict = dict(next(iter(inputs.values())))
        else:
            feed_dict = inputs
        feed = self._prepare_feed(feed_dict)
        return self._run(feed)

    def run(
        self,
        feed: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        return self._run(self._prepare_feed(feed))

    def summary(self) -> dict[str, Any]:
        inputs = [
            {
                "name": spec.name,
                "dtype": spec.dtype_str,
                "shape": list(spec.shape),
            }
            for spec in self._input_specs
        ]
        outputs = [
            {
                "name": out.name,
                "dtype": getattr(out, "type", ""),
                "shape": list(getattr(out, "shape", [])),
            }
            for out in self._session.get_outputs()
        ]
        return {
            "model": self.model_path,
            "providers": self.providers,
            "inputs": inputs,
            "outputs": outputs,
        }

    def benchmark(
        self,
        inputs: Mapping[str, np.ndarray],
        *,
        repeat: int = 100,
        warmup: int = 10,
    ) -> dict[str, Any]:
        repeat, warmup = _validate_benchmark_params(
            repeat=repeat, warmup=warmup
        )
        feed = self._prepare_feed(inputs)
        for _ in range(warmup):
            self._session.run(self._output_names, feed)

        latencies: list[float] = []
        t0 = time.perf_counter()
        for _ in range(repeat):
            start = time.perf_counter()
            self._session.run(self._output_names, feed)
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

    def _run(self, feed: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._binding is not None:
            binding = self._binding
            binding.clear_binding_inputs()
            binding.clear_binding_outputs()
            for name, array in feed.items():
                binding.bind_cpu_input(name, array)
            for name in self._output_names:
                binding.bind_output(name)
            if self._run_options is not None:
                self._session.run_with_iobinding(binding, self._run_options)
            else:
                self._session.run_with_iobinding(binding)
            outputs = binding.copy_outputs_to_cpu()
        else:
            if self._run_options is not None:
                outputs = self._session.run(
                    self._output_names, feed, self._run_options
                )
            else:
                outputs = self._session.run(self._output_names, feed)
        converted: list[np.ndarray] = []
        for out in outputs:
            toarray = getattr(out, "toarray", None)
            if callable(toarray):
                out = toarray()
            converted.append(np.asarray(out))
        return dict(zip(self._output_names, converted, strict=False))

    def _prepare_feed(self, feed: Mapping[str, Any]) -> dict[str, np.ndarray]:
        prepared: dict[str, np.ndarray] = {}
        for spec in self._input_specs:
            if spec.name not in feed:
                raise KeyError(f"Missing required input '{spec.name}'.")
            array = np.asarray(feed[spec.name])
            if spec.dtype is not None and array.dtype != spec.dtype:
                array = array.astype(spec.dtype, copy=False)
            prepared[spec.name] = array
        return prepared

    def _create_session(self) -> ort.InferenceSession:
        session_options = self._build_session_options()
        provider_tuples = self._resolve_providers()
        provider_names = [name for name, _ in provider_tuples]
        provider_options = [opts for _, opts in provider_tuples]

        available = set(ort.get_available_providers())
        missing = [
            name
            for name in provider_names
            if name != "CPUExecutionProvider" and name not in available
        ]
        if missing and self._cfg.fallback_to_cpu:
            provider_names = ["CPUExecutionProvider"]
            provider_options = [
                self._build_provider_options("CPUExecutionProvider")
            ]

        return ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=provider_names,
            provider_options=provider_options,
        )

    def _build_session_options(self) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        cfg = self._cfg

        opts.enable_profiling = bool(cfg.enable_profiling)
        opts.graph_optimization_level = self._resolve_graph_optimization(
            cfg.graph_optimization
        )
        if cfg.execution_mode is not None:
            opts.execution_mode = self._resolve_execution_mode(
                cfg.execution_mode
            )
        if cfg.intra_op_num_threads is not None:
            opts.intra_op_num_threads = int(cfg.intra_op_num_threads)
        if cfg.inter_op_num_threads is not None:
            opts.inter_op_num_threads = int(cfg.inter_op_num_threads)
        if cfg.log_severity_level is not None:
            opts.log_severity_level = int(cfg.log_severity_level)

        for key, value in (cfg.session_config_entries or {}).items():
            opts.add_session_config_entry(str(key), str(value))
        for key, value in self._session_overrides.items():
            if hasattr(opts, key):
                setattr(opts, key, value)
            else:
                opts.add_session_config_entry(str(key), str(value))

        return opts

    def _resolve_graph_optimization(
        self,
        option: str | ort.GraphOptimizationLevel,
    ) -> ort.GraphOptimizationLevel:
        if isinstance(option, ort.GraphOptimizationLevel):
            return option
        normalized = str(option).lower()
        return _GRAPH_OPT_MAP.get(
            normalized, ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

    def _resolve_execution_mode(
        self,
        option: str | ort.ExecutionMode,
    ) -> ort.ExecutionMode:
        if isinstance(option, ort.ExecutionMode):
            return option
        normalized = str(option).lower()
        return _EXECUTION_MODE_MAP.get(
            normalized, ort.ExecutionMode.ORT_SEQUENTIAL
        )

    def _resolve_providers(self) -> list[tuple[str, dict[str, str]]]:
        cfg_providers = self._cfg.provider_options or {}
        provider_specs = self.backend.providers or (
            ProviderSpec("CPUExecutionProvider"),
        )
        merged: list[tuple[str, dict[str, str]]] = []
        for spec in provider_specs:
            opts = self._build_provider_options(
                spec.name, include_device=spec.include_device
            )
            merged_opts = dict(opts)
            for key, value in cfg_providers.get(spec.name, {}).items():
                merged_opts[str(key)] = self._provider_value(value)
            merged.append((spec.name, merged_opts))
        return merged

    def _build_provider_options(
        self,
        name: str,
        *,
        include_device: bool = False,
    ) -> dict[str, str]:
        opts: dict[str, Any] = dict(_DEFAULT_PROVIDER_OPTIONS.get(name, {}))
        if include_device:
            opts["device_id"] = self.device_id
        opts.update(
            self._provider_override
            if (include_device or name == "CPUExecutionProvider")
            else {}
        )
        return {str(k): self._provider_value(v) for k, v in opts.items()}

    def _provider_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "True" if value else "False"
        return str(value)

    def _build_run_options(self) -> ort.RunOptions | None:
        entries = {
            **(self._cfg.run_config_entries or {}),
        }
        if not entries:
            return None
        opts = ort.RunOptions()
        for key, value in entries.items():
            opts.add_run_config_entry(str(key), str(value))
        return opts

    def _extract_metadata(self) -> dict[str, Any] | None:
        try:
            model_meta = self._session.get_modelmeta()
        except Exception:
            return None

        custom = getattr(model_meta, "custom_metadata_map", None)
        if not isinstance(custom, Mapping):
            return None

        parsed: dict[str, Any] = {}
        for key, value in custom.items():
            if isinstance(value, str):
                try:
                    parsed[key] = json.loads(value)
                except Exception:
                    parsed[key] = value
            else:
                parsed[key] = value
        return parsed or None

    def _inspect_inputs(self) -> list[_InputSpec]:
        specs: list[_InputSpec] = []
        for node in self._session.get_inputs():
            dtype = _TYPE_MAP.get(getattr(node, "type", ""))
            raw_shape = list(getattr(node, "shape", []))
            shape: list[int | None] = []
            for dim in raw_shape:
                if isinstance(dim, (int, np.integer)):
                    shape.append(int(dim))
                else:
                    shape.append(None)
            specs.append(
                _InputSpec(
                    name=node.name,
                    dtype=dtype,
                    dtype_str=getattr(node, "type", ""),
                    shape=tuple(shape),
                )
            )
        return specs

    def __repr__(self) -> str:  # pragma: no cover - human readable summary
        return (
            f"ONNXEngine(model='{self.model_path}', "
            f"backend='{self.backend.name}', providers={self.providers})"
        )
