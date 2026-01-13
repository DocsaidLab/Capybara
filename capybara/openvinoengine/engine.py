from __future__ import annotations

import contextlib
import queue
import threading
import time
import warnings
from collections import deque
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["OpenVINOConfig", "OpenVINODevice", "OpenVINOEngine"]

try:
    _BFLOAT16 = np.dtype("bfloat16")
except TypeError:  # pragma: no cover
    _BFLOAT16 = np.float16


class OpenVINODevice(str, Enum):
    auto = "AUTO"
    cpu = "CPU"
    gpu = "GPU"
    npu = "NPU"
    hetero = "HETERO"
    auto_batch = "AUTO_BATCH"

    @classmethod
    def from_any(cls, value: Any) -> OpenVINODevice:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.upper()
            for member in cls:
                if member.value == normalized:
                    return member
        raise ValueError(
            f"Unsupported OpenVINO device '{value}'. "
            f"Available: {[member.value for member in cls]}"
        )


@dataclass(slots=True)
class OpenVINOConfig:
    compile_properties: dict[str, Any] | None = None
    core_properties: dict[str, Any] | None = None
    cache_dir: str | Path | None = None
    num_streams: int | None = None
    num_threads: int | None = None
    # None => use engine defaults
    # 0 => OpenVINO auto (async queue only; sync pool falls back to 1)
    # >=1 => fixed number of requests
    num_requests: int | None = None
    # When disabled, outputs may share OpenVINO-owned buffers and can be
    # overwritten by subsequent inference calls.
    copy_outputs: bool = True


@dataclass(slots=True)
class _InputSpec:
    name: str
    dtype: np.dtype[Any] | None
    shape: tuple[int | None, ...]


def _lazy_import_openvino():
    try:
        import openvino.runtime as ov_runtime  # type: ignore
    except Exception as exc:  # pragma: no cover - missing optional dep
        raise ImportError(
            "OpenVINO is required. Install it via 'pip install openvino-dev'."
        ) from exc
    return ov_runtime


def _normalize_device(device: str | OpenVINODevice) -> str:
    if isinstance(device, OpenVINODevice):
        return device.value
    return str(device).upper()


def _validate_benchmark_params(*, repeat: Any, warmup: Any) -> tuple[int, int]:
    repeat_i = int(repeat)
    warmup_i = int(warmup)
    if repeat_i < 1:
        raise ValueError("repeat must be >= 1.")
    if warmup_i < 0:
        raise ValueError("warmup must be >= 0.")
    return repeat_i, warmup_i


class OpenVINOEngine:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | OpenVINODevice = OpenVINODevice.auto,
        config: OpenVINOConfig | None = None,
        core: Any | None = None,
        input_shapes: dict[str, Any] | None = None,
    ) -> None:
        ov = _lazy_import_openvino()

        self.model_path = str(model_path)
        self.device = _normalize_device(device)
        self._cfg = config or OpenVINOConfig()
        self._core = core or ov.Core()
        self._input_shapes = input_shapes
        self._ov = ov

        if self._cfg.core_properties:
            self._core.set_property(self._cfg.core_properties)

        self._type_map = self._build_type_map(ov)
        self._compiled_model = self._compile_model()
        self._input_specs = self._inspect_inputs()
        self._output_ports = list(self._compiled_model.outputs)
        self._output_names = [
            port.get_any_name() for port in self._output_ports
        ]
        self._copy_outputs = bool(self._cfg.copy_outputs)
        self._request_pool = self._create_request_pool()

    def __call__(self, **inputs: np.ndarray) -> dict[str, np.ndarray]:
        if len(inputs) == 1 and isinstance(
            next(iter(inputs.values())), Mapping
        ):
            feed_dict = dict(next(iter(inputs.values())))
        else:
            feed_dict = inputs
        return self.run(feed_dict)

    def run(self, feed: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        prepared = self._prepare_feed(feed)
        infer_request = self._request_pool.get()
        try:
            infer_request.infer(prepared)
        except Exception:
            try:
                replacement = self._compiled_model.create_infer_request()
            except Exception:  # pragma: no cover - unexpected runtime failure
                replacement = infer_request
            self._request_pool.put(replacement)
            raise
        else:
            try:
                return self._collect_outputs(
                    infer_request, copy_outputs=self._copy_outputs
                )
            finally:
                self._request_pool.put(infer_request)

    def create_async_queue(
        self,
        *,
        num_requests: int | None = None,
        copy_outputs: bool | None = None,
    ) -> OpenVINOAsyncQueue:
        """Create an async inference queue backed by AsyncInferQueue.

        This enables pipelining: while the device is running inference for one
        request, your Python code can prepare the next input(s).

        `num_requests` semantics:
        - None: use `OpenVINOConfig.num_requests` (or a default)
        - 0: let OpenVINO decide the number of requests (AUTO)
        - >=1: fixed number of requests
        """
        return OpenVINOAsyncQueue(
            self,
            num_requests=num_requests,
            copy_outputs=copy_outputs,
        )

    def summary(self) -> dict[str, Any]:
        inputs = [
            {
                "name": spec.name,
                "dtype": str(spec.dtype),
                "shape": list(spec.shape),
            }
            for spec in self._input_specs
        ]
        outputs = [
            {
                "name": port.get_any_name(),
                "dtype": str(port.get_element_type()),
                "shape": list(
                    self._partial_shape_to_tuple(port.get_partial_shape())
                ),
            }
            for port in self._output_ports
        ]
        return {
            "model": self.model_path,
            "device": self.device,
            "inputs": inputs,
            "outputs": outputs,
        }

    def benchmark(
        self,
        feed: Mapping[str, np.ndarray],
        *,
        repeat: int = 100,
        warmup: int = 10,
    ) -> dict[str, Any]:
        repeat, warmup = _validate_benchmark_params(
            repeat=repeat, warmup=warmup
        )
        prepared = self._prepare_feed(feed)
        infer_request = self._compiled_model.create_infer_request()

        for _ in range(warmup):
            infer_request.infer(prepared)

        latencies: list[float] = []
        t0 = time.perf_counter()
        for _ in range(repeat):
            start = time.perf_counter()
            infer_request.infer(prepared)
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

    def benchmark_async(
        self,
        feed: Mapping[str, np.ndarray],
        *,
        repeat: int = 100,
        warmup: int = 10,
        num_requests: int | None = None,
    ) -> dict[str, Any]:
        """Benchmark throughput using AsyncInferQueue."""
        repeat, warmup = _validate_benchmark_params(
            repeat=repeat, warmup=warmup
        )
        prepared = self._prepare_feed(feed)
        default_jobs = self._resolve_async_jobs(
            self._cfg.num_requests, default=2
        )
        jobs = self._resolve_async_jobs(num_requests, default=default_jobs)

        # Warmup with synchronous inference to keep behaviour deterministic.
        warm_req = self._compiled_model.create_infer_request()
        for _ in range(warmup):
            warm_req.infer(prepared)

        latencies: list[float] = []
        with self.create_async_queue(num_requests=jobs) as async_queue:
            in_flight_limit = jobs if jobs > 0 else 2
            in_flight_limit = max(1, in_flight_limit)
            in_flight: deque[tuple[Future[dict[str, np.ndarray]], float]] = (
                deque()
            )

            t0 = time.perf_counter()
            for _ in range(repeat):
                if len(in_flight) >= in_flight_limit:
                    fut, start = in_flight.popleft()
                    fut.result()
                    latencies.append((time.perf_counter() - start) * 1e3)

                start = time.perf_counter()
                fut = async_queue.submit(prepared)
                in_flight.append((fut, start))

            while in_flight:
                fut, start = in_flight.popleft()
                fut.result()
                latencies.append((time.perf_counter() - start) * 1e3)
            total = time.perf_counter() - t0

        arr = np.asarray(latencies, dtype=np.float64)
        return {
            "repeat": repeat,
            "warmup": warmup,
            "num_requests": jobs,
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

    def _compile_model(self):
        model = self._core.read_model(self.model_path)
        self._maybe_reshape_model(model)
        properties = dict(self._cfg.compile_properties or {})
        if self._cfg.cache_dir is not None:
            properties["CACHE_DIR"] = str(self._cfg.cache_dir)
        if self._cfg.num_streams is not None:
            properties["NUM_STREAMS"] = str(self._cfg.num_streams)
        if self._cfg.num_threads is not None:
            properties["INFERENCE_NUM_THREADS"] = str(self._cfg.num_threads)
        return self._core.compile_model(model, self.device, properties)

    def _create_request_pool(self) -> queue.Queue[Any]:
        pool_size = self._resolve_pool_size(self._cfg.num_requests)
        pool: queue.Queue[Any] = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            pool.put(self._compiled_model.create_infer_request())
        return pool

    def _resolve_pool_size(self, value: Any | None) -> int:
        if value is None:
            return 1
        requests = int(value)
        if requests < 0:
            raise ValueError("num_requests must be >= 0.")
        if requests == 0:
            return 1
        return requests

    def _resolve_async_jobs(self, value: Any | None, *, default: int) -> int:
        if value is None:
            return int(default)
        requests = int(value)
        if requests < 0:
            raise ValueError("num_requests must be >= 0.")
        return requests

    def _collect_outputs(
        self,
        infer_request: Any,
        *,
        copy_outputs: bool,
    ) -> dict[str, np.ndarray]:
        """Return output tensors as numpy arrays.

        When `copy_outputs=False`, the returned arrays may share OpenVINO-owned
        buffers and can be overwritten by subsequent inference calls.
        """
        outputs: dict[str, np.ndarray] = {}
        for port, name in zip(
            self._output_ports, self._output_names, strict=True
        ):
            tensor = infer_request.get_tensor(port)
            outputs[name] = np.array(tensor.data, copy=copy_outputs)
        return outputs

    def _maybe_reshape_model(self, model: Any) -> None:
        if not self._input_shapes:
            return
        if not hasattr(model, "reshape"):
            raise RuntimeError(
                "OpenVINO model does not support reshape(), "
                "but input_shapes were provided."
            )

        reshape_map: dict[str, Any] = {}
        for name, shape in self._input_shapes.items():
            reshape_map[name] = self._normalize_shape_dims(shape)

        model.reshape(reshape_map)

    def _normalize_shape_dims(self, shape: Any) -> Any:
        if isinstance(shape, (list, tuple)):
            dims: list[int] = []
            for dim in shape:
                if dim is None:
                    raise ValueError(
                        "input_shapes must use concrete dimensions; got None."
                    )
                dims.append(int(dim))
            return tuple(dims)
        return shape

    def _inspect_inputs(self) -> list[_InputSpec]:
        specs: list[_InputSpec] = []
        for port in self._compiled_model.inputs:
            specs.append(
                _InputSpec(
                    name=port.get_any_name(),
                    dtype=self._type_map.get(port.get_element_type()),
                    shape=self._partial_shape_to_tuple(
                        port.get_partial_shape()
                    ),
                )
            )
        return specs

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

    def _partial_shape_to_tuple(self, shape) -> tuple[int | None, ...]:
        dims: list[int | None] = []
        for dim in shape:
            if hasattr(dim, "is_static") and dim.is_static:
                dims.append(int(dim.get_length()))
            elif isinstance(dim, int):
                dims.append(int(dim))
            else:
                dims.append(None)
        return tuple(dims)

    def _build_type_map(self, ov_module) -> dict[Any, np.dtype[Any]]:
        type_cls = getattr(ov_module, "Type", None)
        if type_cls is None:
            return {}
        mapping: dict[Any, np.dtype[Any]] = {}
        pairs = {
            "f32": np.float32,
            "f16": np.float16,
            "bf16": _BFLOAT16,
            "i64": np.int64,
            "i32": np.int32,
            "i16": np.int16,
            "i8": np.int8,
            "u64": np.uint64,
            "u32": np.uint32,
            "u16": np.uint16,
            "u8": np.uint8,
            "boolean": np.bool_,
        }
        for attr, dtype in pairs.items():
            if hasattr(type_cls, attr):
                mapping[getattr(type_cls, attr)] = dtype
        return mapping

    def __repr__(self) -> str:  # pragma: no cover - helper for debugging
        return (
            f"OpenVINOEngine(model='{self.model_path}', device='{self.device}')"
        )


class OpenVINOAsyncQueue:
    """A small wrapper around OpenVINO AsyncInferQueue.

    Use this to overlap preprocessing/postprocessing with device execution.
    """

    def __init__(
        self,
        engine: OpenVINOEngine,
        *,
        num_requests: int | None = None,
        copy_outputs: bool | None = None,
    ) -> None:
        self._engine = engine
        self._ov = engine._ov
        self._copy_outputs = (
            engine._copy_outputs if copy_outputs is None else bool(copy_outputs)
        )
        self._completion_queue_warned = False
        default_jobs = engine._resolve_async_jobs(
            engine._cfg.num_requests, default=2
        )
        jobs = engine._resolve_async_jobs(num_requests, default=default_jobs)

        infer_queue_cls = getattr(self._ov, "AsyncInferQueue", None)
        if infer_queue_cls is None:
            raise RuntimeError(
                "Your OpenVINO installation does not expose AsyncInferQueue. "
                "Please upgrade 'openvino' / 'openvino-dev' to a newer version."
            )

        self._queue = infer_queue_cls(engine._compiled_model, jobs)
        self._queue.set_callback(self._callback)
        self._closed = False
        self._request_id_lock = threading.Lock()
        self._request_id_seq = 0

    def _allocate_request_id(self) -> int:
        with self._request_id_lock:
            request_id = self._request_id_seq
            self._request_id_seq += 1
        return request_id

    def submit(
        self,
        feed: Mapping[str, np.ndarray],
        *,
        request_id: Any | None = None,
        completion_queue: Any | None = None,
    ) -> Future[dict[str, np.ndarray]]:
        """Start an async request and return a Future of raw model outputs.

        If ``completion_queue`` is provided, this enqueues ``(request_id, outputs)``
        in the OpenVINO callback (non-blocking; events are dropped if the queue is
        full or incompatible).

        Note: ``outputs`` are the raw model outputs (before any decoding). When
        ``copy_outputs=False``, the returned arrays may share OpenVINO-owned
        buffers and can be overwritten by subsequent inference calls; consume them
        immediately.

        The returned Future includes a ``request_id`` attribute that matches the
        ID used for submission. If ``request_id`` is omitted, an integer is
        auto-generated. You can also pass any correlation ID (e.g. a string).
        """
        if self._closed:
            raise RuntimeError("Async queue is closed.")
        resolved_request_id = request_id
        if resolved_request_id is None:
            resolved_request_id = self._allocate_request_id()
        future: Future[dict[str, np.ndarray]] = Future()
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            future.request_id = resolved_request_id  # type: ignore[attr-defined]
        if completion_queue is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                future.completion_queue = completion_queue  # type: ignore[attr-defined]
        prepared = self._engine._prepare_feed(feed)
        self._queue.start_async(
            prepared,
            _AsyncUserdata(
                future=future,
                request_id=resolved_request_id,
                completion_queue=completion_queue,
            ),
        )
        return future

    def wait_all(self) -> None:
        self._queue.wait_all()

    def close(self) -> None:
        if self._closed:
            return
        self.wait_all()
        self._closed = True

    def __enter__(self) -> OpenVINOAsyncQueue:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _callback(self, infer_request: Any, userdata: Any) -> None:
        completion_queue = None
        request_id = None
        if isinstance(userdata, _AsyncUserdata):
            future = userdata.future
            completion_queue = userdata.completion_queue
            request_id = userdata.request_id
        elif isinstance(
            userdata, Future
        ):  # pragma: no cover - legacy defensive
            future = userdata
            completion_queue = getattr(future, "completion_queue", None)
            request_id = getattr(future, "request_id", None)
        else:  # pragma: no cover - defensive
            return
        try:
            outputs = self._engine._collect_outputs(
                infer_request, copy_outputs=self._copy_outputs
            )
        except Exception as exc:  # pragma: no cover - surfaced via future
            future.set_exception(exc)
        else:
            future.set_result(outputs)
            if completion_queue is not None and request_id is not None:
                item = (request_id, dict(outputs))
                try:
                    put_nowait = getattr(completion_queue, "put_nowait", None)
                    if callable(put_nowait):
                        put_nowait(item)
                    else:
                        completion_queue.put(item, block=False)
                except queue.Full:
                    if not self._completion_queue_warned:
                        warnings.warn(
                            "completion_queue is full; dropping async completion "
                            "events to avoid blocking the OpenVINO callback thread.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self._completion_queue_warned = True
                except (TypeError, AttributeError):
                    if not self._completion_queue_warned:
                        warnings.warn(
                            "completion_queue does not support non-blocking put; "
                            "dropping async completion events to avoid blocking "
                            "the OpenVINO callback thread.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self._completion_queue_warned = True
                except Exception:  # pragma: no cover - defensive
                    if not self._completion_queue_warned:
                        warnings.warn(
                            "completion_queue enqueue failed; dropping async "
                            "completion events to avoid blocking the OpenVINO "
                            "callback thread.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        self._completion_queue_warned = True


@dataclass(slots=True)
class _AsyncUserdata:
    future: Future[dict[str, np.ndarray]]
    request_id: Any | None
    completion_queue: Any | None
