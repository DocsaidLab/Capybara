from __future__ import annotations

import importlib
import queue
import sys
import threading
import types
import warnings
from typing import Any

import numpy as np
import pytest


@pytest.fixture()
def openvino_engine_module(monkeypatch):
    class FakeType:
        f32 = object()
        i64 = object()
        boolean = object()

    class FakeDim:
        def __init__(self, value):
            self.value = value
            self.is_static = value is not None

        def get_length(self):
            if self.value is None:
                raise ValueError("dynamic dim")
            return self.value

    class FakePort:
        def __init__(self, name, element_type, shape):
            self._name = name
            self._type = element_type
            self._shape = shape

        def get_any_name(self):
            return self._name

        def get_element_type(self):
            return self._type

        def get_partial_shape(self):
            dims = []
            for value in self._shape:
                if isinstance(value, int):
                    dims.append(FakeDim(value))
                else:
                    dims.append(FakeDim(None))
            return dims

    class FakeTensor:
        def __init__(self, value):
            self.data = np.full((1, 2), value, dtype=np.float32)

    class FakeInferRequest:
        def __init__(self, outputs):
            self._outputs = outputs

        def infer(self, feed):
            self._last_feed = feed

        def get_tensor(self, port):
            idx = self._outputs.index(port)
            return FakeTensor(idx + 1)

    class FakeCompiledModel:
        def __init__(self, input_shapes=None):
            shape = (1, None, 3)
            if input_shapes:
                shape = input_shapes.get("input", shape)
            self.inputs = [
                FakePort("input", FakeType.f32, shape),
            ]
            self.outputs = [
                FakePort("output", FakeType.f32, (1, 2)),
            ]

        def create_infer_request(self):
            return FakeInferRequest(self.outputs)

    class FakeAsyncInferQueue:
        def __init__(self, compiled_model, jobs):
            self._compiled_model = compiled_model
            self._jobs = int(jobs)
            self._callback = None

        def set_callback(self, callback):
            self._callback = callback

        def start_async(self, feed, userdata):
            req = self._compiled_model.create_infer_request()
            req.infer(feed)
            if self._callback is not None:
                self._callback(req, userdata)

        def wait_all(self):
            return None

    class FakeCore:
        def __init__(self):
            self._properties = {}
            self._last_reshape = None

        def set_property(self, props):
            self._properties.update(props)

        def read_model(self, path):
            class FakeModel:
                def __init__(self, p):
                    self.path = p
                    self.reshape_map = {}

                def reshape(self, mapping):
                    self.reshape_map = mapping

            return FakeModel(path)

        def compile_model(self, model, device, properties):
            self._properties["device"] = device
            self._properties.update(properties)
            self._last_reshape = getattr(model, "reshape_map", {})
            return FakeCompiledModel(self._last_reshape)

    fake_runtime: Any = types.ModuleType("openvino.runtime")
    fake_runtime.Type = FakeType
    fake_runtime.Core = FakeCore
    fake_runtime.AsyncInferQueue = FakeAsyncInferQueue
    fake_pkg: Any = types.ModuleType("openvino")
    fake_pkg.runtime = fake_runtime

    monkeypatch.setitem(sys.modules, "openvino", fake_pkg)
    monkeypatch.setitem(sys.modules, "openvino.runtime", fake_runtime)
    module = importlib.reload(
        importlib.import_module("capybara.openvinoengine.engine")
    )
    yield module


def test_openvino_engine_runs_with_stub(openvino_engine_module, tmp_path):
    config_cls = openvino_engine_module.OpenVINOConfig
    engine_cls = openvino_engine_module.OpenVINOEngine
    device_enum = openvino_engine_module.OpenVINODevice

    cfg = config_cls(
        compile_properties={"PERF_HINT": "THROUGHPUT"},
        core_properties={"LOG_LEVEL": "ERROR"},
        cache_dir=tmp_path / "cache",
        num_streams=2,
        num_threads=4,
    )

    engine = engine_cls(
        model_path="model.xml",
        device=device_enum.cpu,
        config=cfg,
    )

    feed = {"input": np.ones((1, 3), dtype=np.float32)}
    outputs = engine(**feed)
    assert outputs["output"].shape == (1, 2)

    summary = engine.summary()
    assert summary["device"] == "CPU"
    assert summary["inputs"][0]["shape"][1] is None


def test_openvino_engine_accepts_input_shapes(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine
    device_enum = openvino_engine_module.OpenVINODevice

    engine = engine_cls(
        model_path="model.xml",
        device=device_enum.npu,
        input_shapes={"input": (2, 3, 5)},
    )

    summary = engine.summary()
    assert summary["inputs"][0]["shape"] == [2, 3, 5]


def test_openvino_engine_async_queue(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine

    engine = engine_cls(model_path="model.xml", device="CPU")
    with engine.create_async_queue(num_requests=2) as q:
        fut = q.submit({"input": np.ones((1, 3), dtype=np.float32)})
        outputs = fut.result(timeout=1)
    assert outputs["output"].shape == (1, 2)


def test_openvino_engine_async_queue_auto_requests(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine

    engine = engine_cls(model_path="model.xml", device="CPU")
    with engine.create_async_queue(num_requests=0) as q:
        fut = q.submit({"input": np.ones((1, 3), dtype=np.float32)})
        outputs = fut.result(timeout=1)
    assert outputs["output"].shape == (1, 2)


def test_openvino_engine_benchmark_async(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine

    engine = engine_cls(model_path="model.xml", device="CPU")
    stats = engine.benchmark_async(
        {"input": np.ones((1, 3), dtype=np.float32)},
        repeat=10,
        warmup=1,
        num_requests=2,
    )
    assert stats["num_requests"] == 2


def test_openvino_engine_async_queue_auto_request_id(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine

    engine = engine_cls(model_path="model.xml", device="CPU")
    completion = queue.Queue()
    with engine.create_async_queue(num_requests=2) as q:
        fut = q.submit(
            {"input": np.ones((1, 3), dtype=np.float32)},
            completion_queue=completion,
        )
        outputs = fut.result(timeout=1)
        req_id, event_outputs = completion.get(timeout=1)

    assert getattr(fut, "request_id", None) is not None
    assert req_id == fut.request_id
    assert event_outputs is not outputs


def test_openvino_engine_async_queue_preserves_request_id(
    openvino_engine_module,
):
    engine_cls = openvino_engine_module.OpenVINOEngine

    engine = engine_cls(model_path="model.xml", device="CPU")
    completion = queue.Queue()
    with engine.create_async_queue(num_requests=2) as q:
        fut = q.submit(
            {"input": np.ones((1, 3), dtype=np.float32)},
            request_id="req-123",
            completion_queue=completion,
        )
        outputs = fut.result(timeout=1)
        req_id, event_outputs = completion.get(timeout=1)

    assert fut.request_id == "req-123"
    assert req_id == "req-123"
    assert event_outputs is not outputs


def test_openvino_engine_async_queue_completion_queue(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine

    engine = engine_cls(model_path="model.xml", device="CPU")
    completion = queue.Queue()
    with engine.create_async_queue(num_requests=2) as q:
        fut = q.submit(
            {"input": np.ones((1, 3), dtype=np.float32)},
            request_id="req-1",
            completion_queue=completion,
        )
        outputs = fut.result(timeout=1)
        req_id, event_outputs = completion.get(timeout=1)

    assert req_id == "req-1"
    assert event_outputs is not outputs
    outputs["mutated"] = np.zeros((1,), dtype=np.float32)
    assert "mutated" not in event_outputs
    assert outputs["output"].shape == (1, 2)
    assert event_outputs["output"].shape == (1, 2)


def test_openvino_engine_async_queue_completion_queue_full_does_not_block(
    openvino_engine_module,
):
    engine_cls = openvino_engine_module.OpenVINOEngine

    engine = engine_cls(model_path="model.xml", device="CPU")
    completion = queue.Queue(maxsize=1)
    completion.put(("sentinel", {}))

    result: dict[str, object] = {}

    def submit_request():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with engine.create_async_queue(num_requests=2) as q:
                fut = q.submit(
                    {"input": np.ones((1, 3), dtype=np.float32)},
                    request_id="req-1",
                    completion_queue=completion,
                )
                result["outputs"] = fut.result(timeout=1)

    thread = threading.Thread(target=submit_request, daemon=True)
    thread.start()
    thread.join(timeout=0.5)

    if thread.is_alive():
        completion.get_nowait()
        thread.join(timeout=0.5)
        pytest.fail("q.submit() blocked when completion_queue was full")

    assert completion.qsize() == 1
    assert completion.get_nowait()[0] == "sentinel"
    assert isinstance(result.get("outputs"), dict)


def test_openvino_engine_async_queue_completion_queue_full_emits_warning(
    openvino_engine_module,
):
    engine_cls = openvino_engine_module.OpenVINOEngine

    engine = engine_cls(model_path="model.xml", device="CPU")
    completion = queue.Queue(maxsize=1)
    completion.put(("sentinel", {}))

    with engine.create_async_queue(num_requests=2) as q:
        with pytest.warns(RuntimeWarning, match="completion_queue is full"):
            fut = q.submit(
                {"input": np.ones((1, 3), dtype=np.float32)},
                request_id="req-1",
                completion_queue=completion,
            )
        outputs = fut.result(timeout=1)

    assert outputs["output"].shape == (1, 2)
    assert completion.qsize() == 1
    assert completion.get_nowait()[0] == "sentinel"


def test_openvino_device_from_any_accepts_strings_and_rejects_unknown(
    openvino_engine_module,
):
    device_enum = openvino_engine_module.OpenVINODevice

    assert device_enum.from_any(device_enum.cpu) is device_enum.cpu
    assert device_enum.from_any("cpu") is device_enum.cpu

    with pytest.raises(ValueError, match="Unsupported OpenVINO device"):
        device_enum.from_any("tpu")


def test_openvino_engine_accepts_wrapped_feed_dict(openvino_engine_module):
    """__call__ supports passing a single mapping payload as a kwarg."""
    engine_cls = openvino_engine_module.OpenVINOEngine
    engine = engine_cls(model_path="model.xml", device="CPU")

    feed = {"input": np.ones((1, 3), dtype=np.float32)}
    outputs = engine(payload=feed)
    assert outputs["output"].shape == (1, 2)


def test_openvino_engine_run_replaces_failed_request(openvino_engine_module):
    """Infer failures should not leave a broken request in the pool."""
    engine_cls = openvino_engine_module.OpenVINOEngine

    engine = engine_cls(model_path="model.xml", device="CPU")
    feed = {"input": np.ones((1, 3), dtype=np.float32)}

    failing_request = engine._request_pool.get_nowait()

    def boom(_prepared):
        raise RuntimeError("infer boom")

    failing_request.infer = boom  # type: ignore[assignment]
    engine._request_pool.put(failing_request)

    with pytest.raises(RuntimeError, match="infer boom"):
        engine.run(feed)

    replacement = engine._request_pool.get_nowait()
    assert replacement is not failing_request
    engine._request_pool.put(replacement)

    outputs = engine.run(feed)
    assert outputs["output"].shape == (1, 2)


def test_openvino_engine_benchmark_sync(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine

    engine = engine_cls(model_path="model.xml", device="CPU")
    stats = engine.benchmark(
        {"input": np.ones((1, 3), dtype=np.float32)},
        repeat=3,
        warmup=1,
    )

    assert stats["repeat"] == 3
    assert "latency_ms" in stats


def test_openvino_engine_benchmark_validates_repeat_and_warmup(
    openvino_engine_module,
):
    engine_cls = openvino_engine_module.OpenVINOEngine
    engine = engine_cls(model_path="model.xml", device="CPU")
    feed = {"input": np.ones((1, 3), dtype=np.float32)}

    with pytest.raises(ValueError, match="repeat must be >= 1"):
        engine.benchmark(feed, repeat=0)

    with pytest.raises(ValueError, match="warmup must be >= 0"):
        engine.benchmark(feed, warmup=-1)

    with pytest.raises(ValueError, match="repeat must be >= 1"):
        engine.benchmark_async(feed, repeat=0)

    with pytest.raises(ValueError, match="warmup must be >= 0"):
        engine.benchmark_async(feed, warmup=-1)


def test_openvino_engine_num_requests_validation(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine
    config_cls = openvino_engine_module.OpenVINOConfig

    engine = engine_cls(
        model_path="model.xml",
        device="CPU",
        config=config_cls(num_requests=0),
    )
    assert engine._request_pool.maxsize == 1

    with pytest.raises(ValueError, match="num_requests must be >= 0"):
        engine_cls(
            model_path="model.xml",
            device="CPU",
            config=config_cls(num_requests=-1),
        )

    with pytest.raises(ValueError, match="num_requests must be >= 0"):
        engine.create_async_queue(num_requests=-1)


def test_openvino_engine_input_shapes_reject_none_dims(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine

    with pytest.raises(ValueError, match="must use concrete dimensions"):
        engine_cls(
            model_path="model.xml",
            device="CPU",
            input_shapes={"input": (1, None, 3)},
        )


def test_openvino_engine_requires_model_reshape_when_input_shapes(
    openvino_engine_module,
):
    engine_cls = openvino_engine_module.OpenVINOEngine

    class NoReshapeCore:
        def read_model(self, path):
            return object()

        def compile_model(self, model, device, properties):  # pragma: no cover
            raise AssertionError(
                "compile_model should not run when reshape fails"
            )

    with pytest.raises(RuntimeError, match="does not support reshape"):
        engine_cls(
            model_path="model.xml",
            device="CPU",
            core=NoReshapeCore(),
            input_shapes={"input": (1, 3, 5)},
        )


def test_openvino_engine_prepare_feed_validates_and_casts(
    openvino_engine_module,
):
    engine_cls = openvino_engine_module.OpenVINOEngine
    engine = engine_cls(model_path="model.xml", device="CPU")

    with pytest.raises(KeyError, match="Missing required input"):
        engine.run({"missing": np.ones((1, 3), dtype=np.float32)})

    feed = {"input": np.ones((1, 3), dtype=np.float64)}
    engine.run(feed)
    req = engine._request_pool.get_nowait()
    assert req._last_feed["input"].dtype == np.float32  # type: ignore[attr-defined]
    engine._request_pool.put(req)


def test_openvino_partial_shape_to_tuple_accepts_int_dims(
    openvino_engine_module,
):
    engine_cls = openvino_engine_module.OpenVINOEngine
    engine = engine_cls(model_path="model.xml", device="CPU")

    assert engine._partial_shape_to_tuple([1, 2, "dyn"]) == (1, 2, None)


def test_openvino_build_type_map_handles_missing_type(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine
    engine = engine_cls(model_path="model.xml", device="CPU")

    assert engine._build_type_map(types.SimpleNamespace()) == {}


def test_openvino_async_queue_requires_asyncinferqueue(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine
    engine = engine_cls(model_path="model.xml", device="CPU")
    engine._ov = types.SimpleNamespace()  # no AsyncInferQueue

    with pytest.raises(RuntimeError, match="AsyncInferQueue"):
        engine.create_async_queue(num_requests=2)


def test_openvino_async_queue_submit_after_close_raises(openvino_engine_module):
    engine_cls = openvino_engine_module.OpenVINOEngine
    engine = engine_cls(model_path="model.xml", device="CPU")

    q = engine.create_async_queue(num_requests=1)
    q.close()
    q.close()  # idempotent
    with pytest.raises(RuntimeError, match="Async queue is closed"):
        q.submit({"input": np.ones((1, 3), dtype=np.float32)})


def test_openvino_async_queue_completion_queue_put_fallback(
    openvino_engine_module,
):
    """Fallback to completion_queue.put(block=False) when put_nowait is absent."""
    engine_cls = openvino_engine_module.OpenVINOEngine
    engine = engine_cls(model_path="model.xml", device="CPU")

    class PutOnlyQueue:
        def __init__(self):
            self.items = []

        def put(self, item, *, block=False):
            self.items.append((item, block))

    completion = PutOnlyQueue()
    with engine.create_async_queue(num_requests=1) as q:
        fut = q.submit(
            {"input": np.ones((1, 3), dtype=np.float32)},
            request_id="req-1",
            completion_queue=completion,
        )
        fut.result(timeout=1)

    assert completion.items[0][0][0] == "req-1"
    assert completion.items[0][1] is False


def test_openvino_async_queue_completion_queue_signature_mismatch_warns(
    openvino_engine_module,
):
    """Completion queues without non-blocking put semantics should warn once."""
    engine_cls = openvino_engine_module.OpenVINOEngine
    engine = engine_cls(model_path="model.xml", device="CPU")

    class BadQueue:
        def put(self, item):
            self.item = item

    completion = BadQueue()
    with engine.create_async_queue(num_requests=1) as q:
        with pytest.warns(
            RuntimeWarning,
            match="does not support non-blocking put",
        ):
            fut = q.submit(
                {"input": np.ones((1, 3), dtype=np.float32)},
                request_id="req-1",
                completion_queue=completion,
            )
        fut.result(timeout=1)


def test_openvino_engine_request_pool_respects_num_requests(
    openvino_engine_module,
):
    config_cls = openvino_engine_module.OpenVINOConfig
    engine_cls = openvino_engine_module.OpenVINOEngine

    cfg = config_cls(num_requests=3)
    engine = engine_cls(model_path="model.xml", device="CPU", config=cfg)

    assert engine._request_pool.maxsize == 3


def test_openvino_engine_input_shapes_accepts_non_sequence_values(
    openvino_engine_module,
):
    engine_cls = openvino_engine_module.OpenVINOEngine

    engine = engine_cls(
        model_path="model.xml",
        device="CPU",
        input_shapes={"input": {"dim": 3}},
    )

    assert engine._core._last_reshape["input"] == {"dim": 3}
