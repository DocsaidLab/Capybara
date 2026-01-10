from __future__ import annotations

import importlib
import sys
import types
from enum import Enum
from typing import Any

import numpy as np
import pytest


@pytest.fixture()
def onnx_engine_module(monkeypatch):
    class GraphOptimizationLevel(Enum):
        ORT_DISABLE_ALL = "disable"
        ORT_ENABLE_BASIC = "basic"
        ORT_ENABLE_EXTENDED = "extended"
        ORT_ENABLE_ALL = "all"

    class ExecutionMode(Enum):
        ORT_SEQUENTIAL = "seq"
        ORT_PARALLEL = "par"

    modelmeta_holder: dict[str, dict[str, object] | None] = {"map": None}

    class SessionOptions:
        def __init__(self):
            self.enable_profiling = False
            self.graph_optimization_level = None
            self.execution_mode = None
            self.intra_op_num_threads = None
            self.inter_op_num_threads = None
            self.log_severity_level = None
            self.config_entries: dict[str, str] = {}

        def add_session_config_entry(self, key: str, value: str):
            self.config_entries[key] = value

    class RunOptions:
        def __init__(self):
            self.entries: dict[str, str] = {}

        def add_run_config_entry(self, key: str, value: str):
            self.entries[key] = value

    class FakeIOBinding:
        def __init__(self, session):
            self.session = session
            self.output_storage: dict[str, np.ndarray] = {}
            self.bound_outputs: list[str] = []
            self.bound_inputs: dict[str, np.ndarray] = {}

        def clear_binding_inputs(self):
            self.bound_inputs.clear()

        def clear_binding_outputs(self):
            self.bound_outputs.clear()
            self.output_storage = {}

        def bind_cpu_input(self, name: str, array: np.ndarray):
            self.bound_inputs[name] = array

        def bind_output(self, name: str):
            self.bound_outputs.append(name)

        def copy_outputs_to_cpu(self) -> list[np.ndarray]:
            return [self.output_storage[name] for name in self.bound_outputs]

    class FakeSession:
        def __init__(
            self, model_path, sess_options, providers, provider_options
        ):
            self.model_path = model_path
            self.sess_options = sess_options
            self._providers = providers
            self._provider_options = provider_options
            self._inputs = [
                types.SimpleNamespace(
                    name="input", type="tensor(float)", shape=[1, "dyn", 4]
                )
            ]
            self._outputs = [
                types.SimpleNamespace(
                    name="output", type="tensor(float)", shape=[1, 4]
                )
            ]
            self._binding = FakeIOBinding(self)
            self.last_run_options = None

        def get_providers(self):
            return self._providers

        def get_provider_options(self):
            return self._provider_options

        def get_modelmeta(self):
            meta_map = modelmeta_holder["map"]
            if meta_map is None:
                raise RuntimeError("no metadata configured")
            return types.SimpleNamespace(custom_metadata_map=meta_map)

        def get_outputs(self):
            return self._outputs

        def get_inputs(self):
            return self._inputs

        def io_binding(self):
            return self._binding

        def run(self, output_names, feed, run_options=None):
            self.last_run_options = run_options
            return [np.ones((1, 4), dtype=np.float32) for _ in output_names]

        def run_with_iobinding(self, binding, run_options=None):
            self.last_run_options = run_options
            for idx, name in enumerate(binding.bound_outputs):
                binding.output_storage[name] = np.full(
                    (1, 4), float(idx), dtype=np.float32
                )

    fake_ort: Any = types.ModuleType("onnxruntime")
    fake_ort.GraphOptimizationLevel = GraphOptimizationLevel
    fake_ort.ExecutionMode = ExecutionMode
    fake_ort.SessionOptions = SessionOptions
    fake_ort.RunOptions = RunOptions
    fake_ort.InferenceSession = FakeSession
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]

    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    module = importlib.reload(
        importlib.import_module("capybara.onnxengine.engine")
    )
    module_any: Any = module
    module_any._test_modelmeta_holder = modelmeta_holder
    yield module


def test_onnx_engine_with_stubbed_runtime(onnx_engine_module):
    engine_config_cls = onnx_engine_module.EngineConfig
    onnx_engine_cls = onnx_engine_module.ONNXEngine

    config = engine_config_cls(
        graph_optimization="basic",
        execution_mode="parallel",
        intra_op_num_threads=2,
        inter_op_num_threads=1,
        log_severity_level=0,
        session_config_entries={"session.use": "1"},
        provider_options={
            "CUDAExecutionProvider": {"arena_extend_strategy": "manual"}
        },
        fallback_to_cpu=True,
        enable_io_binding=True,
        run_config_entries={"run.tag": "demo"},
        enable_profiling=True,
    )

    engine = onnx_engine_cls(
        model_path="model.onnx",
        backend="cuda",
        session_option={"custom": "value"},
        provider_option={"do_copy_in_default_stream": False},
        config=config,
    )

    feed = {"input": np.ones((1, 4), dtype=np.float64)}
    outputs = engine(**feed)
    assert set(outputs) == {"output"}
    assert outputs["output"].dtype == np.float32

    summary = engine.summary()
    assert summary["model"] == "model.onnx"
    assert summary["inputs"][0]["shape"][1] is None

    stats = engine.benchmark(feed, repeat=3, warmup=1)
    assert stats["repeat"] == 3
    assert "mean" in stats["latency_ms"]


def test_onnx_engine_benchmark_validates_repeat_and_warmup(onnx_engine_module):
    onnx_engine_cls = onnx_engine_module.ONNXEngine
    engine = onnx_engine_cls(model_path="model.onnx", backend="cpu")
    feed = {"input": np.ones((1, 4), dtype=np.float32)}

    with pytest.raises(ValueError, match="repeat must be >= 1"):
        engine.benchmark(feed, repeat=0)

    with pytest.raises(ValueError, match="warmup must be >= 0"):
        engine.benchmark(feed, warmup=-1)


def test_onnx_engine_accepts_wrapped_feed_dict(onnx_engine_module):
    """__call__ supports passing a single mapping payload as a kwarg."""
    onnx_engine_cls = onnx_engine_module.ONNXEngine
    engine = onnx_engine_cls(model_path="model.onnx", backend="cpu")

    feed = {"input": np.ones((1, 4), dtype=np.float32)}
    outputs = engine(payload=feed)
    assert outputs["output"].shape == (1, 4)


def test_onnx_engine_run_method_uses_mapping(onnx_engine_module):
    """run(feed) is a stable public API for inference."""
    onnx_engine_cls = onnx_engine_module.ONNXEngine
    engine = onnx_engine_cls(model_path="model.onnx", backend="cpu")

    outputs = engine.run({"input": np.ones((1, 4), dtype=np.float32)})
    assert outputs["output"].dtype == np.float32


def test_onnx_engine_iobinding_without_run_options(onnx_engine_module):
    """When run_config_entries is empty, IO binding runs without RunOptions."""
    onnx_engine_cls = onnx_engine_module.ONNXEngine
    engine_config_cls = onnx_engine_module.EngineConfig

    config = engine_config_cls(enable_io_binding=True, run_config_entries=None)
    engine = onnx_engine_cls(
        model_path="model.onnx", backend="cpu", config=config
    )

    out = engine.run({"input": np.ones((1, 4), dtype=np.float32)})
    assert engine._session.last_run_options is None
    assert np.all(out["output"] == 0.0)


def test_onnx_engine_session_option_overrides_attribute(onnx_engine_module):
    """Known SessionOptions attributes are set directly instead of config entries."""
    onnx_engine_cls = onnx_engine_module.ONNXEngine

    engine = onnx_engine_cls(
        model_path="model.onnx",
        backend="cpu",
        session_option={"enable_profiling": True},
    )

    assert engine._session.sess_options.enable_profiling is True


def test_onnx_engine_accepts_enum_config_values(onnx_engine_module):
    """Enum values pass through without string normalization."""
    onnx_engine_cls = onnx_engine_module.ONNXEngine
    engine_config_cls = onnx_engine_module.EngineConfig

    config = engine_config_cls(
        graph_optimization=onnx_engine_module.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        execution_mode=onnx_engine_module.ort.ExecutionMode.ORT_PARALLEL,
    )

    engine = onnx_engine_cls(
        model_path="model.onnx", backend="cpu", config=config
    )
    assert (
        engine._session.sess_options.graph_optimization_level
        == onnx_engine_module.ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    )
    assert (
        engine._session.sess_options.execution_mode
        == onnx_engine_module.ort.ExecutionMode.ORT_PARALLEL
    )


def test_onnx_engine_extract_metadata_parses_json(onnx_engine_module):
    """Custom metadata JSON payloads are parsed for easier downstream use."""
    onnx_engine_cls = onnx_engine_module.ONNXEngine
    onnx_engine_module._test_modelmeta_holder["map"] = {
        "author": '{"name": "bob"}',
        "note": "plain",
        "count": 3,
    }

    engine = onnx_engine_cls(model_path="model.onnx", backend="cpu")

    assert engine.metadata == {
        "author": {"name": "bob"},
        "note": "plain",
        "count": 3,
    }


def test_onnx_engine_run_uses_run_options_without_iobinding(onnx_engine_module):
    """run_config_entries should build RunOptions even when IO binding is disabled."""
    onnx_engine_cls = onnx_engine_module.ONNXEngine
    engine_config_cls = onnx_engine_module.EngineConfig

    config = engine_config_cls(
        enable_io_binding=False, run_config_entries={"k": "v"}
    )
    engine = onnx_engine_cls(
        model_path="model.onnx",
        backend="cpu",
        config=config,
    )

    out = engine.run({"input": np.ones((1, 4), dtype=np.float32)})
    assert out["output"].shape == (1, 4)
    assert engine._session.last_run_options is not None


def test_onnx_engine_converts_outputs_via_toarray(onnx_engine_module):
    """Some onnxruntime outputs expose a toarray() helper instead of ndarray."""
    onnx_engine_cls = onnx_engine_module.ONNXEngine
    engine_config_cls = onnx_engine_module.EngineConfig

    config = engine_config_cls(enable_io_binding=False)
    engine = onnx_engine_cls(
        model_path="model.onnx", backend="cpu", config=config
    )

    class _FakeOrtValue:
        def __init__(self, value: float) -> None:
            self.value = float(value)

        def toarray(self) -> np.ndarray:
            return np.full((1, 4), self.value, dtype=np.float32)

    engine._session.run = lambda *_args, **_kwargs: [_FakeOrtValue(7.0)]  # type: ignore[method-assign]

    out = engine.run({"input": np.ones((1, 4), dtype=np.float32)})
    assert np.all(out["output"] == 7.0)


def test_onnx_engine_missing_required_input_raises_keyerror(onnx_engine_module):
    onnx_engine_cls = onnx_engine_module.ONNXEngine
    engine = onnx_engine_cls(model_path="model.onnx", backend="cpu")

    with pytest.raises(KeyError, match="Missing required input"):
        engine.run({})


def test_onnx_engine_extract_metadata_returns_none_for_non_mapping(
    onnx_engine_module,
):
    onnx_engine_cls = onnx_engine_module.ONNXEngine
    onnx_engine_module._test_modelmeta_holder["map"] = 123

    engine = onnx_engine_cls(model_path="model.onnx", backend="cpu")
    assert engine.metadata is None
