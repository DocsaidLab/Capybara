from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from capybara.torchengine import TorchEngine, TorchEngineConfig
from capybara.torchengine import engine as engine_module


class _DummyContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeDevice:
    def __init__(self, spec: str | _FakeDevice = "cpu"):
        if isinstance(spec, _FakeDevice):
            self.type = spec.type
            self._spec = spec._spec
        else:
            spec_str = str(spec)
            self._spec = spec_str
            self.type = "cuda" if spec_str.startswith("cuda") else "cpu"

    def __str__(self) -> str:
        return self._spec


class _FakeTensor:
    def __init__(self, array: np.ndarray, dtype, device: _FakeDevice):
        self._array = np.asarray(array, dtype=np.float32)
        self.dtype = dtype
        self.device = device

    def to(self, target):
        if isinstance(target, _FakeDevice):
            self.device = target
        elif isinstance(target, str):
            self.device = _FakeDevice(target)
        else:
            self.dtype = target
        return self

    def detach(self):
        return self

    def contiguous(self):
        return self

    def numpy(self) -> np.ndarray:
        return np.array(self._array, copy=True)


class _FakeTorchModel:
    def __init__(self, torch_ref):
        self._torch = torch_ref
        self.dtype = torch_ref.float32
        self.device = _FakeDevice("cpu")
        self.eval_called = False
        self.calls = 0

    def eval(self):
        self.eval_called = True
        return self

    def to(self, device=None, dtype=None):
        if isinstance(device, _FakeDevice):
            self.device = device
        if dtype in (self._torch.float16, self._torch.float32):
            self.dtype = dtype
        return self

    def half(self):
        self.dtype = self._torch.float16
        return self

    def float(self):
        self.dtype = self._torch.float32
        return self

    def __call__(self, *tensors):
        self.calls += 1
        outputs = []
        for idx in range(2):
            arr = np.full(
                (1, 2, 2, 2), fill_value=self.calls + idx, dtype=np.float32
            )
            outputs.append(_FakeTensor(arr, self.dtype, self.device))
        return tuple(outputs)


class _FakeTorchModule:
    def __init__(self):
        class _FakeDType:
            pass

        self.dtype = _FakeDType
        self.float16 = _FakeDType()
        self.float32 = _FakeDType()
        self.cuda = SimpleNamespace(
            synchronize=lambda device: self._sync_calls.append(str(device)),
            is_available=lambda: True,
        )
        self._sync_calls: list[str] = []
        self._loaded_paths: list[str] = []
        self.jit = SimpleNamespace(load=self._load)

    def _load(self, path, map_location=None):
        self._loaded_paths.append(str(path))
        model = _FakeTorchModel(self)
        model.to(map_location)
        return model

    def device(self, spec):
        return _FakeDevice(spec)

    def no_grad(self):
        return _DummyContext()

    def inference_mode(self):
        return _DummyContext()

    def from_numpy(self, array):
        return _FakeTensor(
            np.asarray(array, dtype=np.float32),
            self.float32,
            _FakeDevice("cpu"),
        )

    def is_tensor(self, obj):
        return isinstance(obj, _FakeTensor)


@pytest.fixture
def fake_torch(monkeypatch):
    stub = _FakeTorchModule()
    monkeypatch.setattr(engine_module, "_lazy_import_torch", lambda: stub)
    return stub


def test_lazy_import_torch_prefers_sysmodules(monkeypatch):
    fake = types.ModuleType("torch")
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert engine_module._lazy_import_torch() is fake


def test_torch_engine_formats_outputs(fake_torch, tmp_path):
    model_path = tmp_path / "fake.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(
        model_path,
        device="cuda:1",
        output_names=("feat_s8", "feat_s16"),
    )
    inputs = {
        "image": np.zeros((1, 3, 4, 4), dtype=np.float32),
    }
    outputs = engine.run(inputs)

    assert set(outputs.keys()) == {"feat_s8", "feat_s16"}
    for value in outputs.values():
        assert value.dtype == np.float32
        assert value.shape == (1, 2, 2, 2)

    # Ensure TorchEngine selected the CUDA device and casted dtype.
    assert engine.device.type == "cuda"
    assert engine.dtype == fake_torch.float32
    assert fake_torch._loaded_paths == [str(model_path)]


def test_torch_engine_call_accepts_kwargs_feed(fake_torch, tmp_path):
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")

    engine = TorchEngine(
        model_path,
        device="cpu",
        output_names=("feat_s8", "feat_s16"),
    )
    outputs = engine(image=np.zeros((1, 3, 4, 4), dtype=np.float32))

    assert set(outputs.keys()) == {"feat_s8", "feat_s16"}


def test_torch_engine_auto_dtype_selects_fp16_when_name_and_cuda(
    fake_torch, tmp_path
):
    """Auto dtype picks fp16 for '*fp16*' model names when running on CUDA."""
    model_path = tmp_path / "demo_fp16.pt"
    model_path.write_bytes(b"torchscript")

    engine = TorchEngine(model_path, device="cuda:0")

    assert engine.device.type == "cuda"
    assert engine.dtype == fake_torch.float16
    assert engine._model.dtype == fake_torch.float16


def test_torch_engine_explicit_fp32_dtype(fake_torch, tmp_path):
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")

    config = TorchEngineConfig(dtype="fp32")
    engine = TorchEngine(model_path, device="cpu", config=config)

    assert engine.dtype == fake_torch.float32
    assert engine._model.dtype == fake_torch.float32


def test_torch_engine_prepare_feed_requires_mapping(fake_torch, tmp_path):
    """Run rejects non-mapping feeds to avoid accidental positional ordering bugs."""
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(model_path, device="cpu")

    with pytest.raises(TypeError, match="feed must be a mapping"):
        engine.run(["not", "a", "mapping"])  # type: ignore[arg-type]


def test_torch_engine_output_names_mismatch_raises(fake_torch, tmp_path):
    """Output key schema must match model outputs to prevent silent mislabeling."""
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(model_path, device="cpu", output_names=("only_one",))

    with pytest.raises(ValueError, match="model produced 2 outputs"):
        engine.run({"image": np.zeros((1, 3, 4, 4), dtype=np.float32)})


def test_torch_engine_formats_mapping_and_tensor_outputs(fake_torch, tmp_path):
    """TorchEngine supports dict outputs and single tensor outputs."""
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(model_path, device="cuda:0")

    def _mapping_model(*_tensors):
        return {
            "feat": _FakeTensor(
                np.ones((1, 2, 2, 2)),
                fake_torch.float16,
                cast(_FakeDevice, engine.device),
            )
        }

    engine._model = _mapping_model  # type: ignore[assignment]
    out = engine.run({"image": np.zeros((1, 3, 4, 4), dtype=np.float32)})
    assert set(out) == {"feat"}
    assert out["feat"].dtype == np.float32

    def _tensor_model(*_tensors):
        return _FakeTensor(
            np.ones((1, 2, 2, 2)),
            fake_torch.float16,
            cast(_FakeDevice, engine.device),
        )

    engine._model = _tensor_model  # type: ignore[assignment]
    out = engine.run({"image": np.zeros((1, 3, 4, 4), dtype=np.float32)})
    assert set(out) == {"output"}


def test_torch_engine_benchmark_honors_cuda_sync_override(fake_torch, tmp_path):
    """Benchmark uses synchronize() only when CUDA + cuda_sync is enabled."""
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(model_path, device="cuda:1")

    stats = engine.benchmark(
        {"image": np.zeros((1, 3, 4, 4), dtype=np.float32)},
        repeat=2,
        warmup=1,
        cuda_sync=True,
    )

    assert stats["repeat"] == 2
    assert stats["warmup"] == 1
    assert "latency_ms" in stats
    assert len(fake_torch._sync_calls) == 5


def test_torch_engine_benchmark_validates_repeat_and_warmup(
    fake_torch, tmp_path
):
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(model_path, device="cpu")
    feed = {"image": np.zeros((1, 3, 4, 4), dtype=np.float32)}

    with pytest.raises(ValueError, match="repeat must be >= 1"):
        engine.benchmark(feed, repeat=0)

    with pytest.raises(ValueError, match="warmup must be >= 0"):
        engine.benchmark(feed, warmup=-1)


def test_torch_engine_call_accepts_wrapped_mapping_and_generates_names(
    fake_torch, tmp_path
):
    """__call__ supports passing a mapping payload and auto-generating output names."""
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(model_path, device="cpu")

    outputs = engine(
        payload={"image": np.zeros((1, 3, 4, 4), dtype=np.float32)}
    )
    assert set(outputs) == {"output_0", "output_1"}


def test_torch_engine_multiple_inputs_use_positional_forward(
    fake_torch, tmp_path
):
    """Multiple inputs are forwarded positionally into the TorchScript model."""
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(model_path, device="cpu")

    outputs = engine.run(
        {
            "a": np.zeros((1, 3, 4, 4), dtype=np.float32),
            "b": np.zeros((1, 3, 4, 4), dtype=np.float32),
        }
    )
    assert set(outputs) == {"output_0", "output_1"}


def test_torch_engine_summary_reports_core_fields(fake_torch, tmp_path):
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(model_path, device="cpu")

    summary = engine.summary()
    assert summary["model"] == str(model_path)
    assert summary["device"] == "cpu"


def test_torch_engine_benchmark_respects_config_cuda_sync_default(
    fake_torch, tmp_path
):
    """cuda_sync defaults to config when override is omitted."""
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(
        model_path,
        device="cuda:0",
        config=TorchEngineConfig(cuda_sync=False),
    )

    engine.benchmark(
        {"image": np.zeros((1, 3, 4, 4), dtype=np.float32)},
        repeat=1,
        warmup=0,
    )
    assert fake_torch._sync_calls == []


def test_torch_engine_accepts_preconstructed_tensors(
    fake_torch, monkeypatch, tmp_path
):
    """Existing torch tensors should pass through without from_numpy conversion."""
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(model_path, device="cpu")

    tensor = fake_torch.from_numpy(np.zeros((1, 3, 4, 4), dtype=np.float32))
    monkeypatch.setattr(
        fake_torch,
        "from_numpy",
        lambda *_args, **_kwargs: pytest.fail(
            "from_numpy should not be called"
        ),
    )

    outputs = engine.run({"image": tensor})
    assert set(outputs) == {"output_0", "output_1"}


def test_torch_engine_device_instance_short_circuits_normalization(
    fake_torch, monkeypatch, tmp_path
):
    """Passing an existing torch.device object should be preserved."""
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    monkeypatch.setattr(fake_torch, "device", _FakeDevice)

    device = _FakeDevice("cuda:0")
    engine = TorchEngine(model_path, device=device)
    assert engine.device is device


def test_torch_engine_dtype_string_and_custom_dtype_handling(
    fake_torch, monkeypatch, tmp_path
):
    """dtype supports explicit strings, custom torch.dtype instances, and errors."""
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")

    fake_dtype_type = type("dtype", (), {})
    monkeypatch.setattr(fake_torch, "dtype", fake_dtype_type, raising=False)

    engine = TorchEngine(
        model_path,
        device="cpu",
        config=TorchEngineConfig(dtype="fp16"),
    )
    assert engine.dtype == fake_torch.float16

    custom_dtype = fake_dtype_type()
    engine = TorchEngine(
        model_path,
        device="cpu",
        config=TorchEngineConfig(dtype=custom_dtype),
    )
    assert engine.dtype is custom_dtype

    with pytest.raises(ValueError, match="Unsupported dtype specification"):
        TorchEngine(
            model_path,
            device="cpu",
            config=TorchEngineConfig(dtype="weird"),
        )


def test_torch_engine_rejects_unsupported_outputs(fake_torch, tmp_path):
    """Unexpected model outputs should raise a clear error."""
    model_path = tmp_path / "demo.pt"
    model_path.write_bytes(b"torchscript")
    engine = TorchEngine(model_path, device="cpu")

    engine._model = lambda *_args: 123  # type: ignore[assignment]
    with pytest.raises(TypeError, match="Unsupported TorchScript output"):
        engine.run({"image": np.zeros((1, 3, 4, 4), dtype=np.float32)})

    engine._model = lambda *_args: {"feat": "bad"}  # type: ignore[assignment]
    with pytest.raises(TypeError, match=r"Model outputs must be torch\.Tensor"):
        engine.run({"image": np.zeros((1, 3, 4, 4), dtype=np.float32)})
