import builtins
import sys
import types
from typing import Any

import pytest

from capybara import runtime as runtime_module
from capybara.runtime import Backend, Runtime


def test_backend_from_any_respects_runtime_boundaries():
    cuda = Backend.from_any("cuda", runtime="onnx")
    assert cuda.name == "cuda"

    with pytest.raises(ValueError):
        Backend.from_any("cuda", runtime="openvino")


def test_runtime_from_any_accepts_runtime_instances():
    rt = Runtime.onnx
    assert Runtime.from_any(rt) is rt


def test_runtime_normalize_backend_defaults():
    rt = Runtime.from_any("onnx")
    backend = rt.normalize_backend(None)

    assert backend.name == rt.default_backend_name
    assert [b.name for b in rt.available_backends()] == list(rt.backend_names)


def test_runtime_accepts_backend_instances():
    rt = Runtime.from_any("openvino")
    backend = rt.normalize_backend(Backend.ov_gpu)

    assert backend.device == "GPU"
    assert backend.runtime == rt.name


def test_auto_backend_prefers_tensorrt(monkeypatch):
    rt = Runtime.from_any("onnx")
    monkeypatch.setattr(
        "capybara.runtime._get_available_onnx_providers",
        lambda: {
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        },
    )

    # When both TensorRT and CUDA providers are visible, prefer CUDA by default.
    assert rt.auto_backend_name() == "cuda"


def test_auto_backend_prefers_rtx(monkeypatch):
    rt = Runtime.from_any("onnx")
    monkeypatch.setattr(
        "capybara.runtime._get_available_onnx_providers",
        lambda: {"NvTensorRTRTXExecutionProvider"},
    )

    assert rt.auto_backend_name() == "tensorrt_rtx"


def test_auto_backend_falls_back_to_cpu(monkeypatch):
    rt = Runtime.from_any("onnx")
    monkeypatch.setattr(
        "capybara.runtime._get_available_onnx_providers",
        lambda: {"CPUExecutionProvider"},
    )

    assert rt.auto_backend_name() == "cpu"


def test_auto_backend_pt_prefers_cuda(monkeypatch):
    rt = Runtime.from_any("pt")
    monkeypatch.setattr(
        runtime_module,
        "_get_torch_capabilities",
        lambda: (True, True),
    )
    assert rt.auto_backend_name() == "cuda"


def test_auto_backend_pt_defaults_to_cpu(monkeypatch):
    rt = Runtime.from_any("pt")
    monkeypatch.setattr(
        runtime_module,
        "_get_torch_capabilities",
        lambda: (False, False),
    )
    assert rt.auto_backend_name() == rt.default_backend_name


def test_auto_backend_openvino_uses_default_when_no_devices(monkeypatch):
    rt = Runtime.from_any("openvino")

    monkeypatch.setattr(
        runtime_module,
        "_get_openvino_devices",
        lambda: set(),
    )

    assert rt.auto_backend_name() == rt.default_backend_name


def test_auto_backend_openvino_prefers_gpu(monkeypatch):
    rt = Runtime.from_any("openvino")

    monkeypatch.setattr(
        runtime_module,
        "_get_openvino_devices",
        lambda: {"GPU.0", "CPU"},
    )

    assert rt.auto_backend_name() == "gpu"


def test_auto_backend_openvino_prefers_npu(monkeypatch):
    rt = Runtime.from_any("openvino")

    monkeypatch.setattr(
        runtime_module,
        "_get_openvino_devices",
        lambda: {"NPU", "CPU"},
    )

    assert rt.auto_backend_name() == "npu"


def test_auto_backend_returns_default_for_unknown_runtime():
    runtime_key = "custom_auto"
    backend_name = "alpha"
    Backend(name=backend_name, runtime_key=runtime_key)
    try:
        rt = Runtime(
            name=runtime_key,
            backend_names=(backend_name,),
            default_backend_name=backend_name,
        )
        assert rt.auto_backend_name() == rt.default_backend_name
    finally:
        Runtime._REGISTRY.pop(runtime_key, None)
        namespace = Backend._REGISTRY.get(runtime_key, {})
        namespace.pop(backend_name, None)
        if not namespace:
            Backend._REGISTRY.pop(runtime_key, None)


def test_backend_registration_rejects_duplicates():
    runtime_key = "temp_runtime"
    name = "temp_backend"
    Backend(name=name, runtime_key=runtime_key)
    try:
        with pytest.raises(ValueError):
            Backend(name=name, runtime_key=runtime_key)
    finally:
        namespace = Backend._REGISTRY.get(runtime_key, {})
        namespace.pop(name, None)
        if not namespace:
            Backend._REGISTRY.pop(runtime_key, None)


def test_backend_from_any_requires_runtime_when_many_registered():
    with pytest.raises(ValueError):
        Backend.from_any("cpu")


def test_backend_instance_must_match_runtime():
    cuda = Backend.from_any("cuda", runtime="onnx")
    with pytest.raises(ValueError):
        Backend.from_any(cuda, runtime="openvino")


def test_runtime_duplicate_registration_rejected():
    with pytest.raises(ValueError):
        Runtime(
            name="onnx",
            backend_names=("cpu",),
            default_backend_name="cpu",
        )


def test_runtime_unknown_backend_reference():
    with pytest.raises(ValueError):
        Runtime(
            name="ghost",
            backend_names=("missing",),
            default_backend_name="missing",
        )


def test_runtime_default_backend_must_be_known():
    runtime_key = "custom_runtime"
    backend_name = "alpha"
    Backend(name=backend_name, runtime_key=runtime_key)
    try:
        with pytest.raises(ValueError):
            Runtime(
                name=runtime_key,
                backend_names=(backend_name,),
                default_backend_name="beta",
            )
    finally:
        namespace = Backend._REGISTRY.get(runtime_key, {})
        namespace.pop(backend_name, None)
        if not namespace:
            Backend._REGISTRY.pop(runtime_key, None)


def test_get_available_onnx_providers_handles_import_failure(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert runtime_module._get_available_onnx_providers() == set()


def test_get_available_onnx_providers_reads_module(monkeypatch):
    module = types.SimpleNamespace(
        get_available_providers=lambda: [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    try:
        providers = runtime_module._get_available_onnx_providers()
        assert providers == {"CUDAExecutionProvider", "CPUExecutionProvider"}
    finally:
        monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)


def test_get_torch_capabilities_handles_import_failure(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert runtime_module._get_torch_capabilities() == (False, False)


def test_get_torch_capabilities_reads_cuda_available(monkeypatch):
    fake_torch: Any = types.ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    fake_torch.cuda = FakeCuda
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    try:
        assert runtime_module._get_torch_capabilities() == (True, True)
    finally:
        monkeypatch.delitem(sys.modules, "torch", raising=False)


def test_get_openvino_devices_handles_import_failure(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("openvino"):
            raise ModuleNotFoundError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert runtime_module._get_openvino_devices() == set()


def test_get_openvino_devices_reads_core_devices(monkeypatch):
    fake_ov_runtime: Any = types.ModuleType("openvino.runtime")

    class FakeCore:
        available_devices = ("GPU.0", "CPU")

    fake_ov_runtime.Core = FakeCore

    fake_ov: Any = types.ModuleType("openvino")
    fake_ov.runtime = fake_ov_runtime

    monkeypatch.setitem(sys.modules, "openvino", fake_ov)
    monkeypatch.setitem(sys.modules, "openvino.runtime", fake_ov_runtime)
    try:
        assert runtime_module._get_openvino_devices() == {"GPU.0", "CPU"}
    finally:
        monkeypatch.delitem(sys.modules, "openvino.runtime", raising=False)
        monkeypatch.delitem(sys.modules, "openvino", raising=False)


def test_backend_from_any_infers_runtime_when_single(monkeypatch):
    monkeypatch.setattr(Backend, "_REGISTRY", {})
    Backend(name="alpha", runtime_key="solo")
    backend = Backend.from_any("alpha")
    assert backend.runtime == "solo"
