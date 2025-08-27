import numpy as np
import pytest

from capybara import Backend, ONNXEngine, get_curdir, get_recommended_backend


def test_ONNXEngine_CPU():
    model_path = get_curdir(__file__).parent / "resources/model_dynamic-axes.onnx"
    engine = ONNXEngine(model_path, backend="cpu")
    for i in range(5):
        xs = {"input": np.random.randn(32, 3, 224, 224).astype("float32")}
        outs = engine(**xs)
        if i:
            assert not np.allclose(outs["output"], prev_outs["output"])
        prev_outs = outs


@pytest.mark.skipif(get_recommended_backend() != Backend.cuda, reason="Linux with GPU only")
def test_ONNXEngine_CUDA():
    model_path = get_curdir(__file__).parent / "resources/model_dynamic-axes.onnx"
    engine = ONNXEngine(model_path, backend=get_recommended_backend())
    for i in range(5):
        xs = {"input": np.random.randn(32, 3, 224, 224).astype("float32")}
        outs = engine(**xs)
        if i:
            assert not np.allclose(outs["output"], prev_outs["output"])
        prev_outs = outs


@pytest.mark.skipif(get_recommended_backend() != "Darwin", reason="Mac only")
def test_ONNXEngine_COREML():
    model_path = get_curdir(__file__).parent / "resources/model_dynamic-axes.onnx"
    engine = ONNXEngine(model_path, backend="coreml")
    for i in range(5):
        xs = {"input": np.random.randn(32, 3, 224, 224).astype("float32")}
        outs = engine(**xs)
        if i:
            assert not np.allclose(outs["output"], prev_outs["output"])
        prev_outs = outs
