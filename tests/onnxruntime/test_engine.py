import platform

import numpy as np
import pytest

from capybara import ONNXEngine, get_curdir


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
def test_ONNXEngine_CUDA():
    model_path = get_curdir(__file__).parent / "resources/model_dynamic-axes.onnx"
    engine = ONNXEngine(model_path, backend="cuda")
    for i in range(5):
        xs = {"input": np.random.randn(32, 3, 224, 224).astype("float32")}
        outs = engine(**xs)
        if i:
            assert not np.allclose(outs["output"], prev_outs["output"])
        prev_outs = outs


def test_ONNXEngine_CPU():
    model_path = get_curdir(__file__).parent / "resources/model_dynamic-axes.onnx"
    engine = ONNXEngine(model_path, backend="cpu")
    for i in range(5):
        xs = {"input": np.random.randn(32, 3, 224, 224).astype("float32")}
        outs = engine(**xs)
        if i:
            assert not np.allclose(outs["output"], prev_outs["output"])
        prev_outs = outs


@pytest.mark.skipif(platform.system() != "Darwin", reason="Mac only")
def test_ONNXEngine_COREML():
    model_path = get_curdir(__file__).parent / "resources/model_dynamic-axes.onnx"
    engine = ONNXEngine(model_path, backend="coreml")
    for i in range(5):
        xs = {"input": np.random.randn(32, 3, 224, 224).astype("float32")}
        outs = engine(**xs)
        if i:
            assert not np.allclose(outs["output"], prev_outs["output"])
        prev_outs = outs
