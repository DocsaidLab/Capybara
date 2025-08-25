import platform

import numpy as np
import pytest

from capybara import ONNXEngineIOBinding, get_curdir


@pytest.mark.skipif(platform.system() != "Linux", reason="Linux only")
def test_ONNXEngineIOBinding_CUDAonly():
    model_path = get_curdir(__file__).parent / "resources/model_dynamic-axes.onnx"
    input_initializer = {"input": np.random.randn(32, 3, 448, 448).astype("float32")}
    engine = ONNXEngineIOBinding(model_path, input_initializer)
    for i in range(5):
        xs = {"input": np.random.randn(32, 3, 448, 448).astype("float32")}
        outs = engine(**xs)
        if i:
            assert not np.allclose(outs["output"], prev_outs["output"])
        prev_outs = outs
