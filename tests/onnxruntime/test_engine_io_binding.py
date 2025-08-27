import numpy as np
import pytest

from capybara import Backend, ONNXEngineIOBinding, get_curdir, get_recommended_backend


@pytest.mark.skipif(get_recommended_backend() != Backend.cuda, reason="Linux with GPU only")
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
