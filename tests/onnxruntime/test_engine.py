import numpy as np

from capybara import ONNXEngine, get_curdir


def test_ONNXEngine():
    model_path = get_curdir(__file__).parent / "resources/model_dynamic-axes.onnx"
    engine = ONNXEngine(model_path, backend="cuda")
    for i in range(5):
        xs = {"input": np.random.randn(32, 3, 224, 224).astype("float32")}
        outs = engine(**xs)
        if i:
            assert not np.allclose(outs["output"], prev_outs["output"])
        prev_outs = outs
