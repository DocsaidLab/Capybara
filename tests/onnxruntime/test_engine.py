import numpy as np

from capybara import ONNXEngine


def test_ONNXEngine():
    model_path = "tests/resources/model.onnx"
    engine = ONNXEngine(model_path, backend='cuda')
    for i in range(5):
        xs = {'inputs': np.random.randn(32, 3, 640, 640).astype('float32')}
        outs = engine(**xs)
        if i:
            assert not np.allclose(outs['outputs'], prev_outs['outputs'])
        prev_outs = outs
