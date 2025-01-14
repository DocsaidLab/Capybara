import numpy as np

from capybara import ONNXEngine, ONNXEngineIOBinding, Timer


def test_ONNXEngineIOBinding():
    model_path = "tests/resources/model.onnx"
    input_initializer = {'inputs': np.random.randn(32, 3, 640, 640).astype('float32')}
    engine = ONNXEngineIOBinding(model_path, input_initializer)
    for i in range(30):
        xs = {'inputs': np.random.randn(32, 3, 640, 640).astype('float32')}
        outs = engine(**xs)
        if i:
            assert not np.allclose(outs['outputs'], prev_outs['outputs'])
        prev_outs = outs
