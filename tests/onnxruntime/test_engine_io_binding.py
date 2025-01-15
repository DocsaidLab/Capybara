import numpy as np

from capybara import ONNXEngineIOBinding, get_curdir


def test_ONNXEngineIOBinding():
    model_path = get_curdir(__file__).parent / "resources/model_dynamic-axes.onnx"
    input_initializer = {'input': np.random.randn(32, 3, 448, 448).astype('float32')}
    engine = ONNXEngineIOBinding(model_path, input_initializer)
    for i in range(30):
        xs = {'input': np.random.randn(32, 3, 448, 448).astype('float32')}
        outs = engine(**xs)
        if i:
            assert not np.allclose(outs['output'], prev_outs['output'])
        prev_outs = outs
