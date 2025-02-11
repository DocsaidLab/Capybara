import numpy as np

from capybara import (
    ONNXEngineIOBinding,
    get_onnx_input_infos,
    get_onnx_output_infos,
    make_onnx_dynamic_axes,
)


def test_get_onnx_input_infos():
    model_path = "tests/resources/model_shape=224x224.onnx"
    input_infos = get_onnx_input_infos(model_path)
    assert input_infos == {"input": {"shape": [1, 3, 224, 224], "dtype": "float32"}}


def test_get_onnx_output_infos():
    model_path = "tests/resources/model_shape=224x224.onnx"
    output_infos = get_onnx_output_infos(model_path)
    assert output_infos == {"output": {"shape": [1, 64, 56, 56], "dtype": "float32"}}


def test_make_onnx_dynamic_axes():
    model_path = "tests/resources/model_shape=224x224.onnx"
    input_infos = get_onnx_input_infos(model_path)
    output_infos = get_onnx_output_infos(model_path)
    input_dims = {k: {0: "b", 2: "h", 3: "w"} for k in input_infos.keys()}
    output_dims = {k: {0: "b", 2: "h", 3: "w"} for k in output_infos.keys()}
    new_model_path = "/tmp/model_dynamic-axes.onnx"
    make_onnx_dynamic_axes(
        model_path,
        new_model_path,
        input_dims=input_dims,
        output_dims=output_dims,
    )
    xs = {"input": np.random.randn(32, 3, 320, 320).astype("float32")}
    engine = ONNXEngineIOBinding(
        new_model_path, input_initializer=xs, session_option={"log_severity_level": 1}
    )
    outs = engine(**xs)
    assert outs["output"].shape == (32, 64, 80, 80)
