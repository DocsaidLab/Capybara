from __future__ import annotations

import types

import onnx
import pytest
from onnx import TensorProto, helper

from capybara.onnxengine import metadata, utils


@pytest.fixture()
def simple_onnx(tmp_path):
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "test-graph", [input_info], [output_info])
    model = helper.make_model(graph)
    path = tmp_path / "simple.onnx"
    onnx.save(model, path)
    return path


def test_get_input_and_output_infos(simple_onnx):
    inputs = utils.get_onnx_input_infos(simple_onnx)
    outputs = utils.get_onnx_output_infos(simple_onnx)
    assert inputs["input"]["shape"] == [1, 3]
    assert outputs["output"]["shape"] == [1, 3]
    assert str(inputs["input"]["dtype"]) == "float32"


def test_get_input_and_output_infos_accept_model_proto(simple_onnx):
    model = onnx.load(simple_onnx)
    inputs = utils.get_onnx_input_infos(model)
    outputs = utils.get_onnx_output_infos(model)
    assert inputs["input"]["shape"] == [1, 3]
    assert outputs["output"]["shape"] == [1, 3]


def test_make_onnx_dynamic_axes_overrides_dims(
    simple_onnx, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        utils,
        "onnxslim",
        types.SimpleNamespace(simplify=lambda model: (model, True)),
    )
    out_path = tmp_path / "dynamic.onnx"
    utils.make_onnx_dynamic_axes(
        model_fpath=simple_onnx,
        output_fpath=out_path,
        input_dims={"input": {0: "batch"}},
        output_dims={"output": {0: "batch"}},
        opset_version=18,
    )
    model = onnx.load(out_path)
    assert (
        model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    )
    assert (
        model.graph.output[0].type.tensor_type.shape.dim[0].dim_param == "batch"
    )


def test_make_onnx_dynamic_axes_adds_default_opset_when_missing(
    monkeypatch, tmp_path
):
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph(
        [node], "no-default-opset", [input_info], [output_info]
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(domain="ai.onnx.ml", version=1)],
    )

    src = tmp_path / "src.onnx"
    out_path = tmp_path / "dynamic.onnx"
    onnx.save(model, src)

    monkeypatch.setattr(
        utils,
        "onnxslim",
        types.SimpleNamespace(simplify=lambda model: (model, True)),
    )
    utils.make_onnx_dynamic_axes(
        model_fpath=src,
        output_fpath=out_path,
        input_dims={"input": {0: "batch"}},
        output_dims={"output": {0: "batch"}},
        opset_version=18,
    )
    updated = onnx.load(out_path)
    assert any(
        opset.domain == "" and opset.version == 18
        for opset in updated.opset_import
    )


def test_make_onnx_dynamic_axes_uses_current_opset_when_version_is_none(
    monkeypatch, tmp_path
):
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph(
        [node], "no-default-opset", [input_info], [output_info]
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid(domain="ai.onnx.ml", version=1)],
    )

    src = tmp_path / "src.onnx"
    out_path = tmp_path / "dynamic.onnx"
    onnx.save(model, src)

    monkeypatch.setattr(onnx.defs, "onnx_opset_version", lambda: 17)
    monkeypatch.setattr(
        utils,
        "onnxslim",
        types.SimpleNamespace(simplify=lambda model: (model, True)),
    )
    utils.make_onnx_dynamic_axes(
        model_fpath=src,
        output_fpath=out_path,
        input_dims={"input": {0: "batch"}},
        output_dims={"output": {0: "batch"}},
        opset_version=None,
    )
    updated = onnx.load(out_path)
    assert any(
        opset.domain == "" and opset.version == 17
        for opset in updated.opset_import
    )


def test_make_onnx_dynamic_axes_accepts_simplify_returning_model(
    simple_onnx, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        utils,
        "onnxslim",
        types.SimpleNamespace(simplify=lambda model: model),
    )
    out_path = tmp_path / "dynamic.onnx"
    utils.make_onnx_dynamic_axes(
        model_fpath=simple_onnx,
        output_fpath=out_path,
        input_dims={"input": {0: "batch"}},
        output_dims={"output": {0: "batch"}},
        opset_version=18,
    )
    assert (tmp_path / "dynamic.onnx").exists()


def test_make_onnx_dynamic_axes_rejects_reshape_nodes(tmp_path):
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3]
    )
    shape_init = helper.make_tensor(
        "shape", TensorProto.INT64, dims=[2], vals=[1, 3]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 3]
    )
    node = helper.make_node("Reshape", ["input", "shape"], ["output"])
    graph = helper.make_graph(
        [node],
        "reshape-graph",
        [input_info],
        [output_info],
        initializer=[shape_init],
    )
    model = helper.make_model(graph)
    src = tmp_path / "reshape.onnx"
    onnx.save(model, src)

    with pytest.raises(ValueError, match="Reshape cannot be trasformed"):
        utils.make_onnx_dynamic_axes(
            model_fpath=src,
            output_fpath=tmp_path / "out.onnx",
            input_dims={"input": {0: "batch"}},
            output_dims={"output": {0: "batch"}},
            opset_version=18,
        )


@pytest.fixture(autouse=True)
def fake_ort(monkeypatch):
    class FakeSession:
        def __init__(self, path, providers=None):
            self._path = path

        def get_modelmeta(self):
            model = onnx.load(self._path)
            mapping = {p.key: p.value for p in model.metadata_props}
            return types.SimpleNamespace(custom_metadata_map=mapping)

    monkeypatch.setattr(
        metadata, "ort", types.SimpleNamespace(InferenceSession=FakeSession)
    )


def test_metadata_roundtrip(simple_onnx, tmp_path):
    out_path = tmp_path / "meta.onnx"
    metadata.write_metadata_into_onnx(
        simple_onnx, out_path, author={"name": "angizero"}
    )
    parsed = metadata.parse_metadata_from_onnx(out_path)
    assert parsed["author"]["name"] == "angizero"

    raw = metadata.get_onnx_metadata(out_path)
    assert "author" in raw


def test_parse_metadata_preserves_non_string_values(monkeypatch, simple_onnx):
    class FakeSession:
        def __init__(self, path, providers=None):
            self._path = path

        def get_modelmeta(self):
            return types.SimpleNamespace(
                custom_metadata_map={"raw": 123, "json": "1"}
            )

    monkeypatch.setattr(
        metadata, "ort", types.SimpleNamespace(InferenceSession=FakeSession)
    )
    parsed = metadata.parse_metadata_from_onnx(simple_onnx)
    assert parsed["raw"] == 123
    assert parsed["json"] == 1
