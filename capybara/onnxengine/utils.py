from pathlib import Path
from typing import Any, cast

import onnx
import onnxslim
from onnx.helper import (
    make_graph,
    make_model,
    make_opsetid,
    tensor_dtype_to_np_dtype,
)

__all__ = [
    "get_onnx_input_infos",
    "get_onnx_output_infos",
]


def get_onnx_input_infos(
    model: str | Path | onnx.ModelProto,
) -> dict[str, dict[str, Any]]:
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)

    def _dim_to_value(dim: Any) -> int | str:
        dim_param = getattr(dim, "dim_param", "") or ""
        if dim_param:
            return str(dim_param)
        dim_value = int(getattr(dim, "dim_value", 0) or 0)
        return dim_value if dim_value != 0 else -1

    return {
        x.name: {
            "shape": [_dim_to_value(d) for d in x.type.tensor_type.shape.dim],
            "dtype": tensor_dtype_to_np_dtype(x.type.tensor_type.elem_type),
        }
        for x in model.graph.input
    }


def get_onnx_output_infos(
    model: str | Path | onnx.ModelProto,
) -> dict[str, dict[str, Any]]:
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)

    def _dim_to_value(dim: Any) -> int | str:
        dim_param = getattr(dim, "dim_param", "") or ""
        if dim_param:
            return str(dim_param)
        dim_value = int(getattr(dim, "dim_value", 0) or 0)
        return dim_value if dim_value != 0 else -1

    return {
        x.name: {
            "shape": [_dim_to_value(d) for d in x.type.tensor_type.shape.dim],
            "dtype": tensor_dtype_to_np_dtype(x.type.tensor_type.elem_type),
        }
        for x in model.graph.output
    }


def make_onnx_dynamic_axes(
    model_fpath: str | Path,
    output_fpath: str | Path,
    input_dims: dict[str, dict[int, str]],
    output_dims: dict[str, dict[int, str]],
    opset_version: int | None = None,
) -> None:
    onnx_model = onnx.load(model_fpath)

    new_graph = make_graph(
        nodes=onnx_model.graph.node,
        name=onnx_model.graph.name,
        inputs=onnx_model.graph.input,
        outputs=onnx_model.graph.output,
        initializer=onnx_model.graph.initializer,
        value_info=None,
    )

    if not any(opset.domain == "" for opset in onnx_model.opset_import):
        if opset_version is None:
            opset_version = int(onnx.defs.onnx_opset_version())
        onnx_model.opset_import.append(
            make_opsetid(domain="", version=opset_version)
        )

    new_model = make_model(new_graph, opset_imports=onnx_model.opset_import)

    for x in new_model.graph.input:
        for name, v in input_dims.items():
            if x.name == name:
                for k, d in v.items():
                    x.type.tensor_type.shape.dim[k].dim_param = d

    for x in new_model.graph.output:
        for name, v in output_dims.items():
            if x.name == name:
                for k, d in v.items():
                    x.type.tensor_type.shape.dim[k].dim_param = d

    for x in new_model.graph.node:
        if x.op_type == "Reshape":
            raise ValueError("Reshape cannot be trasformed to dynamic axes")

    simplify = getattr(onnxslim, "simplify", None)
    if callable(simplify):
        simplified = simplify(new_model)
        if isinstance(simplified, tuple):
            new_model = cast(onnx.ModelProto, simplified[0])
        else:
            new_model = cast(onnx.ModelProto, simplified)
    onnx.save(new_model, output_fpath)
