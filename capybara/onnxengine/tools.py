from pathlib import Path
from typing import Dict, List, Optional, Union

import onnx
import onnxruntime as ort
import onnxslim
from onnx.helper import make_graph, make_model, make_opsetid, tensor_dtype_to_np_dtype

from .enum import Backend

__all__ = [
    "get_onnx_input_infos",
    "get_onnx_output_infos",
    "make_onnx_dynamic_axes",
    "get_recommended_backend",
]


def get_onnx_input_infos(model: Union[str, Path, onnx.ModelProto]) -> Dict[str, List[int]]:
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)
    return {
        x.name: {
            "shape": [d.dim_value if d.dim_value != 0 else -1 for d in x.type.tensor_type.shape.dim],
            "dtype": tensor_dtype_to_np_dtype(x.type.tensor_type.elem_type),
        }
        for x in model.graph.input
    }


def get_onnx_output_infos(model: Union[str, Path, onnx.ModelProto]) -> Dict[str, List[int]]:
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)
    return {
        x.name: {
            "shape": [d.dim_value if d.dim_value != 0 else -1 for d in x.type.tensor_type.shape.dim],
            "dtype": tensor_dtype_to_np_dtype(x.type.tensor_type.elem_type),
        }
        for x in model.graph.output
    }


def make_onnx_dynamic_axes(
    model_fpath: Union[str, Path],
    output_fpath: Union[str, Path],
    input_dims: Dict[str, Dict[int, str]],
    output_dims: Dict[str, Dict[int, str]],
    opset_version: Optional[int] = None,
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
        onnx_model.opset_import.append(make_opsetid(domain="", version=opset_version))

    new_model = make_model(new_graph, opset_imports=onnx_model.opset_import, ir_version=onnx_model.ir_version)

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

    new_model = onnxslim.slim(new_model)
    onnx.save(new_model, output_fpath)


def get_recommended_backend() -> Backend:
    providers = ort.get_available_providers()
    device = ort.get_device()
    if "CUDAExecutionProvider" in providers and device == "GPU":
        return Backend.cuda
    elif "CoreMLExecutionProvider" in providers:
        return Backend.coreml
    else:
        return Backend.cpu
