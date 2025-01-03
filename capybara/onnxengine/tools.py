from pathlib import Path
from typing import Dict, List, Union

import onnx
from onnx.helper import tensor_dtype_to_np_dtype

__all__ = [
    'get_onnx_input_infos',
    'get_onnx_output_infos',
]


def get_onnx_input_infos(model: Union[str, Path, onnx.ModelProto]) -> Dict[str, List[int]]:
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)
    return {
        x.name: {
            'shape': [d.dim_value if d.dim_value != 0 else -1 for d in x.type.tensor_type.shape.dim],
            'dtype': tensor_dtype_to_np_dtype(x.type.tensor_type.elem_type)
        }
        for x in model.graph.input
    }


def get_onnx_output_infos(model: Union[str, Path, onnx.ModelProto]) -> Dict[str, List[int]]:
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)
    return {
        x.name: {
            'shape': [d.dim_value if d.dim_value != 0 else -1 for d in x.type.tensor_type.shape.dim],
            'dtype': tensor_dtype_to_np_dtype(x.type.tensor_type.elem_type)
        }
        for x in model.graph.output
    }
