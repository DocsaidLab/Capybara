import json
from typing import Union

import onnx
import onnxruntime as ort

from ..utils import Path, now


def get_onnx_metadata(
    onnx_path: Union[str, Path],
) -> dict:
    onnx_path = str(onnx_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    metadata = sess.get_modelmeta().custom_metadata_map
    del sess
    return metadata


def write_metadata_into_onnx(
    onnx_path: Union[str, Path],
    out_path: Union[str, Path],
    drop_old_meta: bool = False,
    **kwargs,
):
    onnx_path = str(onnx_path)
    onnx_model = onnx.load(onnx_path)
    meta_data = parse_metadata_from_onnx(onnx_path) if not drop_old_meta else {}

    meta_data.update({"Date": now(fmt="%Y-%m-%d %H:%M:%S"), **kwargs})

    onnx.helper.set_model_props(
        onnx_model,
        {k: json.dumps(v) for k, v in meta_data.items()},
    )
    onnx.save(onnx_model, out_path)


def parse_metadata_from_onnx(
    onnx_path: Union[str, Path],
) -> dict:
    onnx_path = str(onnx_path)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    metadata = {
        k: json.loads(v) for k, v in sess.get_modelmeta().custom_metadata_map.items()
    }
    del sess
    return metadata
