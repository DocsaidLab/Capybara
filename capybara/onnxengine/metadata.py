from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import onnx

from ..utils.time import now

try:  # pragma: no cover - optional dependency
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - handled lazily
    ort = None  # type: ignore[assignment]


def _require_ort():
    if ort is None:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "onnxruntime is required to read/write ONNX metadata. "
            "Install 'onnxruntime' or 'onnxruntime-gpu'."
        )
    return ort


def get_onnx_metadata(onnx_path: str | Path) -> dict[str, Any]:
    ort_mod = _require_ort()
    sess = ort_mod.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    try:
        custom = sess.get_modelmeta().custom_metadata_map
        return dict(custom)
    finally:
        del sess


def write_metadata_into_onnx(
    onnx_path: str | Path,
    out_path: str | Path,
    drop_old_meta: bool = False,
    **kwargs: Any,
) -> None:
    onnx_model = onnx.load(str(onnx_path))
    meta_data: dict[str, Any] = (
        {} if drop_old_meta else parse_metadata_from_onnx(onnx_path)
    )

    meta_data.update({"Date": now(fmt="%Y-%m-%d %H:%M:%S"), **kwargs})

    onnx.helper.set_model_props(
        onnx_model,
        {str(k): json.dumps(v) for k, v in meta_data.items()},
    )
    onnx.save(onnx_model, str(out_path))


def parse_metadata_from_onnx(onnx_path: str | Path) -> dict[str, Any]:
    ort_mod = _require_ort()
    sess = ort_mod.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    try:
        metadata_map = sess.get_modelmeta().custom_metadata_map
        parsed: dict[str, Any] = {}
        for key, raw in metadata_map.items():
            if isinstance(raw, str):
                parsed[key] = json.loads(raw)
            else:
                parsed[key] = raw
        return parsed
    finally:
        del sess
