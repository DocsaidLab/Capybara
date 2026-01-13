from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def test_pyproject_onnxruntime_gpu_extra_uses_hyphenated_package_name():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    assert "onnxruntime-gpu>=1.22.0,<2" in text
    assert "onnxruntime_gpu>=1.22.0,<2" not in text


def test_get_files_does_not_return_directories_when_suffix_none(tmp_path):
    from capybara.utils.files_utils import get_files

    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b", encoding="utf-8")

    paths = get_files(tmp_path, suffix=None, recursive=True, sort_path=False)
    assert paths
    assert all(Path(p).is_file() for p in paths)


def test_get_onnx_infos_preserve_symbolic_dim_params(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    from capybara.onnxengine import utils

    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, ["batch", 3]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, ["batch", 3]
    )
    node = helper.make_node("Identity", ["input"], ["output"])
    graph = helper.make_graph([node], "sym-graph", [input_info], [output_info])
    model = helper.make_model(graph)
    path = tmp_path / "sym.onnx"
    onnx.save(model, path)

    inputs = utils.get_onnx_input_infos(path)
    outputs = utils.get_onnx_output_infos(path)

    assert inputs["input"]["shape"][0] == "batch"
    assert outputs["output"]["shape"][0] == "batch"


def test_pad_constant_value_fills_all_channels_for_rgba():
    from capybara.vision.functionals import pad

    img = np.zeros((10, 10, 4), dtype=np.uint8)
    out = pad(img, pad_size=1, pad_value=128)
    assert out.shape == (12, 12, 4)
    assert out.dtype == np.uint8
    assert out[0, 0].tolist() == [128, 128, 128, 128]


def test_imrotate_constant_border_fills_all_channels_for_rgba():
    from capybara import BORDER
    from capybara.vision.geometric import imrotate

    img = np.zeros((20, 20, 4), dtype=np.uint8)
    out = imrotate(
        img,
        angle=45,
        expand=False,
        bordertype=BORDER.CONSTANT,
        bordervalue=128,
    )
    assert out.shape == img.shape
    assert out[0, 0].tolist() == [128, 128, 128, 128]


def test_imrotate_preserves_dtype_for_float32_inputs():
    from capybara import BORDER
    from capybara.vision.geometric import imrotate

    img = np.zeros((20, 20, 3), dtype=np.float32)
    out = imrotate(
        img,
        angle=10,
        expand=False,
        bordertype=BORDER.CONSTANT,
        bordervalue=0,
    )
    assert out.dtype == np.float32


def test_visualization_package_import_is_lazy():
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import sys
sys.modules.pop("capybara.vision.visualization.draw", None)
sys.modules.pop("capybara.vision.visualization.utils", None)
sys.modules.pop("capybara.vision.visualization", None)
import capybara.vision.visualization as vis
assert "capybara.vision.visualization.draw" not in sys.modules
assert "capybara.vision.visualization.utils" not in sys.modules
"""
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_draw_text_falls_back_when_font_files_missing(monkeypatch, tmp_path):
    import capybara.vision.visualization.draw as draw_mod

    monkeypatch.setattr(draw_mod, "DEFAULT_FONT_PATH", tmp_path / "missing.ttf")

    img = np.full((40, 160, 3), 255, dtype=np.uint8)
    out = draw_mod.draw_text(
        img.copy(),
        "hello",
        location=(5, 5),
        color=(0, 0, 255),
        text_size=18,
        font_path=tmp_path / "also_missing.ttf",
    )
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_draw_line_handles_zero_length_and_validates_gap():
    import capybara.vision.visualization.draw as draw_mod

    img = np.zeros((40, 40, 3), dtype=np.uint8)
    out = draw_mod.draw_line(
        img.copy(),
        pt1=(10, 10),
        pt2=(10, 10),
        color=(0, 255, 0),
        thickness=2,
        style="line",
        gap=8,
        inplace=False,
    )
    assert out.shape == img.shape
    assert out.sum() > 0

    with pytest.raises(ValueError, match="gap must be > 0"):
        draw_mod.draw_line(img.copy(), (0, 0), (10, 10), gap=0)


def test_draw_mask_minmax_normalize_constant_mask_is_safe():
    import capybara.vision.visualization.draw as draw_mod

    img = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.full((20, 30), 7, dtype=np.uint8)

    with np.errstate(divide="raise", invalid="raise"):
        out = draw_mod.draw_mask(img, mask, min_max_normalize=True)
    assert out.shape == img.shape
