from pathlib import Path

import numpy as np
import pytest

from capybara import get_curdir, imread, imwrite

DIR = get_curdir(__file__)


def test_imread():
    # 測試圖片路徑
    image_path = DIR.parent / "resources" / "lena.png"

    # 測試 BGR 格式的圖片讀取
    img_bgr = imread(image_path, color_base="BGR")
    assert isinstance(img_bgr, np.ndarray)
    assert img_bgr.shape[-1] == 3  # BGR圖片的channel數為3

    # 測試灰階格式的圖片讀取
    img_gray = imread(image_path, color_base="GRAY")
    assert isinstance(img_gray, np.ndarray)
    assert len(img_gray.shape) == 2  # 灰階圖片的channel數為1

    # color_base should be case-insensitive
    img_gray2 = imread(image_path, color_base="gray")
    assert isinstance(img_gray2, np.ndarray)
    assert len(img_gray2.shape) == 2

    # 測試heif格式的圖片讀取
    img_heif = imread(DIR.parent / "resources" / "lena.heic", color_base="BGR")
    assert isinstance(img_heif, np.ndarray)
    assert img_heif.shape[-1] == 3  # BGR圖片的channel數為3

    # 測試不存在的圖片路徑
    with pytest.raises(FileExistsError):
        imread("non_existent_image.jpg")


def test_imwrite(tmp_path):
    # 測試用的圖片
    img = np.zeros((100, 100, 3), dtype=np.uint8)  # 建立一個全黑的BGR圖片

    # 測試BGR格式的圖片寫入
    temp_file_path = tmp_path / "temp_image.jpg"
    assert imwrite(img, path=temp_file_path, color_base="BGR")
    assert Path(temp_file_path).exists()

    # 測試不指定路徑時的圖片寫入 (不應污染 repo working directory)
    assert imwrite(img, color_base="BGR")


def test_imwrite_without_path_does_not_create_tmp_file_in_cwd(
    tmp_path, monkeypatch
):
    """Regression: historical implementation wrote `tmp{suffix}` into CWD."""
    monkeypatch.chdir(tmp_path)
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert imwrite(img, color_base="BGR", suffix=".jpg") is True
    assert not (tmp_path / "tmp.jpg").exists()
