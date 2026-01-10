from typing import Any

import cv2
import numpy as np
import pytest

from capybara import (
    BORDER,
    Box,
    Boxes,
    gaussianblur,
    imbinarize,
    imcropbox,
    imcropboxes,
    imcvtcolor,
    imresize_and_pad_if_need,
    meanblur,
    medianblur,
    pad,
)
from capybara.vision.functionals import centercrop, imadjust


def test_meanblur():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試預設ksize
    blurred_img_default = meanblur(img)
    assert blurred_img_default.shape == img.shape

    # 測試指定ksize
    ksize = (5, 5)
    blurred_img_custom = meanblur(img, ksize=ksize)
    assert blurred_img_custom.shape == img.shape


def test_gaussianblur():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試預設ksize和sigmaX
    blurred_img_default = gaussianblur(img)
    assert blurred_img_default.shape == img.shape

    # 測試指定ksize和sigmaX
    ksize = (7, 7)
    sigma_x = 1
    blurred_img_custom = gaussianblur(img, ksize=ksize, sigma_x=sigma_x)
    assert blurred_img_custom.shape == img.shape


def test_medianblur():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試預設ksize
    blurred_img_default = medianblur(img)
    assert blurred_img_default.shape == img.shape

    # 測試指定ksize
    ksize = 5
    blurred_img_custom = medianblur(img, ksize=ksize)
    assert blurred_img_custom.shape == img.shape


def test_imcvtcolor():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試RGB轉灰階
    gray_img = imcvtcolor(img, "RGB2GRAY")
    assert gray_img.shape == (100, 100)

    # 支援 OpenCV 的 COLOR_ 前綴寫法
    gray_img_prefixed = imcvtcolor(img, "COLOR_RGB2GRAY")
    assert gray_img_prefixed.shape == (100, 100)

    # 支援直接傳入 OpenCV 的 conversion code
    gray_img2 = imcvtcolor(img, cv2.COLOR_RGB2GRAY)
    assert gray_img2.shape == (100, 100)

    # 測試RGB轉BGR
    bgr_img = imcvtcolor(img, "RGB2BGR")
    assert bgr_img.shape == img.shape

    # 測試轉換為不支援的色彩空間
    with pytest.raises(ValueError):
        imcvtcolor(img, "RGB2WWW")  # XYZ為不支援的色彩空間


def test_pad_constant_gray():
    # 測試用的灰階圖片
    img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

    # 測試等值填充
    pad_size = 10
    pad_value = 128
    padded_img = pad(img, pad_size=pad_size, pad_value=pad_value)
    assert padded_img.shape == (
        img.shape[0] + 2 * pad_size,
        img.shape[1] + 2 * pad_size,
    )
    assert np.all(padded_img[:pad_size, :] == pad_value)
    assert np.all(padded_img[-pad_size:, :] == pad_value)
    assert np.all(padded_img[:, :pad_size] == pad_value)
    assert np.all(padded_img[:, -pad_size:] == pad_value)


def test_pad_constant_gray_accepts_singleton_tuple_pad_value():
    img = np.random.randint(0, 256, size=(10, 10), dtype=np.uint8)
    padded_img = pad(img, pad_size=1, pad_value=(123,))
    assert padded_img.shape == (12, 12)
    assert int(padded_img[0, 0]) == 123


def test_pad_constant_color():
    # 測試用的彩色圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試等值填充
    pad_size = 5
    pad_value = (255, 0, 0)  # 紅色
    padded_img = pad(img, pad_size=pad_size, pad_value=pad_value)
    assert padded_img.shape == (
        img.shape[0] + 2 * pad_size,
        img.shape[1] + 2 * pad_size,
        img.shape[2],
    )
    assert np.all(padded_img[:pad_size, :, :] == pad_value)
    assert np.all(padded_img[-pad_size:, :, :] == pad_value)
    assert np.all(padded_img[:, :pad_size, :] == pad_value)
    assert np.all(padded_img[:, -pad_size:, :] == pad_value)


def test_pad_rejects_invalid_pad_value_type_and_invalid_image_ndim():
    img = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="pad_value must be"):
        pad(img, pad_size=1, pad_value=[1, 2, 3])  # type: ignore[arg-type]

    bad = np.zeros((1, 2, 3, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="img must be a 2D or 3D"):
        pad(bad, pad_size=1, pad_value=0)


def test_pad_replicate():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試邊緣複製填充
    pad_size = (5, 10)
    padded_img = pad(img, pad_size=pad_size, pad_mode=BORDER.REPLICATE)
    assert padded_img.shape == (
        img.shape[0] + 2 * pad_size[0],
        img.shape[1] + 2 * pad_size[1],
        img.shape[2],
    )


def test_pad_reflect():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試邊緣反射填充
    pad_size = (0, 10, 15, 5)
    padded_img = pad(img, pad_size=pad_size, pad_mode=BORDER.REFLECT)
    assert padded_img.shape == (
        img.shape[0] + pad_size[0] + pad_size[1],
        img.shape[1] + pad_size[2] + pad_size[3],
        img.shape[2],
    )


def test_pad_invalid_input():
    # 測試不支援的填充模式
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        pad(img, pad_size=5, pad_mode="invalid_mode")

    # 測試不合法的填充大小
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        pad_size: Any = (10, 20, 30)
        pad(img, pad_size=pad_size)


def test_imcropbox_custom_box():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用自定義Box物件進行裁剪
    box = Box([20, 30, 80, 60])
    cropped_img = imcropbox(img, box)
    assert cropped_img.shape == (30, 60, 3)


def test_imcropbox_numpy_array():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用NumPy陣列進行裁剪
    box = np.array([20, 30, 80, 60])
    cropped_img = imcropbox(img, box)
    assert cropped_img.shape == (30, 60, 3)


def test_imcropbox_outside_boundary():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 裁剪超出圖片範圍的區域
    box = Box([90, 90, 120, 120])
    cropped_img = imcropbox(img, box)
    assert cropped_img.shape == (10, 10, 3)


def test_imcropbox_invalid_input():
    # 測試不支援的裁剪區域
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        invalid_box: Any = "invalid_box"
        imcropbox(img, invalid_box)

    # 測試不合法的裁剪區域
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        imcropbox(img, np.array([10, 20, 30]))  # 需要4個座標值


def test_imcropbox_padding():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 測試裁剪超出邊界的情況
    box = Box([10, 20, 150, 120])
    cropped_img = imcropbox(img, box)
    assert cropped_img.shape == (80, 90, 3)

    # 測試裁剪超出邊界並進行填充的情況
    box = Box([10, 20, 150, 120])
    cropped_img = imcropbox(img, box, use_pad=True)
    assert cropped_img.shape == (100, 140, 3)


def test_imcropboxes_custom_boxes():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用自定義Boxes物件進行多個裁剪
    boxes = Boxes([Box([10, 20, 80, 60]), Box([30, 40, 90, 70])])
    cropped_images = imcropboxes(img, boxes)

    assert len(cropped_images) == 2
    for cropped_img in cropped_images:
        assert cropped_img.shape[0] <= 60 - 20
        assert cropped_img.shape[1] <= 80 - 10


def test_imcropboxes_numpy_array():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用NumPy陣列進行多個裁剪
    boxes = np.array([[10, 20, 80, 60], [30, 40, 90, 70]])
    cropped_images = imcropboxes(img, boxes)

    assert len(cropped_images) == 2
    for cropped_img in cropped_images:
        assert cropped_img.shape[0] <= 60 - 20
        assert cropped_img.shape[1] <= 80 - 10


def test_imcropboxes_use_pad():
    # 測試用的圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用填充方式進行多個裁剪
    boxes = np.array([[10, 20, 150, 60], [30, 40, 90, 170]])
    cropped_images = imcropboxes(img, boxes, use_pad=True)

    assert len(cropped_images) == 2
    assert cropped_images[0].shape[0] == 60 - 20
    assert cropped_images[0].shape[1] == 150 - 10
    assert cropped_images[1].shape[0] == 170 - 40
    assert cropped_images[1].shape[1] == 90 - 30


def test_imcropboxes_invalid_input():
    # 測試不支援的裁剪區域
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        invalid_boxes: Any = "invalid_boxes"
        imcropboxes(img, invalid_boxes)

    # 測試不合法的裁剪區域
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    with pytest.raises(TypeError):
        imcropboxes(img, np.array([[10, 20, 30]]))  # 需要4個座標值


def test_imbinarize_gray_image():
    # 測試用的灰度圖片
    img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

    # 使用THRESH_BINARY進行二值化
    binarized_img = imbinarize(img, threth=cv2.THRESH_BINARY)
    assert binarized_img.shape == img.shape
    assert np.unique(binarized_img).tolist() == [0, 255]

    # 使用THRESH_BINARY_INV進行二值化
    binarized_img = imbinarize(img, threth=cv2.THRESH_BINARY_INV)
    assert binarized_img.shape == img.shape
    assert np.unique(binarized_img).tolist() == [0, 255]


def test_imbinarize_color_image():
    # 測試用的彩色圖片
    img = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    # 使用THRESH_BINARY進行二值化
    binarized_img = imbinarize(img, threth=cv2.THRESH_BINARY)
    assert binarized_img.shape == img.shape[:-1]
    assert np.unique(binarized_img).tolist() == [0, 255]

    # 使用THRESH_BINARY_INV進行二值化
    binarized_img = imbinarize(img, threth=cv2.THRESH_BINARY_INV)
    assert binarized_img.shape == img.shape[:-1]
    assert np.unique(binarized_img).tolist() == [0, 255]


def test_imbinarize_invalid_input():
    # 測試不支援的圖片維度
    img = np.random.randint(0, 256, size=(100, 100, 100), dtype=np.uint8)
    with pytest.raises(cv2.error):
        imbinarize(img)


def test_imresize_and_pad_if_need():
    img = np.ones((150, 120, 3), dtype="uint8")
    processed = imresize_and_pad_if_need(img, 150, 150)
    np.testing.assert_allclose(
        processed[:, 120:], np.zeros((150, 30, 3), dtype="uint8")
    )

    img = np.ones((151, 119, 3), dtype="uint8")
    processed = imresize_and_pad_if_need(img, 150, 150)
    np.testing.assert_allclose(
        processed[:, 120:], np.zeros((150, 30, 3), dtype="uint8")
    )

    img = np.ones((200, 100, 3), dtype="uint8")
    processed = imresize_and_pad_if_need(img, 100, 100)
    np.testing.assert_allclose(
        processed[:, 50:], np.zeros((100, 50, 3), dtype="uint8")
    )

    img = np.ones((20, 20, 3), dtype="uint8")
    processed = imresize_and_pad_if_need(img, 100, 100)
    np.testing.assert_allclose(processed, np.ones((100, 100, 3), dtype="uint8"))


def test_blur_accepts_numpy_scalar_ksize_and_rejects_invalid_ksize():
    img = np.random.randint(0, 256, size=(20, 20, 3), dtype=np.uint8)
    out = meanblur(img, ksize=np.array(3))
    assert out.shape == img.shape
    out2 = gaussianblur(img, ksize=np.array(5))
    assert out2.shape == img.shape

    with pytest.raises(TypeError, match="ksize"):
        meanblur(img, ksize=(1, 2, 3))  # type: ignore[arg-type]


def test_pad_accepts_none_pad_value_and_validates_pad_value_types():
    img = np.zeros((3, 3, 3), dtype=np.uint8)
    padded = pad(img, pad_size=1, pad_value=None)
    assert padded.shape == (5, 5, 3)
    assert np.all(padded[0] == 0)

    with pytest.raises(ValueError, match="pad_value"):
        invalid_pad_value: Any = (1, 2)
        pad(img, pad_size=1, pad_value=invalid_pad_value)

    gray = np.zeros((3, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="must be an int"):
        pad(gray, pad_size=1, pad_value=(1, 2, 3))


def test_centercrop_produces_square_crop():
    img = np.zeros((10, 20, 3), dtype=np.uint8)
    cropped = centercrop(img)
    assert cropped.shape == (10, 10, 3)


def test_imadjust_returns_input_when_bounds_degenerate():
    img = np.zeros((20, 20), dtype=np.uint8)
    out = imadjust(img)
    assert np.array_equal(out, img)


def test_imadjust_stretches_grayscale_and_color_images():
    gray = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    out = imadjust(gray)
    assert out.shape == gray.shape
    assert out.dtype == np.uint8
    assert out.min() == 0
    assert out.max() == 255
    assert not np.array_equal(out, gray)

    bgr = np.stack([gray] * 3, axis=-1)
    out2 = imadjust(bgr)
    assert out2.shape == bgr.shape
    assert out2.dtype == np.uint8


def test_imresize_and_pad_if_need_can_return_scale():
    img = np.ones((200, 100, 3), dtype=np.uint8)
    out, scale = imresize_and_pad_if_need(img, 100, 100, return_scale=True)
    assert out.shape == (100, 100, 3)
    assert scale == pytest.approx(0.5)
