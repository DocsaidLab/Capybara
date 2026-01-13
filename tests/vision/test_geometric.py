from typing import Any

import numpy as np
import pytest

from capybara import (
    BORDER,
    INTER,
    ROTATE,
    Polygon,
    Polygons,
    imresize,
    imrotate,
    imrotate90,
    imwarp_quadrangle,
    imwarp_quadrangles,
)


@pytest.fixture
def random_img():
    """建立 100x100x3 的隨機影像作為測試用。"""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def small_gray_img():
    """
    建立 3x3 的灰階小影像 (數值小, 方便檢查旋轉結果)。
    為了測試方便, 這裡只有一個通道。
    """
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8)


def test_imresize_both_dims(random_img):
    """測試同時給定新尺寸的調整。"""
    resized_img = imresize(random_img, (200, 300), INTER.BILINEAR)
    assert resized_img.shape == (200, 300, 3)


def test_imresize_single_dim(random_img):
    """只給定單一維度時, 檢查是否能等比例縮放。"""
    orig_h, orig_w = random_img.shape[:2]

    # 測試只給定寬度
    resized_img = imresize(random_img, (None, 50), INTER.BILINEAR)
    assert resized_img.shape[1] == 50
    expected_h = int(orig_h * (50 / orig_w) + 0.5)
    assert resized_img.shape[0] == expected_h

    # 測試只給定高度
    resized_img = imresize(random_img, (50, None), INTER.BILINEAR)
    assert resized_img.shape[0] == 50
    expected_w = int(orig_w * (50 / orig_h) + 0.5)
    assert resized_img.shape[1] == expected_w


def test_imresize_rejects_missing_dimensions(random_img):
    with pytest.raises(ValueError, match="at least one dimension"):
        imresize(random_img, (None, None), INTER.BILINEAR)


def test_imresize_return_scale_when_only_one_dim_provided(random_img):
    _orig_h, orig_w = random_img.shape[:2]
    resized_img, w_scale, h_scale = imresize(
        random_img, (None, 50), INTER.BILINEAR, return_scale=True
    )

    assert resized_img.shape[1] == 50
    assert w_scale == pytest.approx(50 / orig_w)
    assert h_scale == pytest.approx(50 / orig_w)


def test_imresize_return_scale(random_img):
    """測試回傳縮放比例。"""
    orig_h, orig_w = random_img.shape[:2]
    new_h, new_w = 120, 240

    resized_img, w_scale, h_scale = imresize(
        random_img, (new_h, new_w), INTER.BILINEAR, return_scale=True
    )

    assert resized_img.shape == (new_h, new_w, 3)
    assert w_scale == pytest.approx(new_w / orig_w)
    assert h_scale == pytest.approx(new_h / orig_h)


def test_imresize_different_interpolation(random_img):
    """測試不同插值方式。"""
    resized_img_nearest = imresize(random_img, (200, 200), INTER.NEAREST)
    assert resized_img_nearest.shape == (200, 200, 3)

    resized_img_bilinear = imresize(random_img, (200, 200), "BILINEAR")
    assert resized_img_bilinear.shape == (200, 200, 3)

    # 其他插值方式可自行增補


@pytest.mark.parametrize(
    "rotate_code, expected",
    [
        (
            ROTATE.ROTATE_90,
            np.array([[7, 4, 1], [8, 5, 2], [9, 6, 3]], dtype=np.uint8),
        ),
        (
            ROTATE.ROTATE_180,
            np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.uint8),
        ),
        (
            ROTATE.ROTATE_270,
            np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]], dtype=np.uint8),
        ),
    ],
)
def test_imrotate90(small_gray_img, rotate_code, expected):
    """測試 90 度基底旋轉。"""
    rotated = imrotate90(small_gray_img, rotate_code)
    assert np.array_equal(rotated, expected)


@pytest.mark.parametrize("angle, expand", [(90, False), (45, False)])
def test_imrotate_no_expand(random_img, angle, expand):
    """測試不擴展的旋轉, 輸出大小應與原圖相同。"""
    rotated_img = imrotate(random_img, angle=angle, expand=expand)
    assert rotated_img.shape == random_img.shape


@pytest.mark.parametrize("angle, expand", [(90, True), (45, True)])
def test_imrotate_expand(random_img, angle, expand):
    """測試擴展旋轉, 輸出大小應大於或等於原圖。"""
    h, w = random_img.shape[:2]
    rotated_img = imrotate(random_img, angle=angle, expand=expand)
    assert rotated_img.shape[0] >= h
    assert rotated_img.shape[1] >= w


def test_imrotate_with_center(random_img):
    """指定旋轉中心, 檢查旋轉結果維度是否符合預期。"""
    h, w = random_img.shape[:2]
    center = (w // 4, h // 4)
    angle = 30
    rotated_img = imrotate(random_img, angle=angle, center=center, expand=True)
    # 只要確定有擴張並且沒有出錯即可
    assert rotated_img.shape[0] >= h
    assert rotated_img.shape[1] >= w


def test_imrotate_scale_border(random_img):
    """
    測試帶有 scale 與 bordervalue 的旋轉。
    例如提供 (255, 0, 0) 以便檢查邊界是否填上紅色。
    """
    scaled_img = imrotate(
        random_img,
        angle=45,
        scale=1.5,
        bordertype=BORDER.REFLECT,
        bordervalue=(255, 0, 0),
    )
    # 只要確定沒有拋出錯誤並且輸出維度放大即可
    assert scaled_img.shape[0] > random_img.shape[0]
    assert scaled_img.shape[1] > random_img.shape[1]


def test_imrotate_invalid_input(random_img):
    """測試不支援的邊界或插值方式。"""
    with pytest.raises(ValueError):
        imrotate(random_img, angle=90, bordertype="invalid_bordertype")

    with pytest.raises(ValueError):
        imrotate(random_img, angle=90, interpolation="invalid_interpolation")


def test_imrotate_validates_image_ndim_and_border_value_tuple_sizes(
    random_img, small_gray_img
):
    rotated_gray = imrotate(
        small_gray_img,
        angle=15,
        expand=False,
        bordertype=BORDER.CONSTANT,
        bordervalue=(7,),
    )
    assert rotated_gray.shape == small_gray_img.shape

    with pytest.raises(ValueError, match="2D or 3D"):
        imrotate(np.zeros((1, 2, 3, 4), dtype=np.uint8), angle=0)

    with pytest.raises(ValueError, match="bordervalue"):
        imrotate(
            random_img,
            angle=10,
            expand=False,
            bordertype=BORDER.CONSTANT,
            bordervalue=(1, 2),
        )


@pytest.fixture
def default_polygon():
    """產生含有 4 個點的基本 Polygon。"""
    pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
    return Polygon(pts)


def test_imwarp_quadrangle_default(random_img, default_polygon):
    """不指定 dst_size 時, 自動使用 min_area_rectangle 所得長寬。"""
    warped = imwarp_quadrangle(random_img, default_polygon)
    # 大致上應該會有 80x80 以上的圖
    assert warped.shape[0] >= 80
    assert warped.shape[1] >= 80


def test_imwarp_quadrangle_with_dstsize(random_img, default_polygon):
    """指定 dst_size, 檢查變換後的輸出是否符合設定大小。"""
    warped = imwarp_quadrangle(random_img, default_polygon, dst_size=(100, 50))
    assert warped.shape == (50, 100, 3)


def test_imwarp_quadrangle_no_order_points(random_img, default_polygon):
    """
    當 do_order_points = False 時, 檢查程式能否正常執行。
    有些情況下, 使用者已確保點位順序正確, 就可以省略排序。
    """
    warped = imwarp_quadrangle(
        random_img, default_polygon, do_order_points=False
    )
    assert warped.shape[0] >= 80
    assert warped.shape[1] >= 80


def test_imwarp_quadrangle_invalid_polygon(random_img):
    """檢查多種不合法 polygon 之行為。"""
    # 傳入不支援的 polygon 類型
    with pytest.raises(TypeError):
        invalid_polygon: Any = "invalid_polygon"
        imwarp_quadrangle(random_img, invalid_polygon)

    # 傳入長度不是 4 的 polygon
    with pytest.raises(ValueError):
        bad_pts = np.array([[10, 10], [90, 10], [90, 90]], dtype=np.float32)
        imwarp_quadrangle(random_img, bad_pts)


def test_imwarp_quadrangle_swaps_width_height_when_needed(random_img):
    pts = np.array([[10, 10], [30, 10], [30, 90], [10, 90]], dtype=np.float32)
    polygon = Polygon(pts)
    warped = imwarp_quadrangle(random_img, polygon)
    assert warped.shape[0] < warped.shape[1]


@pytest.fixture
def polygons_list():
    """產生 2 個四邊形, 合併成 Polygons。"""
    src_pts_1 = np.array(
        [[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32
    )
    src_pts_2 = np.array(
        [[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32
    )
    return Polygons([Polygon(src_pts_1), Polygon(src_pts_2)])


def test_imwarp_quadrangles(random_img, polygons_list):
    """測試多個四邊形的同時透視變換。"""
    warped_list = imwarp_quadrangles(random_img, polygons_list)
    assert len(warped_list) == 2
    # 簡單檢查大小是否合理
    assert warped_list[0].shape[0] >= 80
    assert warped_list[0].shape[1] >= 80
    assert warped_list[1].shape[0] >= 60
    assert warped_list[1].shape[1] >= 60


def test_imwarp_quadrangles_invalid_type(random_img):
    """檢查不合法的 polygons 輸入。"""
    with pytest.raises(TypeError):
        invalid_polygons: Any = "invalid_polygons"
        imwarp_quadrangles(random_img, invalid_polygons)

    with pytest.raises(TypeError):
        invalid_polygons: Any = [
            Polygon(
                np.array(
                    [[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32
                )
            ),
            "invalid_polygon",
        ]
        imwarp_quadrangles(random_img, invalid_polygons)
