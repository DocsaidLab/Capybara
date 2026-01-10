from typing import Any

import numpy as np
import pytest

from capybara.structures import (
    Box,
    Boxes,
    Keypoints,
    KeypointsList,
    Polygon,
    Polygons,
)
from capybara.vision.visualization.utils import (
    is_numpy_img,
    prepare_box,
    prepare_boxes,
    prepare_color,
    prepare_colors,
    prepare_img,
    prepare_keypoints,
    prepare_keypoints_list,
    prepare_point,
    prepare_polygon,
    prepare_polygons,
    prepare_scale,
    prepare_scales,
    prepare_thickness,
    prepare_thicknesses,
)


def test_is_numpy_img():
    img = np.random.random((100, 100, 3))
    assert is_numpy_img(img) is True

    img = np.random.random((100, 100))
    assert is_numpy_img(img) is True

    img = np.random.random((100,))
    assert is_numpy_img(img) is False

    img = "not an image"
    assert is_numpy_img(img) is False


def test_prepare_color():
    color = (0, 0, 0)
    assert prepare_color(color) == (0, 0, 0)

    color = [0, 0, 0]
    assert prepare_color(color) == (0, 0, 0)

    color = np.array([0, 0, 0])
    assert prepare_color(color) == (0, 0, 0)

    color = 0
    assert prepare_color(color) == (0, 0, 0)

    color: Any = "black"
    with pytest.raises(TypeError):
        prepare_color(color)

    color: Any = (0.1, 0.1, 0.1)
    with pytest.raises(
        TypeError, match=r"[0-9a-zA-Z=,. ]+colors\[2\][0-9a-zA-Z=,. ()[\]]+"
    ):
        prepare_color(color, 2)


def test_prepare_colors():
    colors = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    length = 3
    assert prepare_colors(colors, length) == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]

    colors = (0, 0, 0)
    length = 3
    assert prepare_colors(colors, length) == [(0, 0, 0), (0, 0, 0), (0, 0, 0)]

    colors = np.array([0, 0, 0])
    length = 3
    assert prepare_colors(colors, length) == [(0, 0, 0), (0, 0, 0), (0, 0, 0)]

    colors = 0
    length = 3
    assert prepare_colors(colors, length) == [(0, 0, 0), (0, 0, 0), (0, 0, 0)]

    colors: Any = "black"
    length = 3
    with pytest.raises(TypeError):
        prepare_colors(colors, length)

    colors = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    length = 2
    with pytest.raises(
        ValueError,
        match=r"The length of colors = 3 is not equal to the length = 2.",
    ):
        prepare_colors(colors, length)

    colors = [(0, 0, 0), (1.1, 1.1, 1.1), (2, 2, 2)]
    length = 3
    with pytest.raises(
        TypeError,
        match=r"[0-9a-zA-Z = , . ]+colors\[1\][0-9a-zA-Z = , . ()[\]]+",
    ):
        prepare_colors(colors, length)

    colors: Any = 0.1
    length = 3
    with pytest.raises(
        TypeError,
        match=r"[0-9a-zA-Z = , . ]+colors\[0\][0-9a-zA-Z = , . ()[\]]+",
    ):
        prepare_colors(colors, length)


def test_prepare_img():
    tgt_img1 = np.random.randint(0, 255, (100, 100, 3), dtype="uint8")
    img = tgt_img1.copy()
    np.testing.assert_allclose(prepare_img(img), tgt_img1)

    img = tgt_img1[..., 0].copy()
    tgt_img2 = np.stack([img, img, img], axis=-1)
    np.testing.assert_allclose(prepare_img(img), tgt_img2)

    img = tgt_img1[..., 0, 0].copy()
    with pytest.raises(ValueError):
        prepare_img(img)

    img = np.random.randint(0, 255, (10, 20, 1), dtype="uint8")
    out = prepare_img(img)
    assert out.shape == (10, 20, 3)
    np.testing.assert_allclose(out[..., 0], img[..., 0])

    img: Any = "not an image"
    with pytest.raises(ValueError):
        prepare_img(img)


def test_prepare_box():
    tgt_box = Box((0, 0, 100, 100), box_mode="XYXY")
    assert prepare_box(tgt_box) == tgt_box

    tgt_box = Box((0, 0, 100, 100), box_mode="XYWH")
    assert prepare_box(tgt_box) == tgt_box

    box = (0, 0, 100, 100)
    assert prepare_box(box) == tgt_box

    box = np.array([0, 0, 100, 100])
    assert prepare_box(box) == tgt_box

    box: Any = 0
    with pytest.raises(
        ValueError, match=r"[0-9a-zA-Z=,. ]+0[0-9a-zA-Z=,. ()[\]]+"
    ):
        prepare_box(box)

    box = (0, 0, 100)
    with pytest.raises(
        ValueError, match=r"[0-9a-zA-Z=,. ]+\(0, 0, 100\)[0-9a-zA-Z=,. ()[\]']+"
    ):
        prepare_box(box)

    box = (0, 0, 100, 100, 100)
    with pytest.raises(
        ValueError,
        match=r"[0-9a-zA-Z=,. ]+\(0, 0, 100, 100, 100\)[0-9a-zA-Z=,. ()[\]']+",
    ):
        prepare_box(box)


def test_prepare_boxes():
    boxes = Boxes([(0, 0, 100, 100), (0, 0, 100, 100)], box_mode="XYXY")
    assert prepare_boxes(boxes) == boxes

    boxes_list = [(0, 0, 100, 100), (0, 0, 100, 100)]
    assert prepare_boxes(boxes_list) == boxes

    np_boxes = np.array(boxes_list)
    assert prepare_boxes(np_boxes) == boxes

    boxes = [
        (0, 1),
    ]
    with pytest.raises(
        ValueError,
        match=r"[0-9a-zA-Z=,. ]+boxes\[0\][0-9a-zA-Z=,. ]+\(0, 1\)[0-9a-zA-Z=,. ()[\]']+",
    ):
        prepare_boxes(boxes)


def test_prepare_keypoints():
    tgt_keypoints = Keypoints([(0, 1), (1, 2), (3, 4)])

    keypoints = [(0, 1), (1, 2), (3, 4)]
    assert prepare_keypoints(keypoints) == tgt_keypoints

    np_keypoints = np.array(keypoints)
    assert prepare_keypoints(np_keypoints) == tgt_keypoints

    assert prepare_keypoints(tgt_keypoints) == tgt_keypoints

    keypoints: Any = [[0, 1], [2, 1], [3, 1]]
    with pytest.raises(
        TypeError,
        match=r"[0-9a-zA-Z=,. ]+\[\[0, 1\], \[2, 1\], \[3, 1\]\][0-9a-zA-Z=,. ()[\]]+",
    ):
        prepare_keypoints(keypoints)


def test_prepare_keypoints_list():
    tgt_keypoints_list = KeypointsList(
        [
            [(0, 1), (1, 2), (3, 4)],
            [(0, 1), (1, 2), (3, 3)],
        ]
    )
    assert prepare_keypoints_list(tgt_keypoints_list) == tgt_keypoints_list

    keypoints_list = [
        [(0, 1), (1, 2), (3, 4)],
        [(0, 1), (1, 2), (3, 3)],
    ]
    assert prepare_keypoints_list(keypoints_list) == tgt_keypoints_list

    np_keypoints_list = np.array(keypoints_list)
    assert prepare_keypoints_list(np_keypoints_list) == tgt_keypoints_list

    keypoints_list = [
        [[0, 1], [1, 2], [3, 4]],
        [(0, 1), (1, 2), (3, 3)],
    ]
    with pytest.raises(
        TypeError,
        match=r"[0-9a-zA-Z=,. ]+keypoints_list\[0\][0-9a-zA-Z=,. ()[\]]+",
    ):
        prepare_keypoints_list(keypoints_list)


def test_prepare_polygon():
    tgt_polygon = Polygon([[0, 1], [1, 2], [3, 4]])

    polygon = [[0, 1], [1, 2], [3, 4]]
    assert prepare_polygon(polygon) == tgt_polygon

    np_polygon = np.array(polygon)
    assert prepare_polygon(np_polygon) == tgt_polygon

    polygon = [[0, 1], [1, 2, 3]]
    with pytest.raises(
        TypeError,
        match=r"[0-9a-zA-Z=,. ]+\[\[0, 1\], \[1, 2, 3\]\][0-9a-zA-Z=,. ()[\]]+",
    ):
        prepare_polygon(polygon)


def test_prepare_polygons():
    tgt_polygons = Polygons(
        [
            [[0, 1], [1, 2], [3, 4]],
            [
                [
                    1,
                    2,
                ],
                [3, 3],
                [5, 5],
            ],
        ]
    )
    polygons = [
        [[0, 1], [1, 2], [3, 4]],
        [[1, 2], [3, 3], [5, 5]],
    ]
    assert prepare_polygons(polygons) == tgt_polygons

    np_polygons = np.array(polygons)
    assert prepare_polygons(np_polygons) == tgt_polygons

    polygons = [
        [[0, 1], [1, 2], [3, 4]],
        [[1, 2], [3, 3], [5, 5, 3]],
    ]
    with pytest.raises(
        TypeError, match=r"[0-9a-zA-Z=,. ]+polygons\[1\][0-9a-zA-Z=,. ()[\]]+"
    ):
        prepare_polygons(polygons)


def test_prepare_thickness():
    tgt_thickness = 1

    thickenss = 1.0
    assert prepare_thickness(thickenss) == tgt_thickness

    np_thickenss = np.array([1.0])[0]
    assert prepare_thickness(np_thickenss) == tgt_thickness


def test_prepare_thicknesses():
    tgt_thicknesses = [1, 1]

    thicknesses = 1
    assert prepare_thicknesses(thicknesses, 2) == tgt_thicknesses

    thicknesses = [1.0, 1.0]
    assert prepare_thicknesses(thicknesses) == tgt_thicknesses

    np_thicknesses = [1.0, 1.0]
    assert prepare_thicknesses(np_thicknesses) == tgt_thicknesses


def test_prepare_scale():
    tgt_scale = 1.5

    scale = 1.5
    assert prepare_scale(scale) == tgt_scale

    np_scale = np.array([1.5])[0]
    assert prepare_scale(np_scale) == tgt_scale


def test_prepare_scales():
    tgt_scales = [1.5, 1.4]

    scales = [1.5, 1.4]
    assert prepare_scales(scales) == tgt_scales

    np_scales = np.array([1.5, 1.4])
    assert prepare_scales(np_scales) == tgt_scales

    scales = 1.2
    assert prepare_scales(scales, 2) == [1.2, 1.2]


def test_prepare_point_and_numeric_validations_cover_error_paths():
    point: Any = (1,)
    with pytest.raises(TypeError, match=r"points\[0\]"):
        prepare_point(point, ind=0)

    with pytest.raises(ValueError, match=r"thickness\[s\[1\]\]"):
        prepare_thickness(-2, ind=1)

    with pytest.raises(ValueError, match=r"scale\[s\[1\]\]"):
        prepare_scale(-2, ind=1)
