import numpy as np
import pytest

from capybara import Box, Boxes, Keypoints, KeypointsList, Polygon, Polygons
from capybara.vision.visualization import draw as draw_mod


def test_draw_box_denormalizes_normalized_box():
    img = np.zeros((60, 80, 3), dtype=np.uint8)
    box = Box([0.1, 0.2, 0.9, 0.8], box_mode="XYXY", is_normalized=True)
    out = draw_mod.draw_box(img.copy(), box, color=(0, 255, 0), thickness=1)
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    # (x1, y1) should be denormalized to a non-zero coordinate.
    assert out[12, 8].sum() > 0


def test_draw_boxes_supports_per_box_colors_and_thicknesses():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    boxes = Boxes([[0, 0, 10, 10], [20, 20, 35, 35]], box_mode="XYXY")
    out = draw_mod.draw_boxes(
        img.copy(),
        boxes,
        colors=[(255, 0, 0), (0, 255, 0)],
        thicknesses=[1, 2],
    )
    assert out.shape == img.shape
    assert out.sum() > 0


def test_draw_polygon_and_polygons_support_fillup_and_normalized_coords():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    poly = Polygon(
        [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
        is_normalized=True,
    )
    out_edges = draw_mod.draw_polygon(img.copy(), poly, fillup=False)
    out_fill = draw_mod.draw_polygon(img.copy(), poly, fillup=True)
    assert out_edges.shape == img.shape
    assert out_fill.shape == img.shape
    assert out_edges.sum() > 0
    assert out_fill.sum() > 0

    polys = Polygons([poly, poly.shift(0.0, -0.1)], is_normalized=True)
    out_multi = draw_mod.draw_polygons(
        img.copy(),
        polys,
        colors=[(0, 0, 255), (255, 255, 0)],
        thicknesses=[1, 1],
        fillup=False,
    )
    assert out_multi.shape == img.shape
    assert out_multi.sum() > 0


def test_draw_text_draws_pixels():
    img = np.full((60, 200, 3), 255, dtype=np.uint8)
    out = draw_mod.draw_text(
        img.copy(),
        "hello",
        location=(5, 5),
        color=(0, 0, 255),
        text_size=18,
    )
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert not np.array_equal(out, img)


def test_draw_text_handles_fonts_without_getbbox(monkeypatch):
    from PIL import ImageFont

    class _NoBBoxFont:
        def __init__(self):
            self._font = ImageFont.load_default()

        def getbbox(self, _text):
            raise RuntimeError("boom")

        def __getattr__(self, name: str):
            return getattr(self._font, name)

    monkeypatch.setattr(
        draw_mod, "_load_font", lambda *_args, **_kwargs: _NoBBoxFont()
    )

    img = np.full((40, 160, 3), 255, dtype=np.uint8)
    out = draw_mod.draw_text(img.copy(), "hello", location=(5, 5))
    assert out.shape == img.shape


@pytest.mark.parametrize("style", ["dotted", "line"])
def test_draw_line_supports_styles(style: str):
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    out = draw_mod.draw_line(
        img,
        pt1=(0, 0),
        pt2=(39, 39),
        color=(0, 255, 0),
        thickness=2,
        style=style,
        gap=8,
        inplace=False,
    )
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_draw_line_inplace_and_invalid_style():
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    out = draw_mod.draw_line(
        img,
        pt1=(0, 0),
        pt2=(19, 0),
        color=(255, 0, 0),
        thickness=1,
        style="line",
        gap=5,
        inplace=True,
    )
    assert out is img
    assert out.sum() > 0

    with pytest.raises(ValueError, match="Unknown style"):
        draw_mod.draw_line(img, (0, 0), (10, 10), style="invalid")


def test_draw_point_and_draw_points_preserve_grayscale_shape():
    gray = np.zeros((30, 30), dtype=np.uint8)
    out = draw_mod.draw_point(
        gray.copy(),
        (15, 15),
        scale=1.0,
        color=(255, 0, 0),
        thickness=-1,
    )
    assert out.shape == gray.shape
    assert out.ndim == 2
    assert out.sum() > 0

    out2 = draw_mod.draw_points(
        gray.copy(),
        points=[(5, 5), (25, 25)],
        scales=[1.0, 2.0],
        colors=[(0, 255, 0), (0, 0, 255)],
        thicknesses=[-1, -1],
    )
    assert out2.shape == (*gray.shape, 3)
    assert out2.ndim == 3
    assert out2.sum() > 0


def test_draw_keypoints_and_list_support_normalized_inputs():
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    kpts = Keypoints([(0.25, 0.25), (0.75, 0.75)], is_normalized=True)
    out = draw_mod.draw_keypoints(img.copy(), kpts, scale=1.0, thickness=-1)
    assert out.shape == img.shape
    assert out.sum() > 0

    kpts_list = KeypointsList([kpts, kpts.shift(0.0, -0.1)], is_normalized=True)
    out2 = draw_mod.draw_keypoints_list(
        img.copy(), kpts_list, scales=[1.0, 1.5], thicknesses=[-1, -1]
    )
    assert out2.shape == img.shape
    assert out2.sum() > 0


def test_generate_colors_and_distinct_color():
    np.random.seed(0)
    tri = draw_mod.generate_colors(3, scheme="triadic")
    assert len(tri) == 3
    assert all(isinstance(c, tuple) and len(c) == 3 for c in tri)

    hsv = draw_mod.generate_colors(3, scheme="hsv")
    assert len(hsv) == 3

    analogous = draw_mod.generate_colors(3, scheme="analogous")
    assert len(analogous) == 3

    square = draw_mod.generate_colors(3, scheme="square")
    assert len(square) == 3

    unknown = draw_mod.generate_colors(2, scheme="not-a-scheme")
    assert unknown == []

    assert 0 <= draw_mod._vdc(1, base=2) < 1
    c0 = draw_mod.distinct_color(0)
    c1 = draw_mod.distinct_color(1)
    assert isinstance(c0, tuple) and len(c0) == 3
    assert c0 != c1

    assert draw_mod._label_to_index("123") == 123
    assert draw_mod._label_to_index("cat") == draw_mod._label_to_index("cat")


def test_draw_mask_normalization_and_shape_checks():
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    mask = np.arange(20 * 30, dtype=np.uint8).reshape(20, 30)
    out = draw_mod.draw_mask(img, mask, min_max_normalize=True)
    assert out.shape == img.shape

    mask_bgr = np.stack([mask] * 3, axis=-1)
    out2 = draw_mod.draw_mask(img, mask_bgr, min_max_normalize=False)
    assert out2.shape == img.shape

    bad_mask = np.zeros((20, 30, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="Mask must be either 2D"):
        draw_mod.draw_mask(img, bad_mask)


def test_draw_detection_and_draw_detections_end_to_end():
    img = np.zeros((80, 120, 3), dtype=np.uint8)
    box = Box([0.05, 0.0, 0.5, 0.2], box_mode="XYXY", is_normalized=True)

    out = draw_mod.draw_detection(
        img.copy(),
        box,
        label="cat",
        score=0.9,
        color=None,
        thickness=None,
        box_alpha=0.5,
        text_bg_alpha=0.5,
    )
    assert out.shape == img.shape
    assert out.sum() > 0

    boxes = Boxes([[0, 0, 20, 20], [30, 30, 60, 60]], box_mode="XYXY")
    with pytest.raises(ValueError, match="Number of boxes must match"):
        draw_mod.draw_detections(img, boxes, labels=["only-one"])
    with pytest.raises(ValueError, match="Number of scores must match"):
        draw_mod.draw_detections(img, boxes, labels=["a", "b"], scores=[0.1])

    out2 = draw_mod.draw_detections(
        img.copy(),
        boxes,
        labels=["a", "b"],
        scores=[0.1, 0.2],
        colors=[(0, 255, 0), (255, 0, 0)],
        thicknesses=[1, 2],
        text_colors=[(255, 255, 255), (0, 0, 0)],
        text_sizes=[12, 13],
        box_alpha=1.0,
        text_bg_alpha=0.5,
    )
    assert out2.shape == img.shape
    assert out2.sum() > 0
