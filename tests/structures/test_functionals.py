import numpy as np
import pytest

from capybara import (
    Box,
    Boxes,
    Keypoints,
    Polygon,
    jaccard_index,
    pairwise_ioa,
    pairwise_iou,
    polygon_iou,
)
from capybara.structures.functionals import (
    calc_angle,
    is_inside_box,
    pairwise_intersection,
    poly_angle,
)

test_functionals_error_param = [
    (
        pairwise_iou,
        ([(1, 2, 3, 4)], [(1, 2, 3, 4)]),
        TypeError,
        "Input type of boxes1 and boxes2 must be Boxes",
    ),
    (
        pairwise_iou,
        [Boxes([(1, 1, 0, 2)], "XYWH"), Boxes([(1, 1, 0, 2)], "XYWH")],
        ValueError,
        "Some boxes in Boxes has invaild value",
    ),
    (
        pairwise_ioa,
        ([(1, 2, 3, 4)], [(1, 2, 3, 4)]),
        TypeError,
        "Input type of boxes1 and boxes2 must be Boxes",
    ),
    (
        pairwise_ioa,
        [Boxes([(1, 1, 0, 2)], "XYWH"), Boxes([(1, 1, 0, 2)], "XYWH")],
        ValueError,
        "Some boxes in Boxes has invaild value",
    ),
]


@pytest.mark.parametrize(
    "fn, test_input, error, match", test_functionals_error_param
)
def test_functionals_error(fn, test_input, error, match):
    with pytest.raises(error, match=match):
        fn(*test_input)


test_pairwise_iou_param = [
    (
        Boxes(np.array([[10, 10, 20, 20], [15, 15, 25, 25]]), "XYXY"),
        Boxes(
            np.array([[10, 10, 20, 20], [15, 15, 25, 25], [25, 25, 10, 10]]),
            "XYWH",
        ),
        np.array([[1 / 4, 1 / 28, 0], [1 / 4, 4 / 25, 0]], dtype="float32"),
    )
]


@pytest.mark.parametrize("boxes1, boxes2, expected", test_pairwise_iou_param)
def test_pairwise_iou(boxes1, boxes2, expected):
    assert (pairwise_iou(boxes1, boxes2) == expected).all()


test_pairwise_ioa_param = [
    (
        Boxes(np.array([[10, 10, 20, 20]]), "XYXY"),
        Boxes(np.array([[15, 15, 20, 20], [20, 20, 10, 10]]), "XYWH"),
        np.array([[1 / 16, 0]], dtype="float32"),
    )
]


@pytest.mark.parametrize("boxes1, boxes2, expected", test_pairwise_ioa_param)
def test_pairwise_ioa(boxes1, boxes2, expected):
    assert (pairwise_ioa(boxes1, boxes2) == expected).all()


test_polygon_iou_param = [
    (
        Polygon(np.array([[0, 0], [0, 10], [10, 10], [10, 0]])),
        Polygon(np.array([[5, 5], [5, 15], [15, 15], [15, 5]])),
        25 / 175,
    )
]


@pytest.mark.parametrize("poly1, poly2, expected", test_polygon_iou_param)
def test_polygon_iou(poly1, poly2, expected):
    assert polygon_iou(poly1, poly2) == expected


test_jaccard_index_param = [
    (
        np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
        np.array([[5, 5], [5, 15], [15, 15], [15, 5]]),
        (100, 100),
        25 / 175,
    )
]


@pytest.mark.parametrize(
    "pred_poly, gt_poly, img_size, expected", test_jaccard_index_param
)
def test_jaccard_index(pred_poly, gt_poly, img_size, expected):
    assert jaccard_index(pred_poly, gt_poly, img_size) == expected


test_jaccard_index_error_param = [
    (
        np.array([[0, 0], [0, 10], [10, 10]]),
        np.array([[5, 5], [5, 15], [15, 15], [15, 5]]),
        (100, 100),
        ValueError,
        "Input polygon must be 4-point polygon.",
    ),
    (
        np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
        np.array([[5, 5], [5, 15], [15, 15], [15, 5], [5, 5]]),
        (100, 100),
        ValueError,
        "Input polygon must be 4-point polygon.",
    ),
]


@pytest.mark.parametrize(
    "pred_poly, gt_poly, img_size, error, match", test_jaccard_index_error_param
)
def test_jaccard_index_error(pred_poly, gt_poly, img_size, error, match):
    with pytest.raises(error, match=match):
        jaccard_index(pred_poly, gt_poly, img_size)


def test_pairwise_intersection_rejects_non_boxes():
    with pytest.raises(TypeError, match="must be Boxes"):
        pairwise_intersection([(0, 0, 1, 1)], [(0, 0, 1, 1)])  # type: ignore[arg-type]


def test_jaccard_index_requires_image_size():
    pred = np.zeros((4, 2), dtype=np.float32)
    gt = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="image size"):
        jaccard_index(pred, gt, None)  # type: ignore[arg-type]


def test_jaccard_index_returns_zero_when_shapely_raises(monkeypatch):
    import capybara.structures.functionals as fn_mod

    monkeypatch.setattr(
        fn_mod,
        "ShapelyPolygon",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )
    pred = np.zeros((4, 2), dtype=np.float32)
    gt = np.zeros((4, 2), dtype=np.float32)
    assert jaccard_index(pred, gt, (100, 100)) == 0


def test_jaccard_index_clamps_intersection_area_close_to_min(monkeypatch):
    import capybara.structures.functionals as fn_mod

    monkeypatch.setattr(
        fn_mod.cv2, "getPerspectiveTransform", lambda *_: np.eye(3)
    )
    monkeypatch.setattr(fn_mod.cv2, "perspectiveTransform", lambda pts, _m: pts)

    class _FakePoly:
        def __init__(
            self, *, area: float, intersection_area: float | None = None
        ) -> None:
            self._area = area
            self._intersection_area = intersection_area

        @property
        def area(self) -> float:
            return self._area

        def __and__(self, _other):
            assert self._intersection_area is not None
            return _FakePoly(area=self._intersection_area)

    target = _FakePoly(area=1.0, intersection_area=1.00000000005)
    pred = _FakePoly(area=1.0)
    factory = iter([target, pred])
    monkeypatch.setattr(
        fn_mod, "ShapelyPolygon", lambda *_args, **_kwargs: next(factory)
    )

    pred_poly = np.zeros((4, 2), dtype=np.float32)
    gt_poly = np.zeros((4, 2), dtype=np.float32)
    assert jaccard_index(pred_poly, gt_poly, (10, 10)) == pytest.approx(1.0)


def test_polygon_iou_rejects_non_polygon_and_returns_zero_on_errors(
    monkeypatch,
):
    with pytest.raises(TypeError, match="must be Polygon"):
        polygon_iou("bad", Polygon(np.zeros((4, 2), dtype=np.float32)))  # type: ignore[arg-type]

    import capybara.structures.functionals as fn_mod

    class _FakePoly:
        def __init__(
            self,
            *,
            area: float,
            intersection_area: float | None = None,
            raise_intersection: bool = False,
        ) -> None:
            self._area = area
            self._intersection_area = intersection_area
            self._raise_intersection = raise_intersection

        @property
        def area(self) -> float:
            return self._area

        def intersection(self, _other):
            if self._raise_intersection:
                raise ValueError("boom")
            assert self._intersection_area is not None
            return _FakePoly(area=self._intersection_area)

    poly1_shape = _FakePoly(area=2.0, intersection_area=1.00000000005)
    poly2_shape = _FakePoly(area=1.0)
    factory = iter([poly1_shape, poly2_shape])
    monkeypatch.setattr(
        fn_mod, "ShapelyPolygon", lambda *_args, **_kwargs: next(factory)
    )

    poly1 = Polygon(np.zeros((4, 2), dtype=np.float32))
    poly2 = Polygon(np.zeros((4, 2), dtype=np.float32))
    assert polygon_iou(poly1, poly2) == pytest.approx(0.5)

    poly1_shape_err = _FakePoly(area=1.0, raise_intersection=True)
    poly2_shape_err = _FakePoly(area=1.0)
    factory_err = iter([poly1_shape_err, poly2_shape_err])
    monkeypatch.setattr(
        fn_mod, "ShapelyPolygon", lambda *_args, **_kwargs: next(factory_err)
    )
    assert polygon_iou(poly1, poly2) == 0


def test_is_inside_box_calc_angle_and_poly_angle():
    box = Box((0, 0, 10, 10), box_mode="XYXY")
    assert is_inside_box(Keypoints([(1, 1), (9, 9)]), box)
    assert not is_inside_box(Keypoints([(1, 1), (11, 9)]), box)

    assert calc_angle(np.array([0, 1]), np.array([-1, 0])) == pytest.approx(
        90.0
    )
    assert calc_angle(np.array([0, 1]), np.array([1, 0])) == pytest.approx(
        270.0
    )

    poly1 = Polygon(
        np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
    )
    assert poly_angle(poly1) == pytest.approx(90.0)

    poly2 = Polygon(
        np.array([[0, 0], [1, 0], [0, -1], [1, -1]], dtype=np.float32)
    )
    assert poly_angle(poly1, poly2) == pytest.approx(270.0)
