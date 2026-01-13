from typing import Any

import numpy as np
import pytest

from capybara import Box, Boxes, BoxMode


def test_box_invalid_input_type():
    with pytest.raises(TypeError):
        invalid_input: Any = "invalid_input"
        Box(invalid_input)


def test_box_invalid_input_shape():
    with pytest.raises(TypeError):
        Box([1, 2, 3, 4, 5])  # 長度為5而非4, 不符合預期的box格式


def test_box_accepts_is_normalized_flag():
    array = np.array([0.1, 0.2, 0.3, 0.4])
    box = Box(array, is_normalized=True)
    assert box.is_normalized is True


def test_box_invalid_box_mode():
    with pytest.raises(KeyError):
        array = np.array([1, 2, 3, 4])
        Box(array, box_mode="invalid_mode")


def test_box_array_conversion():
    array = [1, 2, 3, 4]
    box = Box(array)
    assert np.allclose(box._array, np.array(array, dtype="float32"))


# Test Box initialization


def test_box_init():
    # Create a box in XYXY format
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    assert isinstance(box, Box), "Initialization of Box failed."


# Test conversion of Box format


def test_box_convert():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    converted_box = box.convert(BoxMode.XYWH)
    assert np.allclose(converted_box.numpy(), np.array([50, 50, 50, 50])), (
        "Box conversion failed."
    )


# Test calculation of area of Box


def test_box_area():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    assert box.area == 2500, "Box area calculation failed."


# Test Box.copy() method


def test_box_copy():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    copied_box = box.copy()
    assert copied_box is not box and (copied_box._array == box._array).all(), (
        "Box copy failed."
    )


# Test Box conversion to numpy array


def test_box_numpy():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    arr = box.numpy()
    assert isinstance(arr, np.ndarray) and np.allclose(
        arr, np.array([50, 50, 100, 100])
    ), "Box to numpy conversion failed."


# Test Box normalization


def test_box_normalize():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    normalized_box = box.normalize(200, 200)
    assert np.allclose(
        normalized_box.numpy(), np.array([0.25, 0.25, 0.5, 0.5])
    ), "Box normalization failed."


# Test Box denormalization


def test_box_denormalize():
    box = Box((0.25, 0.25, 0.5, 0.5), box_mode=BoxMode.XYXY, is_normalized=True)
    denormalized_box = box.denormalize(200, 200)
    assert np.allclose(
        denormalized_box.numpy(), np.array([50, 50, 100, 100])
    ), "Box denormalization failed."


# Test Box clipping


def test_box_clip():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    clipped_box = box.clip(60, 60, 90, 90)
    assert np.allclose(clipped_box.numpy(), np.array([60, 60, 90, 90])), (
        "Box clipping failed."
    )


def test_box_clip_preserves_box_mode_for_xywh_inputs():
    box = Box((10, 20, 5, 5), box_mode=BoxMode.XYWH)
    clipped_box = box.clip(0, 0, 12, 30)
    assert clipped_box.box_mode == BoxMode.XYWH
    assert np.allclose(clipped_box.numpy(), np.array([10, 20, 2, 5]))


def test_box_convert_preserves_is_normalized_flag():
    box = Box((0.1, 0.2, 0.3, 0.4), box_mode=BoxMode.XYXY, is_normalized=True)
    converted = box.convert(BoxMode.XYWH)
    assert converted.is_normalized is True


# Test Box shifting


def test_box_shift():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    shifted_box = box.shift(10, -10)
    assert np.allclose(shifted_box.numpy(), np.array([60, 40, 110, 90])), (
        "Box shifting failed."
    )


# Test Box scaling


def test_box_scale():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    scaled_box = box.scale(dsize=(20, 0))
    assert np.allclose(scaled_box.numpy(), np.array([40, 50, 110, 100])), (
        "Box scaling failed."
    )


# Test Box to_list


def test_box_to_list():
    box = Box((50, 50, 50, 50), box_mode=BoxMode.XYXY)
    assert box.to_list() == [50, 50, 50, 50], "Boxes tolist failed."


# Test Box to_polygon


def test_box_to_polygon():
    box = Box((50, 50, 100, 100), box_mode=BoxMode.XYXY)
    polygon = box.to_polygon()
    assert np.allclose(
        polygon.numpy(), np.array([[50, 50], [100, 50], [100, 100], [50, 100]])
    ), "Box convert_to_polygon failed."


# Test Boxes initialization


def test_boxes_invalid_input_type():
    with pytest.raises(TypeError):
        invalid_input: Any = "invalid_input"
        Boxes(invalid_input)


def test_boxes_invalid_input_shape():
    with pytest.raises(TypeError):
        Boxes([[1, 2, 3, 4, 5]])


def test_boxes_accepts_is_normalized_flag():
    array = np.array([0.1, 0.2, 0.3, 0.4])
    box = Boxes([array], is_normalized=True)
    assert box.is_normalized is True


def test_boxes_invalid_box_mode():
    with pytest.raises(KeyError):
        array = np.array([1, 2, 3, 4])
        Boxes([array], box_mode="invalid_mode")


def test_boxes_array_conversion():
    array = [[1, 2, 3, 4]]
    box = Boxes(array)
    assert np.allclose(box._array, np.array(array, dtype="float32"))


def test_boxes_init():
    # Create boxes in XYXY format
    boxes = Boxes(
        [(50, 50, 100, 100), [60, 60, 120, 120]], box_mode=BoxMode.XYXY
    )
    assert isinstance(boxes, Boxes), "Initialization of Boxes failed."


# Test conversion of Boxes format


def test_boxes_convert():
    boxes = Boxes(
        [(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY
    )
    converted_boxes = boxes.convert(BoxMode.XYWH)
    assert np.allclose(
        converted_boxes.numpy(), np.array([[50, 50, 50, 50], [60, 60, 60, 60]])
    ), "Boxes conversion failed."


# Test calculation of area of Boxes


def test_boxes_area():
    boxes = Boxes(
        [(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY
    )
    assert np.allclose(boxes.area, np.array([2500, 3600])), (
        "Boxes area calculation failed."
    )


# Test Boxes.copy() method


def test_boxes_copy():
    boxes = Boxes(
        [(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY
    )
    copied_boxes = boxes.copy()
    assert (
        copied_boxes is not boxes
        and (copied_boxes._array == boxes._array).all()
    ), "Boxes copy failed."


# Test Boxes conversion to numpy array


def test_boxes_numpy():
    boxes = Boxes(
        [(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY
    )
    arr = boxes.numpy()
    assert isinstance(arr, np.ndarray) and np.allclose(
        arr, np.array([(50, 50, 100, 100), (60, 60, 120, 120)])
    ), "Boxes to numpy conversion failed."


# Test Boxes normalization


def test_boxes_normalize():
    boxes = Boxes(
        [(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY
    )
    normalized_boxes = boxes.normalize(200, 200)
    assert np.allclose(
        normalized_boxes.numpy(),
        np.array([[0.25, 0.25, 0.5, 0.5], [0.3, 0.3, 0.6, 0.6]]),
    ), "Boxes normalization failed."


# Test Boxes denormalization


def test_boxes_denormalize():
    boxes = Boxes(
        [(0.25, 0.25, 0.5, 0.5), (0.3, 0.3, 0.6, 0.6)],
        box_mode=BoxMode.XYXY,
        is_normalized=True,
    )
    denormalized_boxes = boxes.denormalize(200, 200)
    assert np.allclose(
        denormalized_boxes.numpy(),
        np.array([(50, 50, 100, 100), (60, 60, 120, 120)]),
    ), "Boxes denormalization failed."


# Test Boxes clipping


def test_boxes_clip():
    boxes = Boxes(
        [(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY
    )
    clipped_boxes = boxes.clip(60, 60, 90, 90)
    assert np.allclose(
        clipped_boxes.numpy(), np.array([(60, 60, 90, 90), (60, 60, 90, 90)])
    ), "Boxes clipping failed."


# Test Boxes shifting


def test_boxes_shift():
    boxes = Boxes(
        [(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY
    )
    shifted_boxes = boxes.shift(10, -10)
    assert np.allclose(
        shifted_boxes.numpy(), np.array([(60, 40, 110, 90), (70, 50, 130, 110)])
    ), "Boxes shifting failed."


# Test Boxes scaling


def test_boxes_scale():
    boxes = Boxes(
        [(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY
    )
    scaled_boxes = boxes.scale(dsize=(20, 0))
    assert np.allclose(
        scaled_boxes.numpy(), np.array([(40, 50, 110, 100), (50, 60, 130, 120)])
    ), "Boxes scaling failed."


def test_boxes_scale_supports_fy_with_single_box():
    """Regression: fy scaling should not index the 4th row for small N."""
    boxes = Boxes([(0, 0, 10, 10)], box_mode=BoxMode.XYWH)
    scaled = boxes.scale(fy=2.0).convert(BoxMode.XYWH).numpy()
    np.testing.assert_allclose(
        scaled, np.array([[0, -5, 10, 20]], dtype=np.float32)
    )


# Test Boxes get_empty_index


def test_boxes_get_empty_index():
    boxes = Boxes([(50, 50, 50, 50), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    assert boxes.get_empty_index() == 0, "Boxes get_empty_index failed."


# Test Boxes drop_empty


def test_boxes_drop_empty():
    boxes = Boxes([(50, 50, 50, 50), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    boxes = boxes.drop_empty()
    assert np.allclose(boxes.numpy(), np.array([(60, 60, 120, 120)])), (
        "Boxes drop_empty failed."
    )


# Test Boxes tolist


def test_boxes_tolist():
    boxes = Boxes([(50, 50, 50, 50), (60, 60, 120, 120)], box_mode=BoxMode.XYXY)
    assert boxes.tolist() == [[50, 50, 50, 50], [60, 60, 120, 120]], (
        "Boxes tolist failed."
    )


# Test Boxes to_polygons


def test_boxes_to_polygons():
    boxes = Boxes(
        [(50, 50, 100, 100), (60, 60, 120, 120)], box_mode=BoxMode.XYXY
    )
    polygons = boxes.to_polygons()
    assert np.allclose(
        polygons.numpy(),
        np.array(
            [
                [[50, 50], [100, 50], [100, 100], [50, 100]],
                [[60, 60], [120, 60], [120, 120], [60, 120]],
            ]
        ),
    ), "Boxes convert_to_polygons failed."


def test_boxmode_convert_supports_cxcywh_and_align_code_int_and_invalid():
    xywh = np.array([10, 20, 30, 40], dtype=np.float32)

    cxcywh = BoxMode.convert(xywh, 1, "cxcywh")
    np.testing.assert_allclose(
        cxcywh, np.array([25, 40, 30, 40], dtype=np.float32)
    )

    xyxy = BoxMode.convert(cxcywh, BoxMode.CXCYWH, BoxMode.XYXY)
    np.testing.assert_allclose(
        xyxy, np.array([10, 20, 40, 60], dtype=np.float32)
    )

    back_xywh = BoxMode.convert(xyxy, BoxMode.XYXY, BoxMode.XYWH)
    np.testing.assert_allclose(back_xywh, xywh)

    with pytest.raises(TypeError, match="not int, str, or BoxMode"):
        invalid_box_mode: Any = object()
        BoxMode.align_code(invalid_box_mode)


def test_box_repr_getitem_slice_and_eq_non_box():
    box = Box((1, 2, 3, 4), box_mode=BoxMode.XYXY)
    assert "Box(" in repr(box)

    np.testing.assert_allclose(box[:2], np.array([1, 2], dtype=np.float32))
    assert (box == object()) is False


def test_box_invalid_numpy_array_shape_raises():
    with pytest.raises(TypeError, match="got shape"):
        Box(np.zeros((2, 2), dtype=np.float32))


def test_box_square_warns_on_normalize_denormalize_clip_nan_and_scale_fx_fy():
    box_xywh = Box((0, 0, 10, 5), box_mode=BoxMode.XYWH)
    square = box_xywh.square().convert(BoxMode.XYWH).numpy()
    np.testing.assert_allclose(square, np.array([2.5, 0.0, 5.0, 5.0]))

    with pytest.warns(UserWarning, match="forced to do normalization"):
        _ = Box((0.1, 0.1, 0.2, 0.2), is_normalized=True).normalize(10, 10)

    with pytest.warns(UserWarning, match="forced to do denormalization"):
        _ = Box((1, 2, 3, 4), is_normalized=False).denormalize(10, 10)

    with pytest.raises(ValueError, match="infinite or NaN"):
        Box((np.nan, 0, 1, 1)).clip(0, 0, 10, 10)

    scaled = Box((10, 10, 10, 20), box_mode=BoxMode.XYWH).scale(fx=2.0, fy=0.5)
    np.testing.assert_allclose(
        scaled.convert(BoxMode.XYWH).numpy(), np.array([5, 15, 20, 10])
    )


def test_box_to_polygon_rejects_non_positive_size_and_properties_work():
    with pytest.raises(ValueError, match="invaild value"):
        Box((0, 0, -1, 2), box_mode=BoxMode.XYWH).to_polygon()

    box = Box((10, 20, 30, 40), box_mode=BoxMode.XYWH)
    assert box.width == 30
    assert box.height == 40
    np.testing.assert_allclose(
        box.left_top, np.array([10, 20], dtype=np.float32)
    )
    np.testing.assert_allclose(
        box.right_bottom, np.array([40, 60], dtype=np.float32)
    )
    np.testing.assert_allclose(
        box.left_bottom, np.array([10, 60], dtype=np.float32)
    )
    np.testing.assert_allclose(
        box.right_top, np.array([40, 20], dtype=np.float32)
    )
    assert box.aspect_ratio == 30 / 40
    np.testing.assert_allclose(box.center, np.array([25, 40], dtype=np.float32))


def test_boxes_repr_indexing_eq_and_constructor_from_boxes():
    boxes = Boxes([[0, 0, 10, 10], [20, 20, 30, 30]], box_mode=BoxMode.XYXY)
    assert "Boxes(" in repr(boxes)

    assert isinstance(boxes[[1]], Boxes)
    assert len(boxes[[1]]) == 1
    assert isinstance(boxes[:1], Boxes)
    assert len(boxes[:1]) == 1

    mask = np.array([True, False])
    assert len(boxes[mask]) == 1

    with pytest.raises(TypeError, match="Boxes indices"):
        _ = boxes["0"]  # type: ignore[index]

    assert (boxes == object()) is False

    converted = Boxes(boxes, box_mode=BoxMode.XYWH)
    assert converted.box_mode == BoxMode.XYWH
    assert len(converted) == len(boxes)


def test_boxes_square_warns_clip_nan_scale_fx_and_to_polygons_invalid():
    boxes_xywh = Boxes([[0, 0, 10, 5], [10, 10, 6, 8]], box_mode=BoxMode.XYWH)
    squared = boxes_xywh.square().convert(BoxMode.XYWH).numpy()
    assert np.allclose(squared[:, 2], squared[:, 3])

    with pytest.warns(UserWarning, match="forced to do normalization"):
        _ = Boxes([[0.1, 0.1, 0.2, 0.2]], is_normalized=True).normalize(10, 10)

    with pytest.warns(UserWarning, match="forced to do denormalization"):
        _ = Boxes([[1, 2, 3, 4]], is_normalized=False).denormalize(10, 10)

    with pytest.raises(ValueError, match="infinite or NaN"):
        Boxes([[np.nan, 0, 1, 1]], box_mode=BoxMode.XYXY).clip(0, 0, 10, 10)

    scaled = Boxes([[10, 10, 10, 20]], box_mode=BoxMode.XYWH).scale(fx=2.0)
    np.testing.assert_allclose(
        scaled.convert(BoxMode.XYWH).numpy(), np.array([[5, 10, 20, 20]])
    )

    with pytest.raises(ValueError, match="invaild value"):
        Boxes([[0, 0, -1, 2]], box_mode=BoxMode.XYWH).to_polygons()


def test_boxes_properties_work():
    boxes = Boxes([[10, 20, 30, 40], [0, 0, 2, 4]], box_mode=BoxMode.XYWH)
    np.testing.assert_allclose(boxes.width, np.array([30, 2], dtype=np.float32))
    np.testing.assert_allclose(
        boxes.height, np.array([40, 4], dtype=np.float32)
    )
    np.testing.assert_allclose(
        boxes.left_top, np.array([[10, 20], [0, 0]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        boxes.right_bottom,
        np.array([[40, 60], [2, 4]], dtype=np.float32),
    )
    np.testing.assert_allclose(boxes.aspect_ratio, np.array([0.75, 0.5]))
    np.testing.assert_allclose(
        boxes.center, np.array([[25, 40], [1, 2]], dtype=np.float32)
    )
