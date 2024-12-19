import numpy as np
import pytest

from capybara import Box, Boxes, BoxMode, Keypoints, KeypointsList


def test_invalid_input_type():
    with pytest.raises(TypeError):
        Keypoints("invalid_input")


def test_invalid_input_shape():
    with pytest.raises(ValueError):
        Keypoints([(1, 2, 3, 4), (1, 2, 3, 4)])


def test_normalized_array():
    array = np.array([[0.1, 0.2], [0.3, 0.4]])
    keypoints = Keypoints(array, is_normalized=True)
    assert keypoints.is_normalized is True


def test_keypoints_numpy():
    array = np.array([[1, 2], [3, 4]])
    keypoints = Keypoints(array)
    assert np.allclose(keypoints.numpy(), array), "Keypoints numpy conversion failed."


def test_keypoints_copy():
    keypoints = Keypoints([(1, 2), (3, 4)])
    copied_keypoints = keypoints.copy()
    assert np.allclose(keypoints.numpy(), copied_keypoints.numpy()), "Keypoints copy failed."


def test_keypoints_shift():
    keypoints = Keypoints([(1, 2), (3, 4)])
    shifted_keypoints = keypoints.shift(10, 10)
    assert np.allclose(shifted_keypoints.numpy(), np.array(
        [[11, 12], [13, 14]])), "Keypoints shift failed."


def test_keypoints_scale():
    keypoints = Keypoints([(1, 2), (3, 4)])
    scaled_keypoints = keypoints.scale(10, 10)
    assert np.allclose(scaled_keypoints.numpy(), np.array(
        [[10, 20], [30, 40]])), "Keypoints scale failed."


def test_keypoints_normalize():
    keypoints = Keypoints([(1, 2), (3, 4)])
    normalized_keypoints = keypoints.normalize(100, 100)
    assert np.allclose(normalized_keypoints.numpy(), np.array(
        [[0.01, 0.02], [0.03, 0.04]])), "Keypoints normalization failed."


def test_keypoints_denormalize():
    keypoints = Keypoints([(0.01, 0.02), (0.03, 0.04)], is_normalized=True)
    denormalized_keypoints = keypoints.denormalize(100, 100)
    assert np.allclose(denormalized_keypoints.numpy(), np.array(
        [[1, 2], [3, 4]])), "Keypoints denormalization failed."


def test_keypoints_is_inside_box():
    keypoints = Keypoints([(1, 2), (3, 4)])
    box = Box((0, 0, 10, 10), box_mode=BoxMode.XYXY)
    assert keypoints.is_inside_box(box) == True, "Keypoints is inside box failed."


def test_keypoints_list_numpy():
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    keypoints_list = KeypointsList(array)
    assert np.allclose(keypoints_list.numpy(), array), "KeypointsList numpy conversion failed."


def test_keypoints_list_copy():
    keypoints_list = KeypointsList([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    copied_keypoints_list = keypoints_list.copy()
    assert np.allclose(keypoints_list.numpy(), copied_keypoints_list.numpy()), "KeypointsList copy failed."


def test_keypoints_list_shift():
    keypoints_list = KeypointsList([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    shifted_keypoints_list = keypoints_list.shift(10, 10)
    assert np.allclose(shifted_keypoints_list.numpy(), np.array(
        [[[11, 12], [13, 14]], [[15, 16], [17, 18]]])), "KeypointsList shift failed."


def test_keypoints_list_scale():
    keypoints_list = KeypointsList([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    scaled_keypoints_list = keypoints_list.scale(10, 10)
    assert np.allclose(scaled_keypoints_list.numpy(), np.array(
        [[[10, 20], [30, 40]], [[50, 60], [70, 80]]])), "KeypointsList scale failed."


def test_keypoints_list_normalize():
    keypoints_list = KeypointsList([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    normalized_keypoints_list = keypoints_list.normalize(100, 100)
    assert np.allclose(normalized_keypoints_list.numpy(), np.array(
        [[[0.01, 0.02], [0.03, 0.04]], [[0.05, 0.06], [0.07, 0.08]]])), "KeypointsList normalization failed."


def test_keypoints_list_denormalize():
    keypoints_list = KeypointsList([[(0.01, 0.02), (0.03, 0.04)], [(0.05, 0.06), (0.07, 0.08)]], is_normalized=True)
    denormalized_keypoints_list = keypoints_list.denormalize(100, 100)
    assert np.allclose(denormalized_keypoints_list.numpy(), np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])), "KeypointsList denormalization failed."


def test_keypoints_list_is_inside_boxes():
    keypoints_list = KeypointsList([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    boxes = Boxes([(0, 0, 10, 10), (5, 5, 15, 15)], box_mode=BoxMode.XYXY)
    np.testing.assert_equal(
        keypoints_list.is_inside_boxes(boxes), np.array([False, True]),
        err_msg="KeypointsList is inside boxes failed."
    )


def test_keypoints_list_cat():
    keypoints_list1 = KeypointsList([[(1, 2), (3, 4)]])
    keypoints_list2 = KeypointsList([[(5, 6), (7, 8)]])
    cat_keypoints_list = KeypointsList.cat([keypoints_list1, keypoints_list2])
    assert np.allclose(cat_keypoints_list.numpy(), np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]]])), "KeypointsList concatenation failed."
