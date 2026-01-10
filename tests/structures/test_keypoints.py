from typing import Any

import numpy as np
import pytest

from capybara import Keypoints, KeypointsList


def test_invalid_input_type():
    with pytest.raises(TypeError):
        invalid_input: Any = "invalid_input"
        Keypoints(invalid_input)


def test_invalid_input_shape():
    with pytest.raises(ValueError):
        invalid_input: Any = [(1, 2, 3, 4), (1, 2, 3, 4)]
        Keypoints(invalid_input)


def test_keypoints_eat_itself():
    keypoints1 = Keypoints([(1, 2), (3, 4)])
    keypoints2 = Keypoints(keypoints1)
    assert np.allclose(keypoints1.numpy(), keypoints2.numpy()), (
        "Keypoints eat itself failed."
    )


def test_normalized_array():
    array = np.array([[0.1, 0.2], [0.3, 0.4]])
    keypoints = Keypoints(array, is_normalized=True)
    assert keypoints.is_normalized is True


def test_keypoints_numpy():
    array = np.array([[1, 2], [3, 4]])
    keypoints = Keypoints(array)
    assert np.allclose(keypoints.numpy(), array), (
        "Keypoints numpy conversion failed."
    )


def test_keypoints_copy():
    keypoints = Keypoints([(1, 2), (3, 4)])
    copied_keypoints = keypoints.copy()
    assert np.allclose(keypoints.numpy(), copied_keypoints.numpy()), (
        "Keypoints copy failed."
    )


def test_keypoints_shift():
    keypoints = Keypoints([(1, 2), (3, 4)])
    shifted_keypoints = keypoints.shift(10, 10)
    assert np.allclose(
        shifted_keypoints.numpy(), np.array([[11, 12], [13, 14]])
    ), "Keypoints shift failed."


def test_keypoints_scale():
    keypoints = Keypoints([(1, 2), (3, 4)])
    scaled_keypoints = keypoints.scale(10, 10)
    assert np.allclose(
        scaled_keypoints.numpy(), np.array([[10, 20], [30, 40]])
    ), "Keypoints scale failed."


def test_keypoints_normalize():
    keypoints = Keypoints([(1, 2), (3, 4)])
    normalized_keypoints = keypoints.normalize(100, 100)
    assert np.allclose(
        normalized_keypoints.numpy(), np.array([[0.01, 0.02], [0.03, 0.04]])
    ), "Keypoints normalization failed."


def test_keypoints_denormalize():
    keypoints = Keypoints([(0.01, 0.02), (0.03, 0.04)], is_normalized=True)
    denormalized_keypoints = keypoints.denormalize(100, 100)
    assert np.allclose(
        denormalized_keypoints.numpy(), np.array([[1, 2], [3, 4]])
    ), "Keypoints denormalization failed."


def test_keypoints_list_empty_input():
    keypoints_list = KeypointsList([])
    np.testing.assert_allclose(
        keypoints_list.numpy(), np.array([], dtype="float32")
    )


def test_keypoints_list_numpy():
    array = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    keypoints_list = KeypointsList(array)
    assert np.allclose(keypoints_list.numpy(), array), (
        "KeypointsList numpy conversion failed."
    )


def test_keypoints_list_copy():
    keypoints_list = KeypointsList([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    copied_keypoints_list = keypoints_list.copy()
    assert np.allclose(keypoints_list.numpy(), copied_keypoints_list.numpy()), (
        "KeypointsList copy failed."
    )


def test_keypoints_list_shift():
    keypoints_list = KeypointsList([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    shifted_keypoints_list = keypoints_list.shift(10, 10)
    assert np.allclose(
        shifted_keypoints_list.numpy(),
        np.array([[[11, 12], [13, 14]], [[15, 16], [17, 18]]]),
    ), "KeypointsList shift failed."


def test_keypoints_list_scale():
    keypoints_list = KeypointsList([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    scaled_keypoints_list = keypoints_list.scale(10, 10)
    assert np.allclose(
        scaled_keypoints_list.numpy(),
        np.array([[[10, 20], [30, 40]], [[50, 60], [70, 80]]]),
    ), "KeypointsList scale failed."


def test_keypoints_list_normalize():
    keypoints_list = KeypointsList([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    normalized_keypoints_list = keypoints_list.normalize(100, 100)
    assert np.allclose(
        normalized_keypoints_list.numpy(),
        np.array([[[0.01, 0.02], [0.03, 0.04]], [[0.05, 0.06], [0.07, 0.08]]]),
    ), "KeypointsList normalization failed."


def test_keypoints_list_denormalize():
    keypoints_list = KeypointsList(
        [[(0.01, 0.02), (0.03, 0.04)], [(0.05, 0.06), (0.07, 0.08)]],
        is_normalized=True,
    )
    denormalized_keypoints_list = keypoints_list.denormalize(100, 100)
    assert np.allclose(
        denormalized_keypoints_list.numpy(),
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    ), "KeypointsList denormalization failed."


def test_keypoints_list_cat():
    keypoints_list1 = KeypointsList([[(1, 2), (3, 4)]])
    keypoints_list2 = KeypointsList([[(5, 6), (7, 8)]])
    cat_keypoints_list = KeypointsList.cat([keypoints_list1, keypoints_list2])
    assert np.allclose(
        cat_keypoints_list.numpy(),
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    ), "KeypointsList concatenation failed."


def test_keypoints_repr_eq_and_validation_and_warnings():
    kpts = Keypoints([(1, 2), (3, 4)])
    assert "Keypoints(" in repr(kpts)
    assert (kpts == object()) is False

    with pytest.raises(ValueError, match="ndim"):
        Keypoints(np.zeros((2, 2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="labels"):
        Keypoints(np.array([[1, 2, 3], [4, 5, -1]], dtype=np.float32))

    with pytest.warns(UserWarning, match="forced to do normalization"):
        Keypoints([(0.1, 0.2)], is_normalized=True).normalize(10, 10)

    with pytest.warns(UserWarning, match="forced to do denormalization"):
        Keypoints([(1, 2)], is_normalized=False).denormalize(10, 10)


def test_keypoints_point_colors_can_be_updated():
    kpts = Keypoints([(1, 2), (3, 4)])
    colors = kpts.point_colors
    assert len(colors) == 2
    assert all(isinstance(c, tuple) and len(c) == 3 for c in colors)

    kpts.point_colors = "viridis"
    colors2 = kpts.point_colors
    assert len(colors2) == 2


def test_keypoints_list_getitem_setitem_repr_eq_and_point_colors():
    kpts_list = KeypointsList([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
    assert isinstance(kpts_list[0], Keypoints)
    assert isinstance(kpts_list[:1], KeypointsList)

    with pytest.raises(TypeError, match="not a keypoint"):
        kpts_list[0] = "bad"  # type: ignore[assignment]

    kpts_list[0] = Keypoints([(9, 10), (11, 12)])
    np.testing.assert_allclose(
        kpts_list[0].numpy(), np.array([[9, 10], [11, 12]], dtype=np.float32)
    )

    assert "KeypointsList(" in repr(kpts_list)
    assert (kpts_list == object()) is False

    assert len(kpts_list.point_colors) == 2
    kpts_list.point_colors = "viridis"
    assert len(kpts_list.point_colors) == 2


def test_keypoints_list_validation_errors_and_warnings_and_empty_point_colors():
    kpts_list = KeypointsList([[(1, 2), (3, 4)]])
    kpts_list2 = KeypointsList(kpts_list)
    np.testing.assert_allclose(kpts_list2.numpy(), kpts_list.numpy())

    with pytest.raises(ValueError, match="ndim"):
        KeypointsList(np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="shape\\[-1\\]"):
        KeypointsList(np.zeros((1, 2, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="labels"):
        KeypointsList(np.array([[[1, 2, 3], [4, 5, 9]]], dtype=np.float32))

    with pytest.warns(UserWarning, match="keypoints_list"):
        KeypointsList([[(0.1, 0.2), (0.3, 0.4)]], is_normalized=True).normalize(
            10, 10
        )

    with pytest.warns(UserWarning, match="forced to do denormalization"):
        KeypointsList([[(1, 2), (3, 4)]], is_normalized=False).denormalize(
            10, 10
        )

    assert KeypointsList([]).point_colors == []


def test_keypoints_list_cat_validation_errors():
    with pytest.raises(TypeError, match="should be a list"):
        KeypointsList.cat("not-a-list")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="is empty"):
        KeypointsList.cat([])

    with pytest.raises(TypeError, match="must be KeypointsList"):
        KeypointsList.cat([KeypointsList([]), "bad"])  # type: ignore[list-item]


def test_keypoints_colormap_falls_back_without_matplotlib(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib":
            raise ModuleNotFoundError("boom")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    kpts = Keypoints([(0, 0), (1, 1)])
    assert len(kpts.point_colors) == 2


def test_keypoints_list_rejects_invalid_types():
    with pytest.raises(TypeError, match="Input array is not"):
        KeypointsList(123)  # type: ignore[arg-type]


def test_keypoints_list_set_point_colors_noops_when_empty():
    keypoints_list = KeypointsList([])
    keypoints_list.set_point_colors("rainbow")
    assert keypoints_list.point_colors == []
