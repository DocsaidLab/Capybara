from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import numpy as np

from ...structures import (
    Box,
    Boxes,
    BoxMode,
    Keypoints,
    KeypointsList,
    Polygon,
    Polygons,
)
from ...structures.boxes import _Box, _Boxes, _Number
from ...structures.keypoints import _Keypoints, _KeypointsList
from ...structures.polygons import _Polygon, _Polygons

_Color: TypeAlias = int | Sequence[int] | tuple[int, int, int] | np.ndarray
_Colors: TypeAlias = _Color | Sequence[_Color] | np.ndarray
_Point: TypeAlias = Sequence[int] | tuple[int, int] | np.ndarray
_Points: TypeAlias = Sequence[_Point] | np.ndarray
_Thickness: TypeAlias = _Number
_Thicknesses: TypeAlias = _Thickness | Sequence[_Thickness] | np.ndarray
_Scale: TypeAlias = _Number
_Scales: TypeAlias = _Scale | Sequence[_Scale] | np.ndarray


__all__ = [
    "is_numpy_img",
    "prepare_box",
    "prepare_boxes",
    "prepare_color",
    "prepare_colors",
    "prepare_img",
    "prepare_keypoints",
    "prepare_keypoints_list",
    "prepare_point",
    "prepare_points",
    "prepare_polygon",
    "prepare_polygons",
    "prepare_scale",
    "prepare_scales",
    "prepare_thickness",
    "prepare_thicknesses",
]


def is_numpy_img(x: Any) -> bool:
    if not isinstance(x, np.ndarray):
        return False
    return x.ndim == 2 or (x.ndim == 3 and x.shape[-1] in [1, 3])


def prepare_color(
    color: _Color, ind: int | None = None
) -> tuple[int, int, int]:
    """
    This function prepares the color input for opencv.

    Args:
        color (_Color): unsual color input.
        ind (int, optional): use for iterating preparing. Defaults to None.

    Raises:
        TypeError: If the input color is not a number, tuple, or numpy array.

    Returns:
        Tuple[int, int, int]: a tuple of 3 integers.
    """
    cond1 = isinstance(color, int)
    cond2 = (
        isinstance(color, (tuple, list))
        and len(color) == 3
        and isinstance(color[0], int)
        and isinstance(color[1], int)
        and isinstance(color[2], int)
    )
    cond3 = (
        isinstance(color, np.ndarray) and color.ndim == 1 and len(color) == 3
    )
    if not (cond1 or cond2 or cond3):
        i = "" if ind is None else f"s[{ind}]"
        raise TypeError(
            f"The input color{i} = {color} is invalid. Should be {_Color}"
        )
    c = (color,) * 3 if cond1 else color
    c = tuple(np.array(c, dtype=int).tolist())
    return c


def prepare_colors(
    colors: _Colors, length: int | None = None
) -> list[tuple[int, int, int]]:
    """
    This function prepares the colors input for opencv.

    Args:
        colors (_Colors): unsual color inputs.
        length (int): the length of the colors.

    Returns:
        List[Tuple[int, int, int]]: a list of tuples of 3 integers.
    """
    try:
        c = prepare_color(cast(Any, colors), 0)
    except TypeError:
        if not isinstance(colors, (list, tuple, np.ndarray)):
            raise
        if length is not None and len(colors) != length:
            raise ValueError(
                f"The length of colors = {len(colors)} is not equal to the length = {length}."
            ) from None
        return [prepare_color(color, i) for i, color in enumerate(colors)]
    repeat = 1 if length is None else length
    return [c] * repeat


def prepare_img(img: np.ndarray, ind: int | None = None) -> np.ndarray:
    """
    This function prepares the image input for opencv.

    Args:
        img (np.ndarray): unsual image input.
        ind (Optional[int], optional): use for iterating preparing. Defaults to None.

    Raises:
        ValueError: If the input image is not a valid numpy image.

    Returns:
        np.ndarray: a valid numpy image for opencv.
    """
    if is_numpy_img(img):
        if img.ndim == 2:
            img = img[..., None].repeat(3, axis=-1)
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = img.repeat(3, axis=-1)
    else:
        i = "" if ind is None else f"s[{ind}]"
        raise ValueError(f"The input image{i} is not invalid numpy image.")
    return img


def prepare_box(
    box: _Box,
    ind: int | None = None,
    src_mode: str | BoxMode = BoxMode.XYXY,
    dst_mode: str | BoxMode = BoxMode.XYXY,
) -> Box:
    """
    This function prepares the box input to XYXY format.

    Args:
        box (_Box): unsual box input and not be normalized.
        ind (Optional[int], optional): use for iterating preparing. Defaults to None.
        src_mode (Union[str, BoxMode], optional): Box mode of source box.
        dst_mode (Union[str, BoxMode], optional): Box mode of destination box.

    Returns:
        Box: a valid Box instance.
    """
    try:
        is_normalized = box.is_normalized if isinstance(box, Box) else False
        src_mode = box.box_mode if isinstance(box, Box) else src_mode
        box = Box(box, box_mode=src_mode, is_normalized=is_normalized).convert(
            dst_mode
        )
    except Exception as exc:
        i = "" if ind is None else f"es[{ind}]"
        raise ValueError(
            f"The input box{i} is invalid value = {box}. Should be {_Box}"
        ) from exc
    return box


def prepare_boxes(
    boxes: _Boxes,
    src_mode: str | BoxMode = BoxMode.XYXY,
    dst_mode: str | BoxMode = BoxMode.XYXY,
) -> Boxes:
    """
    This function prepares the boxes input to XYXY format.

    Args:
        boxes (_Boxes): unsual boxes input. not be normalized.
        src_mode (Union[str, BoxMode], optional): Box mode of source boxes.
        dst_mode (Union[str, BoxMode], optional): Box mode of destination boxes.

    Returns:
        Boxes: a valid Boxes instance.
    """
    if isinstance(boxes, Boxes):
        boxes = boxes.convert(dst_mode)
    else:
        boxes = Boxes(
            [
                prepare_box(box, i, src_mode, dst_mode)
                for i, box in enumerate(boxes)
            ]
        )
    return boxes


def prepare_keypoints(
    keypoints: _Keypoints, ind: int | None = None
) -> Keypoints:
    """
    This function prepares the keypoints input.

    Args:
        keypoints (_Keypoints): unsual keypoints input.
        ind (Optional[int], optional): use for iterating preparing. Defaults to None.

    Returns:
        Keypoints: a valid Keypoints instance.
    """
    if isinstance(keypoints, Keypoints):
        return keypoints
    try:
        keypoints = Keypoints(keypoints)
    except Exception as exc:
        i = "" if ind is None else f"_list[{ind}]"
        raise TypeError(
            f"The input keypoints{i} is invalid value = {keypoints}. Should be {_Keypoints}"
        ) from exc
    return keypoints


def prepare_keypoints_list(keypoints_list: _KeypointsList) -> KeypointsList:
    """
    This function prepares the keypoints list input.

    Args:
        keypoints_list (_KeypointsList): unsual keypoints list input.

    Returns:
        KeypointsList: a valid KeypointsList instance.
    """
    if isinstance(keypoints_list, KeypointsList):
        return keypoints_list
    keypoints_list = KeypointsList(
        [
            prepare_keypoints(keypoints, i)
            for i, keypoints in enumerate(keypoints_list)
        ]
    )
    return keypoints_list


def prepare_polygon(polygon: _Polygon, ind: int | None = None) -> Polygon:
    if isinstance(polygon, Polygon):
        return polygon
    try:
        polygon = Polygon(polygon)
    except Exception as exc:
        i = "" if ind is None else f"s[{ind}]"
        raise TypeError(
            f"The input polygon{i} is invalid value = {polygon}. Should be {_Polygon}"
        ) from exc
    return polygon


def prepare_polygons(polygons: _Polygons) -> Polygons:
    if isinstance(polygons, Polygons):
        return polygons
    polygons = Polygons(
        [prepare_polygon(polygon, i) for i, polygon in enumerate(polygons)]
    )
    return polygons


def prepare_point(point: _Point, ind: int | None = None) -> tuple:
    cond1 = (
        isinstance(point, (tuple, list))
        and len(point) == 2
        and isinstance(point[0], _Number)
        and isinstance(point[1], _Number)
    )
    cond2 = (
        isinstance(point, np.ndarray) and point.ndim == 1 and len(point) == 2
    )
    if not (cond1 or cond2):
        i = "" if ind is None else f"s[{ind}]"
        raise TypeError(f"The input point{i} is invalid.")
    return tuple(np.array(point, dtype=int).tolist())


def prepare_points(points: _Points) -> list[_Point]:
    ps = []
    for i, point in enumerate(points):
        ps.append(prepare_point(point, i))
    return ps


def prepare_thickness(thickness: _Thickness, ind: int | None = None) -> int:
    if not isinstance(thickness, _Number) or thickness < -1:
        i = "" if ind is None else f"s[{ind}]"
        raise ValueError(f"The thickness[{i}] = {thickness} is not correct. \n")
    value = np.array(thickness, dtype="int").tolist()
    return int(value)


def prepare_thicknesses(
    thicknesses: _Thicknesses, length: int | None = None
) -> list[int]:
    if isinstance(thicknesses, _Number):
        thickness = prepare_thickness(thicknesses, 0)
        repeat = 1 if length is None else length
        cs = [thickness] * repeat
    else:
        cs = []
        for i, thickness in enumerate(thicknesses):
            cs.append(prepare_thickness(thickness, i))
    return cs


def prepare_scale(scale: _Scale, ind: int | None = None) -> float:
    if not isinstance(scale, _Number) or scale < -1:
        i = "" if ind is None else f"s[{ind}]"
        raise ValueError(f"The scale[{i}] = {scale} is not correct. \n")
    value = np.array(scale, dtype=float).tolist()
    return float(value)


def prepare_scales(scales: _Scales, length: int | None = None) -> list[float]:
    if isinstance(scales, _Number):
        scale = prepare_scale(scales, 0)
        repeat = 1 if length is None else length
        cs = [scale] * repeat
    else:
        cs = []
        for i, scale in enumerate(scales):
            cs.append(prepare_scale(scale, i))
    return cs
