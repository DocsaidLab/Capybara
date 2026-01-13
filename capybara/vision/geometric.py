from typing import Literal, overload

import cv2
import numpy as np

from ..enums import BORDER, INTER, ROTATE
from ..structures import Polygon, Polygons, order_points_clockwise

__all__ = [
    "imresize",
    "imrotate",
    "imrotate90",
    "imwarp_quadrangle",
    "imwarp_quadrangles",
]


@overload
def imresize(
    img: np.ndarray,
    size: tuple[int | None, int | None],
    interpolation: str | int | INTER = INTER.BILINEAR,
    return_scale: Literal[False] = False,
) -> np.ndarray: ...


@overload
def imresize(
    img: np.ndarray,
    size: tuple[int | None, int | None],
    interpolation: str | int | INTER = INTER.BILINEAR,
    return_scale: Literal[True] = True,
) -> tuple[np.ndarray, float, float]: ...


def imresize(
    img: np.ndarray,
    size: tuple[int | None, int | None],
    interpolation: str | int | INTER = INTER.BILINEAR,
    return_scale: bool = False,
) -> np.ndarray | tuple[np.ndarray, float, float]:
    """
    This function is used to resize image.

    Args:
        img (np.ndarray):
            A numpy image.
        size (Tuple[int, int]):
            The size of the resized image. If only one dimension is given,
            calculate the other one maintaining the aspect ratio.
        interpolation (Union[str, int, INTER]):
            Method of interpolation. Default: INTER.BILINEAR.
        return_scale (bool):
            Return scale or not. Default: False.

    Returns:
        img : only resized img or resized img and scale.
    """

    interpolation = INTER.obj_to_enum(interpolation)

    raw_h, raw_w = img.shape[:2]
    h, w = size

    # If only one dimension is given, calculate the other one maintaining
    # the aspect ratio.
    if h is None and w is not None:
        scale = w / raw_w
        h = int(raw_h * scale + 0.5)  # round to nearest integer
    elif w is None and h is not None:
        scale = h / raw_h
        w = int(raw_w * scale + 0.5)  # round to nearest integer

    if h is None or w is None:
        raise ValueError("`size` must provide at least one dimension.")

    resized_img = cv2.resize(img, (w, h), interpolation=interpolation.value)

    if return_scale:
        if "scale" not in locals():  # calculate scale if not already done
            w_scale = w / raw_w
            h_scale = h / raw_h
        else:
            w_scale = h_scale = scale
        return resized_img, w_scale, h_scale
    else:
        return resized_img


def imrotate90(img, rotate_code: ROTATE) -> np.ndarray:
    """
    Function to rotate image with 90-based rotate_code.

    Args:
        img (np.ndarray): Numpy image.
        rotate_code (RotateCode): Rotation code.

    Returns:
        Rotated img (np.ndarray)
    """
    return cv2.rotate(img.copy(), rotate_code)


def imrotate(
    img: np.ndarray,
    angle: float,
    scale: float = 1,
    interpolation: str | int | INTER = INTER.BILINEAR,
    bordertype: str | int | BORDER = BORDER.CONSTANT,
    bordervalue: int | tuple[int, ...] | None = None,
    expand: bool = True,
    center: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Rotate the image by angle.

    Args:
        img (np.ndarray): Image to be rotated.
        angle (float): In degrees clockwise order.
        interpolation (Union[str, int, Interpolation], optional):
            interpolation type, only works as bordertype is not in constant mode.
            Default to Interpolation.BILINEAR.
        bordertype (BorderType, optional):
            border type. Default to BorderType.CONSTANT.
        bordervalue (Union[int, Tuple[int, int, int]], optional):
            border's filling value, only works as bordertype is BorderType.CONSTANT.
            Defaults to None.
        expand (bool, optional):
            Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
            Defaults to False.
        center (Union[Tuple[int], List[int]], optional):
            Optional center of rotation.
            Default is the center of the image (None).

    Returns:
        rotated img: rotated img.
    """
    bordertype = BORDER.obj_to_enum(bordertype)
    interpolation = INTER.obj_to_enum(interpolation)

    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = int(img.shape[-1])
    else:
        raise ValueError("img must be a 2D or 3D numpy image.")

    if bordervalue is None:
        bordervalue = 0
    elif isinstance(bordervalue, int):
        bordervalue = (
            int(bordervalue)
            if channels == 1
            else tuple(int(bordervalue) for _ in range(channels))
        )
    elif isinstance(bordervalue, tuple):
        if channels == 1 and len(bordervalue) == 1:
            bordervalue = int(bordervalue[0])
        elif len(bordervalue) == channels:
            bordervalue = tuple(int(v) for v in bordervalue)
        else:
            raise ValueError(
                f"channel of image is {channels} but bordervalue is {bordervalue}."
            )

    h, w = img.shape[:2]
    center = center or (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle=angle, scale=scale)
    if expand:
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])

        # compute the new bounding dimensions of the image
        new_w = int((h * sin) + (w * cos)) + 1
        new_h = int((h * cos) + (w * sin)) + 1

        # adjust the rotation matrix to take into account translation
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        # perform the actual rotation and return the image
        dst = cv2.warpAffine(
            img,
            matrix,
            (new_w, new_h),
            flags=interpolation.value,
            borderMode=bordertype.value,
            borderValue=bordervalue,
        )
    else:
        dst = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=interpolation.value,
            borderMode=bordertype.value,
            borderValue=bordervalue,
        )

    return dst


def imwarp_quadrangle(
    img: np.ndarray,
    polygon: Polygon | np.ndarray,
    dst_size: tuple[int, int] | None = None,
    do_order_points: bool = True,
) -> np.ndarray:
    """
    Apply a 4-point perspective transform to an image using a given polygon.

    Args:
        img (np.ndarray):
            The input image to be transformed.
        polygon (Union[Polygon, np.ndarray]):
            The polygon object containing the four points defining the transform.
        dst_size (Tuple[int, int], optional):
            The size of the transformed image. If not given, the size is calculated
            from the min area rectangle of the polygon.
            format: (width, height). Default: None.
        do_order_points (bool, optional):
            Order points clockwise or not. Default: True.
            The order of the points should be:
                Top-left, Top-right, Bottom-right, Bottom-left.

    Raises:
        TypeError:
            If img is not a numpy ndarray or polygon is not a Polygon object.
        ValueError:
            If the polygon does not contain exactly four points.

    Returns:
        np.ndarray: The transformed image.
    """
    if isinstance(polygon, np.ndarray):
        polygon = Polygon(polygon)

    if not isinstance(polygon, Polygon):
        raise TypeError(f"Input type of polygon {type(polygon)} not supported.")

    if len(polygon) != 4:
        raise ValueError(
            "Input polygon, which is not contain 4 points is invalid."
        )

    if dst_size is None:
        width, height = polygon.min_box_wh
        if width < height:
            width, height = height, width
    else:
        width, height = dst_size

    width = int(width)
    height = int(height)

    src_pts = polygon.numpy()
    if do_order_points:
        src_pts = order_points_clockwise(src_pts)

    dst_pts = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype="float32"
    )

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (width, height))


def imwarp_quadrangles(
    img: np.ndarray,
    polygons: Polygons,
    dst_size: tuple[int, int] | None = None,
    do_order_points: bool = True,
) -> list[np.ndarray]:
    """
    Apply a 4-point perspective transform to an image using a given polygons.

    Args:
        img (np.ndarray):
            The input image to be transformed.
        polygons (Polygons):
            The polygons object containing the four points defining the transform.
        dst_size (Tuple[int, int], optional):
            The size of the transformed image. If not given, the size is calculated
            from the min area rectangle of the polygon.
            format: (width, height). Default: None.
        do_order_points (bool, optional):
            Order points clockwise or not. Default: True.
            The order of the points should be:
                Top-left, Top-right, Bottom-right, Bottom-left.

    Returns:
        List[np.ndarray]: The transformed image.
    """
    if not isinstance(polygons, Polygons):
        raise TypeError(
            f"Input type of polygons {type(polygons)} not supported."
        )
    return [
        imwarp_quadrangle(
            img, poly, dst_size=dst_size, do_order_points=do_order_points
        )
        for poly in polygons
    ]
