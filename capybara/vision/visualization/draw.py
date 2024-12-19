from pathlib import Path
from typing import List, Tuple, Union

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ...structures import Polygons
from ...structures.boxes import _Box, _Boxes
from ...structures.keypoints import _Keypoints, _KeypointsList
from ...structures.polygons import _Polygon, _Polygons
from ...utils import download_from_docsaid, get_curdir
from ..geometric import imresize
from .utils import (_Color, _Colors, _Point, _Points, _Scale, _Scales,
                    _Thickness, _Thicknesses, prepare_box, prepare_boxes,
                    prepare_color, prepare_colors, prepare_img,
                    prepare_keypoints, prepare_keypoints_list, prepare_point,
                    prepare_points, prepare_polygon, prepare_polygons,
                    prepare_scale, prepare_scales, prepare_thickness,
                    prepare_thicknesses)

__all__ = [
    'draw_box', 'draw_boxes', 'draw_polygon', 'draw_polygons', 'draw_text',
    'generate_colors', 'draw_ocr_infos', 'draw_mask',
    # 'draw_point', 'draw_points',
    # 'draw_keypoints', 'draw_keypoints_list',
]

DIR = get_curdir(__file__)

if not (font_path := DIR / "NotoSansMonoCJKtc-VF.ttf").exists():
    file_id = "DiSqmnitNm3ZTPj"
    download_from_docsaid(file_id, font_path.name, str(font_path))


def draw_box(
    img: np.ndarray,
    box: _Box,
    color: _Color = (0, 255, 0),
    thickness: _Thickness = 2
) -> np.ndarray:
    """
    Draws a bounding box on the image.

    Args:
        img (np.ndarray):
            The image to draw on, as a numpy ndarray.
        box (_Box):
            The bounding box to draw, either as a Box object or as a numpy
            ndarray of the form [x1, y1, x2, y2].
        color (_Color, optional):
            The color of the box to draw. Defaults to (0, 255, 0).
        thickness (_Thickness, optional):
            The thickness of the box lines to draw. Defaults to 2.

    Returns:
        np.ndarray: The image with the drawn box, as a numpy ndarray.
    """
    img = prepare_img(img)
    box = prepare_box(box)
    color = prepare_color(color)
    thickness = prepare_thickness(thickness)
    if box.is_normalized:
        h, w = img.shape[:2]
        box = box.denormalize(w, h)
    x1, y1, x2, y2 = box.numpy().astype(int).tolist()
    return cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)


def draw_boxes(
    img: np.ndarray,
    boxes: _Boxes,
    colors: _Colors = (0, 255, 0),
    thicknesses: _Thicknesses = 2
) -> np.ndarray:
    """
    Draws multiple bounding boxes on the image.

    Args:
        img (np.ndarray):
            The image to draw on, as a numpy ndarray.
        boxes (_Boxes):
            The bounding boxes to draw, either as a list of Box objects or as a
            2D numpy ndarray.
        color (_Colors, optional):
            The color of the boxes to draw. This can be a single color or a list
            of colors. Defaults to (0, 255, 0).
        thickness (_Thicknesses, optional):
            The thickness of the boxes lines to draw. This can be a single
            thickness or a list of thicknesses. Defaults to 2.

    Returns:
        np.ndarray: The image with the drawn boxes, as a numpy ndarray.
    """
    boxes = prepare_boxes(boxes)
    colors = prepare_colors(colors, len(boxes))
    thicknesses = prepare_thicknesses(thicknesses, len(boxes))
    for box, c, t in zip(boxes, colors, thicknesses):
        draw_box(img, box, color=c, thickness=t)
    return img


def draw_polygon(
    img: np.ndarray,
    polygon: _Polygon,
    color: _Color = (0, 255, 0),
    thickness: _Thickness = 2,
    fillup=False,
    **kwargs
):
    """
    Draw a polygon on the input image.

    Args:
        img (np.ndarray):
            The input image on which the polygon will be drawn.
        polygon (_Polygon):
            The points of the polygon. It can be either a list of points in the
            format [(x1, y1), (x2, y2), ...] or a Polygon object.
        color (Tuple[int, int, int], optional):
            The color of the polygon (BGR format).
            Defaults to (0, 255, 0) (green).
        thickness (int, optional):
            The thickness of the polygon's edges.
            Defaults to 2.
        fill (bool, optional):
            Whether to fill the polygon with the specified color.
            Defaults to False.

    Returns:
        np.ndarray: The image with the drawn polygon.
    """
    img = prepare_img(img)
    polygon = prepare_polygon(polygon)
    color = prepare_color(color)
    thickness = prepare_thickness(thickness)
    if polygon.is_normalized:
        h, w = img.shape[:2]
        polygon = polygon.denormalize(w, h)
    polygon = polygon.numpy().astype(int).tolist()

    if fillup:
        img = cv2.fillPoly(img, [polygon], color=color, **kwargs)
    else:
        img = cv2.polylines(img, [polygon], isClosed=True, color=color, thickness=thickness, **kwargs)

    return img


def draw_polygons(
    img: np.ndarray,
    polygons: _Polygons,
    colors: _Colors = (0, 255, 0),
    thicknesses: _Thicknesses = 2,
    fillup=False,
    **kwargs
):
    """
    Draw polygons on the input image.

    Args:
        img (np.ndarray):
            The input image on which the polygons will be drawn.
        polygons (_Polygons):
            A list of polygons to draw. Each polygon can be represented either
            as a list of points in the format [(x1, y1), (x2, y2), ...] or as a
            Polygon object.
        colors (_Colors, optional):
            The color(s) of the polygons in BGR format.
            If a single color is provided, it will be used for all polygons.
            If multiple colors are provided, each polygon will be drawn with the
            corresponding color.
            Defaults to (0, 255, 0) (green).
        thicknesses (_Thicknesses, optional):
            The thickness(es) of the polygons' edges.
            If a single thickness value is provided, it will be used for all
            polygons. If multiple thickness values are provided, each polygon
            will be drawn with the corresponding thickness.
            Defaults to 2.
        fillup (bool, optional):
            Whether to fill the polygons with the specified color(s).
            If set to True, the polygons will be filled; otherwise, only their
            edges will be drawn.
            Defaults to False.

    Returns:
        np.ndarray: The image with the drawn polygons.
    """
    polygons = prepare_polygons(polygons)
    colors = prepare_colors(colors, len(polygons))
    thicknesses = prepare_thicknesses(thicknesses, len(polygons))
    for polygon, c, t in zip(polygons, colors, thicknesses):
        draw_polygon(img, polygon, color=c, thickness=t, fillup=fillup, **kwargs)
    return img


def draw_text(
    img: np.ndarray,
    text: str,
    location: _Point,
    color: _Color = (0, 0, 0),
    text_size: int = 12,
    font_path: Union[str, Path] = None,
    **kwargs
) -> np.ndarray:
    """
    Draw specified text on the given image at the provided location.

    Args:
        img (np.ndarray):
            Image on which to draw the text.
        text (str):
            Text string to be drawn.
        location (_Point):
            x, y coordinates on the image where the text should be drawn.
        color (_Color, optional):
            RGB values of the text color. Default is black (0, 0, 0).
        text_size (int, optional):
            Size of the text to be drawn. Default is 12.
        font_path (str, optional):
            Path to the font file to be used.
            If not provided, a default font "NotoSansMonoCJKtc-VF.ttf" is used.
        **kwargs:
            Additional arguments for drawing, depending on the underlying
            library or method used.

    Returns:
        np.ndarray: Image with the text drawn on it.
    """
    img = prepare_img(img)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    draw = ImageDraw.Draw(img)
    font_path = DIR / "NotoSansMonoCJKtc-VF.ttf" if font_path is None else font_path
    font = ImageFont.truetype(str(font_path), size=text_size)

    _, top, _, bottom = font.getbbox(text)
    _, offset = font.getmask2(text)
    text_height = bottom - top

    offset_y = int(0.5 * (font.size - text_height) - offset[1])
    loc = prepare_point(location)
    loc = (loc[0], loc[1] + offset_y)
    kwargs.update({'fill': (color[2], color[1], color[0])})
    draw.text(loc, text, font=font, **kwargs)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img


def draw_point(
    img: np.ndarray,
    point: _Point,
    scale: _Scale = 1.0,
    color: _Color = (0, 255, 0),
    thickness: _Thickness = -1,
) -> np.ndarray:
    '''
    Draw a point on the image.

    Args:
        img (np.ndarray):
            Image on which to draw the point.
        point (_Point):
            The point to draw.
        scale (_Scale, optional):
            The scale of the point. Defaults to 1.0.
        color (_Color, optional):
            The color of the point. Defaults to (0, 255, 0).
        thickness (_Thickness, optional):
            The thickness of the point. Defaults to -1.

    Returns:
        np.ndarray: Image with the drawn point.
    '''
    is_gray_img = img.ndim == 2
    img = prepare_img(img)
    point = prepare_point(point)
    color = prepare_color(color)
    h, w = img.shape[:2]
    size = 1 + (np.sqrt(h * w) * 0.002 * scale).round().astype(int).item()
    img = cv2.circle(img, point, radius=size, color=color, lineType=cv2.LINE_AA, thickness=thickness)
    img = img[..., 0] if is_gray_img else img
    return img


def draw_points(
    img: np.ndarray,
    points: _Points,
    scales: _Scales = 1.,
    colors: _Colors = (0, 255, 0),
    thicknesses: _Thicknesses = -1,
) -> np.ndarray:
    '''
    Draw multiple points on the image.

    Args:
        img (np.ndarray):
            Image on which to draw the points.
        points (_Points):
            The points to draw.
        scales (_Scales, optional):
            The scales of the points. Defaults to 1..
        colors (_Colors, optional):
            The colors of the points. Defaults to (0, 255, 0).
        thicknesses (_Thicknesses, optional):
            The thicknesses of the points. Defaults to -1.

    Returns:
        np.ndarray: Image with the drawn points.
    '''
    img = prepare_img(img).copy()
    points = prepare_points(points)
    colors = prepare_colors(colors, len(points))
    thicknesses = prepare_thicknesses(thicknesses, len(points))
    scales = prepare_scales(scales, len(points))

    for p, s, c, t in zip(points, scales, colors, thicknesses):
        img = draw_point(img, p, s, c, t)

    return img


def draw_keypoints(
    img: np.ndarray,
    keypoints: _Keypoints,
    scale: _Scale = 1.,
    thickness: _Thickness = -1
) -> np.ndarray:
    '''
    Draw keypoints on the image.

    Args:
        img (np.ndarray):
            Image on which to draw the keypoints.
        keypoints (_Keypoints):
            The keypoints to draw.
        scale (float, optional):
            The scale of the keypoints. Defaults to 1..
        thickness (_Thickness, optional):
            The thickness of the keypoints. Defaults to -1.

    Returns:
        np.ndarray: Image with the drawn keypoints.
    '''
    img = prepare_img(img)
    keypoints = prepare_keypoints(keypoints)

    if keypoints.is_normalized:
        h, w = img.shape[:2]
        keypoints = keypoints.denormalize(w, h)

    colors = prepare_colors(np.array(keypoints.point_colors), len(keypoints))
    scale = prepare_scale(scale)
    thickness = prepare_thickness(thickness)
    points = keypoints.numpy()[..., :2]
    for p, c in zip(points, colors):
        img = draw_point(img, p, scale, c, thickness)
    return img


def draw_keypoints_list(
    img: np.ndarray,
    keypoints_list: _KeypointsList,
    scales: _Scales = 1.,
    thicknesses: _Thicknesses = -1
) -> np.ndarray:
    '''
    Draw keypoints list on the image.

    Args:
        img (np.ndarray):
            Image on which to draw the keypoints.
        keypoints_list (KeypointsList):
            The keypoints list to draw.
        scales (_Scales, optional):
            The scales of the keypoints. Defaults to 1..
        thicknesses (_Thicknesses, optional):
            The thicknesses of the keypoints. Defaults to -1.

    Returns:
        np.ndarray: Image with the drawn keypoints list.
    '''
    img = prepare_img(img)
    keypoints_list = prepare_keypoints_list(keypoints_list)
    scales = prepare_scales(scales, len(keypoints_list))
    thicknesses = prepare_thicknesses(thicknesses, len(keypoints_list))
    for ps, s, t in zip(keypoints_list, scales, thicknesses):
        img = draw_keypoints(img, ps, s, t)
    return img


def generate_colors_from_cmap(n: int, scheme: str) -> List[tuple]:
    cm = matplotlib.cm.get_cmap(scheme)
    return [cm(i/n)[:-1] for i in range(n)]


def generate_triadic_colors(n: int) -> List[tuple]:
    base_hue = np.random.rand()
    return [matplotlib.colors.hsv_to_rgb(((base_hue + i / 3.0) % 1, 1, 1)) for i in range(n)]


def generate_analogous_colors(n: int) -> List[tuple]:
    base_hue = np.random.rand()
    step = 0.05
    return [matplotlib.colors.hsv_to_rgb(((base_hue + i * step) % 1, 1, 1)) for i in range(n)]


def generate_square_colors(n: int) -> List[tuple]:
    base_hue = np.random.rand()
    return [matplotlib.colors.hsv_to_rgb(((base_hue + i / 4.0) % 1, 1, 1)) for i in range(n)]


def generate_colors(n: int, scheme: str = 'hsv') -> List[tuple]:
    """
    Generates n different colors based on the chosen color scheme.
    """
    color_generators = {
        'triadic': generate_triadic_colors,
        'analogous': generate_analogous_colors,
        'square': generate_square_colors
    }

    if scheme in color_generators:
        colors = color_generators[scheme](n)
    else:
        try:
            colors = generate_colors_from_cmap(n, scheme)
        except ValueError:
            print(
                f"Color scheme '{scheme}' not recognized. Returning empty list.")
            colors = []

    return [tuple(int(c * 255) for c in color) for color in colors]


def draw_ocr_infos(
    img: np.ndarray,
    texts: List[str],
    polygons: Polygons,
    colors: tuple = None,
    concat_axis: int = 1,
    thicknesses: int = 2,
    font_path: str = None,
) -> np.ndarray:
    """
    Draw the OCR results on the image.

    Args:
        img (np.ndarray):
            The image to draw on.
        texts (List[str]):
            List of detected text strings.
        polygons (D.Polygons):
            List of polygons representing the boundaries of detected texts.
        colors (tuple, optional):
            RGB values for the drawing color.
            If not provided, generates unique colors for each text.
        concat_axis (int, optional):
            Axis for concatenating the original image and the annotated one.
            Default is 1 (horizontal).
        thicknesses (int, optional):
            Thickness of the drawn polygons.
            Default is 2.
        font_path (str, optional):
            Path to the font file to be used.
            If not provided, a default font "NotoSansMonoCJKtc-VF.ttf" is used.

    Returns:
        np.ndarray: An image with the original and annotated images concatenated.
    """
    if colors is None:
        colors = generate_colors(len(texts), scheme='square')

    export_img1 = draw_polygons(
        img, polygons, color=colors, thickness=thicknesses)
    export_img2 = draw_polygons(
        np.zeros_like(img) + 255,
        polygons,
        color=colors,
        thickness=thicknesses
    )

    for text, region in zip(texts, polygons):
        text_size = max(int(0.65 * min(region.min_box_wh)), 8)
        export_img2 = draw_text(
            export_img2, text, region.numpy()[0],
            text_size=text_size,
            font_path=font_path
        )

    return np.concatenate([export_img1, export_img2], axis=concat_axis)


def draw_mask(
    img: np.ndarray,
    mask: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    weight: Tuple[float, float] = (0.5, 0.5),
    gamma: float = 0,
    min_max_normalize: bool = False
) -> np.ndarray:
    """
    Draw the mask on the image.

    Args:
        img (np.ndarray):
            The image to draw on.
        mask (np.ndarray):
            The mask to draw.
        colormap (int, optional):
            The colormap to use for the mask. Defaults to cv2.COLORMAP_JET.
        weight (Tuple[float, float], optional):
            Weights for the image and the mask. Defaults to (0.5, 0.5).
        gamma (float, optional):
            Gamma value for the mask. Defaults to 0.
        min_max_normalize (bool, optional):
            Whether to normalize the mask to the range [0, 1]. Defaults to False.

    Returns:
        np.ndarray: The image with the drawn mask.
    """

    # Ensure the input image has 3 channels
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    else:
        img = img.copy()  # Avoid modifying the original image

    # Normalize mask if required
    if min_max_normalize:
        mask = mask.astype(np.float32)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)  # Ensure mask is uint8 for color mapping

    # Ensure mask is single-channel before applying color map
    if mask.ndim == 3 and mask.shape[-1] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    elif mask.ndim != 2:
        raise ValueError("Mask must be either 2D or 3-channel image")

    mask = imresize(mask, size=(img.shape[0], img.shape[1]))
    mask = cv2.applyColorMap(mask, colormap)
    img_mask = cv2.addWeighted(img, weight[0], mask, weight[1], gamma)

    return img_mask
