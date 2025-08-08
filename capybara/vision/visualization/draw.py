import colorsys
import functools
import hashlib
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
from ...utils import download_from_google, get_curdir
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
    'generate_colors', 'draw_mask', 'draw_point', 'draw_points', 'draw_keypoints',
    'draw_keypoints_list', 'draw_detection', 'draw_detections',
]

DIR = get_curdir(__file__)

DEFAULT_FONT_PATH = DIR / "NotoSansMonoCJKtc-VF.ttf"


if not (font_path := DIR / DEFAULT_FONT_PATH).exists():
    file_id = "1m6jvsBGKgQsxzpIoe4iEp_EqFxYXe7T1"
    download_from_google(file_id, font_path.name, DIR)


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
    polygon = polygon.numpy().astype(int)

    if fillup:
        img = cv2.fillPoly(img, [polygon], color=color, **kwargs)
    else:
        img = cv2.polylines(img, [polygon], isClosed=True,
                            color=color, thickness=thickness, **kwargs)

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
        draw_polygon(img, polygon, color=c, thickness=t,
                     fillup=fillup, **kwargs)
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
    font_path = DEFAULT_FONT_PATH if font_path is None else font_path
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


def draw_line(
    img: np.ndarray,
    pt1: _Point,
    pt2: _Point,
    color: _Color = (0, 255, 0),
    thickness: _Thickness = 1,
    style: str = 'dotted',
    gap: int = 20,
    inplace: bool = False,
):
    '''
    Draw a line on the image.

    Args:
        img (np.ndarray):
            Image on which to draw the line.
        pt1 (_Point):
            The starting point of the line.
        pt2 (_Point):
            The ending point of the line.
        color (_Color, optional):
            The color of the line. Defaults to (0, 255, 0).
        thickness (_Thickness, optional):
            The thickness of the line. Defaults to 1.
        style (str, optional):
            The style of the line. It can be either 'dotted' or 'line'.
        gap (int, optional):
            The gap between the dots. Defaults to 20.
        inplace (bool, optional):
            Whether to draw on the input image directly or return a new image.
            Defaults to False.

    Raises:
        ValueError:
            If the style is not 'dotted' or 'line'.

    Returns:
        np.ndarray: Image with the drawn line.
    '''
    img = img.copy() if not inplace else img
    img = prepare_img(img)
    pt1 = prepare_point(pt1)
    pt2 = prepare_point(pt2)
    dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    elif style == 'line':
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1
    else:
        raise ValueError(f"Unknown style: {style}")
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
    img = cv2.circle(img, point, radius=size, color=color,
                     lineType=cv2.LINE_AA, thickness=thickness)
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


def _vdc(n: int, base: int = 2) -> float:
    """Return n-th Van der Corput radical inverse in given base."""
    vdc, denom = 0.0, 1.0
    while n:
        n, rem = divmod(n, base)
        denom *= base
        vdc += rem / denom
    return vdc


@functools.lru_cache(maxsize=None)
def distinct_color(idx: int) -> _Color:
    """Generate a perceptually distinct BGR color for class *idx*.

    1. Hue:
        use VDC sequence (0, 0.5, 0.25, 0.75, …) to maximize separation
        between close indices.
    2. Saturation / Value: cycle every 20 / 10 ids to avoid hue-only clashes.
    """
    hue = _vdc(idx + 1)                       # (0,1)   ─ 長距離跳耀
    sat_cycle = (0.65, 0.80, 0.50)  # ┐週期性變化
    val_cycle = (1.00, 0.90, 0.80)  # ┘增亮／加深
    s = sat_cycle[(idx // 20) % len(sat_cycle)]
    v = val_cycle[(idx // 10) % len(val_cycle)]
    r, g, b = colorsys.hsv_to_rgb(hue, s, v)
    return int(b * 255), int(g * 255), int(r * 255)


def _label_to_index(label: str) -> int:
    """Convert arbitrary label to stable int id."""
    try:
        return int(label)
    except ValueError:
        return int(hashlib.sha1(label.encode()).hexdigest()[:8], 16)


def draw_detection(
    img: np.ndarray,
    box: _Box,
    label: str,
    score: Union[float, None] = None,
    color: _Color | None = None,
    thickness: _Thickness | None = None,
    text_color: _Color = (255, 255, 255),
    font_path: Union[str, Path] | None = None,
    text_size: int | None = None,
    box_alpha: float = 1.0,
    text_bg_alpha: float = 0.6,
) -> np.ndarray:
    """
    Draw a detection box with label (and optional confidence) onto an image,
    ensuring all coordinates are valid and text background rect is sorted.

    Args:
        img: OpenCV BGR image.
        box: Bounding box, absolute or normalized.
        label: Class name.
        score: Confidence in [0,1]; if provided, appended to label.
        color: BGR color for box; if None, auto choose by hashing label.
        thickness: Line thickness; if None, auto choose by image size.
        text_color: BGR text color.
        font_path: Path to TTF font.
        text_size: Font size in points; if None, ~10% of box height (min 12).
        box_alpha: Box opacity (1 = solid).
        text_bg_alpha: Background opacity for text.

    Returns:
        Annotated BGR image.
    """
    # 1. Prepare canvas and box
    canvas = prepare_img(img).copy()
    box_obj = prepare_box(box)
    if box_obj.is_normalized:
        h_img, w_img = canvas.shape[:2]
        box_obj = box_obj.denormalize(w_img, h_img)
    x1, y1, x2, y2 = box_obj.numpy().astype(int).tolist()

    # 2. Choose box color and thickness
    if color is None:
        idx = _label_to_index(label)
        draw_color = distinct_color(idx)
    else:
        draw_color = color

    if thickness is None:
        # proportional to image diagonal
        diag = (canvas.shape[0]**2 + canvas.shape[1]**2)**0.5
        line_thickness = max(1, int(diag * 0.002 + 0.5))
    else:
        line_thickness = thickness

    # 3. Draw box (with optional transparency)
    if box_alpha >= 1.0:
        cv2.rectangle(canvas, (x1, y1), (x2, y2),
                      draw_color, line_thickness, cv2.LINE_AA)
    else:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2),
                      draw_color, line_thickness, cv2.LINE_AA)
        canvas = cv2.addWeighted(overlay, box_alpha, canvas, 1 - box_alpha, 0)

    # 4. Prepare label text
    text = f"{label} {score*100:.1f}%" if score is not None else label

    # auto font size (~10% of box height, min 12)
    if text_size is None:
        text_size = max(12, int((y2 - y1) * 0.10))

    # 5. Measure text size with PIL
    font_file = DEFAULT_FONT_PATH if font_path is None else Path(font_path)
    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, "RGBA")
    font = ImageFont.truetype(str(font_file), size=text_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # 6. Compute and clamp background rectangle coords
    pad = max(1, line_thickness // 2)
    raw_x0, raw_y0 = x1, y1 - text_h - 2 * pad
    raw_x1, raw_y1 = x1 + text_w + 2 * pad, y1

    h_img, w_img = canvas.shape[:2]
    bg_x0 = max(0, raw_x0)
    bg_y0 = max(0, raw_y0)
    bg_x1 = min(w_img, raw_x1)
    bg_y1 = min(h_img, raw_y1)

    # ensure sorted: (x0,y0) is top-left, (x1,y1) bottom-right
    x0, x1_ = sorted([bg_x0, bg_x1])
    y0, y1_ = sorted([bg_y0, bg_y1])

    # 7. Draw semi-transparent background
    draw.rectangle(
        [(x0, y0), (x1_, y1_)],
        fill=(*draw_color[::-1], int(text_bg_alpha * 255)),
    )

    # 8. Draw text (PIL expecting RGB)
    draw.text((x0 + pad, y0 + pad), text,
              font=font, fill=text_color[::-1])

    # 9. Convert back to BGR OpenCV
    annotated = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return annotated


def draw_detections(
    img: np.ndarray,
    boxes: _Boxes,
    labels: List[str],
    scores: List[float] = None,
    colors: _Colors = None,
    thicknesses: _Thicknesses = None,
    text_colors: _Colors = (255, 255, 255),
    font_path: Union[str, Path] = None,
    text_sizes: List[int] = None,
    box_alpha: float = 1.0,
    text_bg_alpha: float = 0.6,
) -> np.ndarray:
    """
    Draw multiple detection boxes with labels onto an image.

    Args:
        img (np.ndarray): OpenCV BGR image.
        boxes (_Boxes): List of bounding boxes.
        labels (List[str]): Class names for each box.
        scores (List[float], optional): Confidence scores for each box.
        colors (_Colors, optional): BGR colors for each box.
        thicknesses (_Thicknesses, optional): Line thicknesses for each box.
        text_colors (_Colors, optional): BGR text colors.
        font_path (Union[str, Path], optional): Path to TTF font.
        text_sizes (List[int], optional): Font sizes in points for each label.
        box_alpha (float, optional): Box opacity (1 = solid).
        text_bg_alpha (float, optional): Background opacity for text.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    canvas = prepare_img(img).copy()
    boxes = prepare_boxes(boxes)

    if len(boxes) != len(labels):
        raise ValueError("Number of boxes must match number of labels")

    if scores is not None and len(scores) != len(labels):
        raise ValueError("Number of scores must match number of labels")

    if colors is not None:
        colors = prepare_colors(colors, len(boxes))
    else:
        colors = [None] * len(boxes)

    if thicknesses is not None:
        thicknesses = prepare_thicknesses(thicknesses, len(boxes))
    else:
        thicknesses = [None] * len(boxes)

    if text_colors is not None:
        text_colors = prepare_colors(text_colors, len(boxes))

    if text_sizes is not None:
        text_sizes = [int(size) for size in text_sizes]

    for i, box in enumerate(boxes):
        label = labels[i]
        score = scores[i] if scores is not None else None
        color = colors[i] if colors is not None else None
        thickness = thicknesses[i] if thicknesses is not None else None
        text_color = text_colors[i] if text_colors is not None else (
            255, 255, 255)
        text_size = text_sizes[i] if text_sizes is not None else None

        canvas = draw_detection(
            canvas,
            box,
            str(label),
            score=score,
            color=color,
            thickness=thickness,
            text_color=text_color,
            font_path=font_path,
            text_size=text_size,
            box_alpha=box_alpha,
            text_bg_alpha=text_bg_alpha
        )

    return canvas
