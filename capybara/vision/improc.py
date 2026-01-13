import os
import tempfile
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import piexif
import pillow_heif
import pybase64
from pdf2image import convert_from_bytes, convert_from_path
from turbojpeg import TurboJPEG

from ..enums import IMGTYP, ROTATE
from .functionals import imcvtcolor
from .geometric import imrotate90

__all__ = [
    "b64_to_img",
    "b64_to_npy",
    "b64str_to_img",
    "b64str_to_npy",
    "get_orientation_code",
    "imdecode",
    "imencode",
    "img_to_b64",
    "img_to_b64str",
    "imread",
    "imwrite",
    "is_numpy_img",
    "jpgdecode",
    "jpgencode",
    "jpgread",
    "npy_to_b64",
    "npy_to_b64str",
    "npyread",
    "pdf2imgs",
    "pngdecode",
    "pngencode",
]

jpeg = TurboJPEG()


def is_numpy_img(x: Any) -> bool:
    """
    x == ndarray (H x W x C)
    """
    return isinstance(x, np.ndarray) and (
        x.ndim == 2 or (x.ndim == 3 and x.shape[-1] in [1, 3])
    )


def get_orientation_code(stream: str | Path | bytes):
    try:
        exif_dict = piexif.load(stream)
    except Exception:
        return None

    orientation = exif_dict.get("0th", {}).get(piexif.ImageIFD.Orientation)
    if orientation == 3:
        return ROTATE.ROTATE_180
    if orientation == 6:
        return ROTATE.ROTATE_90
    if orientation == 8:
        return ROTATE.ROTATE_270
    return None


def jpgencode(img: np.ndarray, quality: int = 90) -> bytes | None:
    byte_ = None
    if is_numpy_img(img):
        with suppress(Exception):
            encoded = jpeg.encode(img, quality=quality)
            if isinstance(encoded, tuple):
                encoded = encoded[0]
            byte_ = cast(bytes, encoded)
    return byte_


def jpgdecode(byte_: bytes) -> np.ndarray | None:
    try:
        bgr_array = jpeg.decode(byte_)
        code = get_orientation_code(byte_)
        bgr_array = (
            imrotate90(bgr_array, code) if code is not None else bgr_array
        )
    except Exception as _:
        bgr_array = None

    return bgr_array


def jpgread(img_file: str | Path) -> np.ndarray | None:
    with open(str(img_file), "rb") as f:
        binary_img = f.read()
        bgr_array = jpgdecode(binary_img)

    return bgr_array


def pngencode(img: np.ndarray, compression: int = 1) -> bytes | None:
    byte_ = None
    if is_numpy_img(img):
        with suppress(Exception):
            byte_ = cv2.imencode(
                ".png",
                img,
                params=[int(cv2.IMWRITE_PNG_COMPRESSION), compression],
            )[1].tobytes()
    return byte_


def pngdecode(byte_: bytes) -> np.ndarray | None:
    try:
        enc = np.frombuffer(byte_, "uint8")
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    except Exception as _:
        img = None
    return img


def imencode(
    img: np.ndarray,
    imgtyp: str | int | IMGTYP = IMGTYP.JPEG,
    **kwargs: object,
) -> bytes | None:
    if "IMGTYP" in kwargs:
        if imgtyp != IMGTYP.JPEG:
            raise TypeError("imgtyp and IMGTYP were both provided.")
        imgtyp = cast(str | int | IMGTYP, kwargs.pop("IMGTYP"))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    imgtyp_enum = IMGTYP.obj_to_enum(imgtyp)
    encode_fn = jpgencode if imgtyp_enum == IMGTYP.JPEG else pngencode
    return encode_fn(img)


def imdecode(byte_: bytes) -> np.ndarray | None:
    try:
        img = jpgdecode(byte_)
        img = pngdecode(byte_) if img is None else img
    except Exception as _:
        img = None
    return img


def img_to_b64(
    img: np.ndarray,
    imgtyp: str | int | IMGTYP = IMGTYP.JPEG,
    **kwargs: object,
) -> bytes | None:
    if "IMGTYP" in kwargs:
        if imgtyp != IMGTYP.JPEG:
            raise TypeError("imgtyp and IMGTYP were both provided.")
        imgtyp = cast(str | int | IMGTYP, kwargs.pop("IMGTYP"))
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    imgtyp_enum = IMGTYP.obj_to_enum(imgtyp)
    encode_fn = jpgencode if imgtyp_enum == IMGTYP.JPEG else pngencode
    try:
        encoded = encode_fn(img)
        if encoded is None:
            return None
        b64 = pybase64.b64encode(encoded)
    except Exception as _:
        b64 = None
    return b64


def npy_to_b64(x: np.ndarray, dtype="float32") -> bytes:
    return pybase64.b64encode(x.astype(dtype).tobytes())


def npy_to_b64str(
    x: np.ndarray, dtype="float32", string_encode: str = "utf-8"
) -> str:
    return pybase64.b64encode(x.astype(dtype).tobytes()).decode(string_encode)


def img_to_b64str(
    img: np.ndarray,
    imgtyp: str | int | IMGTYP = IMGTYP.JPEG,
    string_encode: str = "utf-8",
    **kwargs: object,
) -> str | None:
    b64 = img_to_b64(img, imgtyp, **kwargs)
    return b64.decode(string_encode) if isinstance(b64, bytes) else None


def b64_to_img(b64: bytes) -> np.ndarray | None:
    try:
        img = imdecode(pybase64.b64decode(b64))
    except Exception as _:
        img = None
    return img


def b64str_to_img(
    b64str: str | None, string_encode: str = "utf-8"
) -> np.ndarray | None:
    if b64str is None:
        warnings.warn("b64str is None.", stacklevel=2)
        return None

    if not isinstance(b64str, str):
        raise ValueError("b64str is not a string.")

    return b64_to_img(b64str.encode(string_encode))


def b64_to_npy(x: bytes, dtype="float32") -> np.ndarray:
    return np.frombuffer(pybase64.b64decode(x), dtype=dtype)


def b64str_to_npy(
    x: str, dtype="float32", string_encode: str = "utf-8"
) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode(x.encode(string_encode)), dtype=dtype
    )


def npyread(path: str | Path) -> np.ndarray | None:
    try:
        with open(str(path), "rb") as f:
            img = np.load(f)
    except Exception as _:
        img = None
    return img


def imread(
    path: str | Path, color_base: str = "BGR", verbose: bool = False
) -> np.ndarray | None:
    """
    This function reads an image from a given file path and converts its color
    base if necessary.

    Args:
        path (Union[str, Path]):
            The path to the image file to be read.
        color_base (str, optional):
            The desired color base for the image. If not 'BGR', will attempt to
            convert using 'imcvtcolor' function. Defaults to 'BGR'.
        verbose (bool, optional):
            If set to True, a warning will be issued when the read image is None.
            Defaults to False.

    Raises:
        FileExistsError:
            If the image file at the specified path does not exist.

    Returns:
        Union[np.ndarray, None]:
            The image as a numpy ndarray if successful, None otherwise.
    """
    if not Path(path).exists():
        raise FileExistsError(f"{path} can not found.")

    color_base = color_base.upper()

    if Path(path).suffix.lower() == ".heic":
        heif_file = pillow_heif.open_heif(
            str(path), convert_hdr_to_8bit=True, bgr_mode=True
        )
        img = np.asarray(heif_file)
    else:
        img = jpgread(path)
        img = cv2.imread(str(path)) if img is None else img

    if img is None:
        if verbose:
            warnings.warn("Got a None type image.", stacklevel=2)
        return

    if color_base != "BGR":
        img = imcvtcolor(img, cvt_mode=f"BGR2{color_base}")

    return img


def imwrite(
    img: np.ndarray,
    path: str | Path | None = None,
    color_base: str = "BGR",
    suffix: str = ".jpg",
) -> bool:
    """
    Writes an image to a file with optional color base conversion.

    Args:
        img (np.ndarray):
            The image to write, as a numpy ndarray.
        path (Union[str, Path], optional):
            The path where to write the image file. If None, writes to a temporary
            file. Defaults to None.
        color_base (str, optional):
            The current color base of the image. If not 'BGR', the function will
            attempt to convert it to 'BGR'. Defaults to 'BGR'.
        suffix (str, optional):
            The suffix of the temporary file if path is None. Defaults to '.jpg'.

    Returns:
        bool: True if the write operation is successful, False otherwise.
    """
    color_base = color_base.upper()
    if color_base != "BGR":
        img = imcvtcolor(img, cvt_mode=f"{color_base}2BGR")
    if path is None:
        fd, target = tempfile.mkstemp(prefix="capybara_", suffix=suffix)
        os.close(fd)
    else:
        target = str(path)
    return bool(cv2.imwrite(target, img))


def pdf2imgs(stream: str | Path | bytes) -> list[np.ndarray] | None:
    """
    Function for converting a PDF document to numpy images.

    Args:
        file_dir (str): A path of PDF document.

    Returns:
        img: Images will be a list of np image representing each page of the PDF document.
    """
    try:
        if isinstance(stream, bytes):
            pil_imgs = convert_from_bytes(stream)
        else:
            pil_imgs = convert_from_path(stream)
        return [
            imcvtcolor(np.array(img), cvt_mode="RGB2BGR") for img in pil_imgs
        ]
    except Exception as _:
        return
