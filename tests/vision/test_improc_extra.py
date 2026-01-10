from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import capybara.vision.improc as improc
from capybara import IMGTYP, ROTATE


def test_is_numpy_img_accepts_2d_and_3d_and_rejects_other_shapes():
    assert improc.is_numpy_img(np.zeros((10, 10), dtype=np.uint8))
    assert improc.is_numpy_img(np.zeros((10, 10, 1), dtype=np.uint8))
    assert improc.is_numpy_img(np.zeros((10, 10, 3), dtype=np.uint8))
    assert not improc.is_numpy_img(np.zeros((10, 10, 4), dtype=np.uint8))
    assert not improc.is_numpy_img("not-an-array")  # type: ignore[arg-type]


def test_get_orientation_code_maps_known_values(monkeypatch):
    def fake_load(_: Any):
        return {"0th": {improc.piexif.ImageIFD.Orientation: 6}}

    monkeypatch.setattr(improc.piexif, "load", fake_load)
    assert improc.get_orientation_code(b"") == ROTATE.ROTATE_90

    def fake_load_3(_: Any):
        return {"0th": {improc.piexif.ImageIFD.Orientation: 3}}

    monkeypatch.setattr(improc.piexif, "load", fake_load_3)
    assert improc.get_orientation_code(b"") == ROTATE.ROTATE_180

    def fake_load_8(_: Any):
        return {"0th": {improc.piexif.ImageIFD.Orientation: 8}}

    monkeypatch.setattr(improc.piexif, "load", fake_load_8)
    assert improc.get_orientation_code(b"") == ROTATE.ROTATE_270

    def fake_load_other(_: Any):
        return {"0th": {improc.piexif.ImageIFD.Orientation: 1}}

    monkeypatch.setattr(improc.piexif, "load", fake_load_other)
    assert improc.get_orientation_code(b"") is None

    monkeypatch.setattr(
        improc.piexif,
        "load",
        lambda _: (_ for _ in ()).throw(Exception("boom")),
    )
    assert improc.get_orientation_code(b"") is None


def test_jpgencode_handles_tuple_return_and_failures(monkeypatch):
    class FakeJPEG:
        def __init__(self, *, raise_encode: bool = False) -> None:
            self.raise_encode = raise_encode

        def encode(self, img: np.ndarray, quality: int = 90):
            if self.raise_encode:
                raise RuntimeError("boom")
            return (b"xx", b"ignored")

    monkeypatch.setattr(improc, "jpeg", FakeJPEG())
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    assert improc.jpgencode(img) == b"xx"

    monkeypatch.setattr(improc, "jpeg", FakeJPEG(raise_encode=True))
    assert improc.jpgencode(img) is None

    assert improc.jpgencode(np.zeros((8, 8, 4), dtype=np.uint8)) is None


def test_jpgdecode_rotates_based_on_orientation(monkeypatch):
    img = np.zeros((5, 5, 3), dtype=np.uint8)

    class FakeJPEG:
        def decode(self, _: bytes) -> np.ndarray:
            return img

    monkeypatch.setattr(improc, "jpeg", FakeJPEG())
    monkeypatch.setattr(
        improc, "get_orientation_code", lambda _: ROTATE.ROTATE_90
    )
    monkeypatch.setattr(improc, "imrotate90", lambda arr, code: arr + 1)
    out = improc.jpgdecode(b"bytes")
    assert isinstance(out, np.ndarray)
    assert out.sum() > 0

    class BadJPEG:
        def decode(self, _: bytes) -> np.ndarray:
            raise RuntimeError("bad")

    monkeypatch.setattr(improc, "jpeg", BadJPEG())
    assert improc.jpgdecode(b"bytes") is None


def test_pngencode_pngdecode_roundtrip_and_invalid_inputs():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[2:4, 2:4] = 255

    enc = improc.pngencode(img)
    assert isinstance(enc, bytes)
    dec = improc.pngdecode(enc)
    assert isinstance(dec, np.ndarray)
    assert dec.shape == img.shape

    assert improc.pngencode(np.zeros((10, 10, 4), dtype=np.uint8)) is None
    assert improc.pngdecode(b"not-a-real-image") is None


def test_imencode_selects_format_and_validates_kwargs(monkeypatch):
    monkeypatch.setattr(improc, "jpgencode", lambda _: b"jpg")
    monkeypatch.setattr(improc, "pngencode", lambda _: b"png")

    dummy = np.zeros((2, 2, 3), dtype=np.uint8)
    assert improc.imencode(dummy, IMGTYP.JPEG) == b"jpg"
    assert improc.imencode(dummy, IMGTYP.PNG) == b"png"
    assert improc.imencode(dummy, IMGTYP="PNG") == b"png"

    with pytest.raises(TypeError, match="both provided"):
        improc.imencode(dummy, "png", IMGTYP="jpeg")

    with pytest.raises(TypeError, match="Unexpected keyword"):
        improc.imencode(dummy, unexpected=1)  # type: ignore[arg-type]


def test_imdecode_falls_back_to_png_when_jpeg_decode_fails(monkeypatch):
    monkeypatch.setattr(improc, "jpgdecode", lambda _: None)
    monkeypatch.setattr(
        improc, "pngdecode", lambda _: np.zeros((1, 1, 3), dtype=np.uint8)
    )
    out = improc.imdecode(b"blob")
    assert isinstance(out, np.ndarray)

    monkeypatch.setattr(
        improc, "jpgdecode", lambda _: (_ for _ in ()).throw(RuntimeError("x"))
    )
    assert improc.imdecode(b"blob") is None


def test_img_to_b64_and_back(monkeypatch):
    dummy = np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(improc, "jpgencode", lambda _: b"\x00\x01")
    b64 = improc.img_to_b64(dummy, IMGTYP.JPEG)
    assert isinstance(b64, bytes)

    b64str = improc.img_to_b64str(dummy, IMGTYP.JPEG)
    assert isinstance(b64str, str)

    monkeypatch.setattr(
        improc, "imdecode", lambda _: np.ones((1, 1, 3), dtype=np.uint8)
    )
    out = improc.b64_to_img(b64)
    assert isinstance(out, np.ndarray)


def test_img_to_b64_accepts_imgtyp_via_imgtyp_kwarg(monkeypatch):
    dummy = np.zeros((2, 2, 3), dtype=np.uint8)

    monkeypatch.setattr(improc, "pngencode", lambda _: b"\x00\x01")
    b64 = improc.img_to_b64(dummy, IMGTYP="PNG")
    assert isinstance(b64, bytes)


def test_b64str_to_img_validates_and_warns(monkeypatch):
    monkeypatch.setattr(
        improc, "b64_to_img", lambda _: np.zeros((1, 1, 3), dtype=np.uint8)
    )
    assert improc.b64str_to_img("AA==") is not None

    with pytest.warns(UserWarning, match="b64str is None"):
        assert improc.b64str_to_img(None) is None

    with pytest.raises(ValueError, match="not a string"):
        improc.b64str_to_img(123)  # type: ignore[arg-type]


def test_npy_b64_roundtrip_and_npyread(tmp_path: Path):
    arr = np.array([1.0, 2.0], dtype=np.float32)
    b64 = improc.npy_to_b64(arr)
    out = improc.b64_to_npy(b64)
    np.testing.assert_allclose(out, arr)

    b64s = improc.npy_to_b64str(arr)
    out2 = improc.b64str_to_npy(b64s)
    np.testing.assert_allclose(out2, arr)

    file = tmp_path / "x.npy"
    np.save(file, arr)
    loaded = improc.npyread(file)
    assert loaded is not None
    np.testing.assert_allclose(loaded, arr)

    assert improc.npyread(tmp_path / "missing.npy") is None


def test_pdf2imgs_handles_bytes_and_paths_and_failures(
    monkeypatch, tmp_path: Path
):
    from PIL import Image

    pil = Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8))

    monkeypatch.setattr(improc, "convert_from_bytes", lambda _: [pil])
    monkeypatch.setattr(improc, "convert_from_path", lambda _: [pil])
    monkeypatch.setattr(improc, "imcvtcolor", lambda arr, cvt_mode: arr)

    out = improc.pdf2imgs(b"%PDF-1.0")
    assert out is not None
    assert len(out) == 1
    assert isinstance(out[0], np.ndarray)

    pdf_path = tmp_path / "a.pdf"
    pdf_path.write_bytes(b"%PDF-1.0")
    out2 = improc.pdf2imgs(pdf_path)
    assert out2 is not None
    assert len(out2) == 1

    monkeypatch.setattr(
        improc,
        "convert_from_path",
        lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert improc.pdf2imgs(pdf_path) is None


def test_pngdecode_returns_none_when_cv2_imdecode_raises(monkeypatch):
    monkeypatch.setattr(
        improc.cv2,
        "imdecode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("x")),
    )
    assert improc.pngdecode(b"bytes") is None


def test_img_to_b64_validates_kwargs_and_handles_encode_none_and_exceptions(
    monkeypatch,
):
    dummy = np.zeros((2, 2, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="both provided"):
        improc.img_to_b64(dummy, "png", IMGTYP="jpeg")

    with pytest.raises(TypeError, match="Unexpected keyword"):
        improc.img_to_b64(dummy, unexpected=1)  # type: ignore[arg-type]

    monkeypatch.setattr(improc, "jpgencode", lambda *_args, **_kwargs: None)
    assert improc.img_to_b64(dummy, IMGTYP.JPEG) is None

    monkeypatch.setattr(improc, "jpgencode", lambda *_args, **_kwargs: b"x")
    monkeypatch.setattr(
        improc.pybase64,
        "b64encode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )
    assert improc.img_to_b64(dummy, IMGTYP.JPEG) is None


def test_b64_to_img_returns_none_when_b64decode_raises(monkeypatch):
    monkeypatch.setattr(
        improc.pybase64,
        "b64decode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )
    assert improc.b64_to_img(b"@@@") is None


def test_imread_warns_when_image_is_none(monkeypatch, tmp_path: Path):
    path = tmp_path / "x.jpg"
    path.write_bytes(b"not-a-real-jpg")

    monkeypatch.setattr(improc, "jpgread", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(improc.cv2, "imread", lambda *_args, **_kwargs: None)

    with pytest.warns(UserWarning, match="None type image"):
        assert improc.imread(path, verbose=True) is None


def test_imwrite_converts_color_base(monkeypatch, tmp_path: Path):
    calls: list[str] = []

    def fake_imcvtcolor(img: np.ndarray, *, cvt_mode: str) -> np.ndarray:
        calls.append(cvt_mode)
        return img

    monkeypatch.setattr(improc, "imcvtcolor", fake_imcvtcolor)
    monkeypatch.setattr(improc.cv2, "imwrite", lambda *_args, **_kwargs: True)

    img = np.zeros((2, 2, 3), dtype=np.uint8)
    out_path = tmp_path / "out.jpg"
    assert improc.imwrite(img, path=out_path, color_base="RGB")
    assert calls == ["RGB2BGR"]
