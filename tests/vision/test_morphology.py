import numpy as np
import pytest

from capybara.enums import MORPH
from capybara.vision import morphology as morph


@pytest.mark.parametrize(
    "fn",
    [
        morph.imerode,
        morph.imdilate,
        morph.imopen,
        morph.imclose,
        morph.imgradient,
        morph.imtophat,
        morph.imblackhat,
    ],
)
def test_morphology_ops_preserve_shape_and_dtype(fn):
    img = np.zeros((20, 20), dtype=np.uint8)
    img[8:12, 8:12] = 255

    out = fn(img, ksize=3, kstruct=MORPH.RECT)
    assert out.shape == img.shape
    assert out.dtype == img.dtype

    out2 = fn(img, ksize=(5, 3), kstruct="CROSS")
    assert out2.shape == img.shape

    out3 = fn(img, ksize=3, kstruct=MORPH.RECT.value)
    assert out3.shape == img.shape


@pytest.mark.parametrize(
    "fn",
    [
        morph.imerode,
        morph.imdilate,
        morph.imopen,
        morph.imclose,
        morph.imgradient,
        morph.imtophat,
        morph.imblackhat,
    ],
)
def test_morphology_invalid_ksize_raises(fn):
    img = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(TypeError):
        fn(img, ksize=(1, 2, 3))  # type: ignore[arg-type]


def test_morphology_invalid_kstruct_raises():
    img = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError):
        morph.imerode(img, kstruct="cross")
