from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import numpy as np
import pytest

import capybara as cb
from capybara import (
    DataclassCopyMixin,
    DataclassToJsonMixin,
    EnumCheckMixin,
    dict_to_jsonable,
)

MOCK_IMAGE = np.zeros((5, 5, 3), dtype="uint8")
BASE64_PNG_IMAGE = "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAIAAAACDbGyAAAADElEQVQIHWNgoC4AAABQAAFhFZyBAAAAAElFTkSuQmCC"
BASE64_NPY_IMAGE = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
data = [
    (
        {
            "box": cb.Box((0, 0, 1, 1)),
            "boxes": cb.Boxes([(0, 0, 1, 1)]),
            "keypoints": cb.Keypoints([(0, 1), (1, 0)]),
            "keypoints_list": cb.KeypointsList(
                [[(0, 1), (1, 0)], [(0, 1), (2, 0)]]
            ),
            "polygon": cb.Polygon([(0, 0), (1, 0), (1, 1)]),
            "polygons": cb.Polygons([[(0, 0), (1, 0), (1, 1)]]),
            "np_bool": np.bool_(True),
            "np_float": np.float64(1),
            "np_number": np.array(1),
            "np_array": np.array([1, 2]),
            "image": MOCK_IMAGE,
            "dict": {"box": cb.Box((0, 0, 1, 1))},
            "str": "test",
            "int": 1,
            "float": 0.6,
            "tuple": (1, 1),
            "pow": 1e10,
        },
        {"image": lambda x: cb.img_to_b64str(x, cb.IMGTYP.PNG)},
        {
            "box": [0, 0, 1, 1],
            "boxes": [[0, 0, 1, 1]],
            "keypoints": [[0, 1], [1, 0]],
            "keypoints_list": [[[0, 1], [1, 0]], [[0, 1], [2, 0]]],
            "polygon": [[0, 0], [1, 0], [1, 1]],
            "polygons": [[[0, 0], [1, 0], [1, 1]]],
            "np_bool": True,
            "np_float": 1.0,
            "np_number": 1,
            "np_array": [1, 2],
            "image": BASE64_PNG_IMAGE,
            "dict": {"box": [0, 0, 1, 1]},
            "str": "test",
            "int": 1,
            "float": 0.6,
            "tuple": [1, 1],
            "pow": 1e10,
        },
    ),
    (
        {"image": MOCK_IMAGE},
        {"image": lambda x: cb.npy_to_b64str(x)},
        {"image": BASE64_NPY_IMAGE},
    ),
    (
        {"images": [{"image": MOCK_IMAGE}]},
        {"image": lambda x: cb.npy_to_b64str(x)},
        {"images": [{"image": BASE64_NPY_IMAGE}]},
    ),
]


@pytest.mark.parametrize("x,jsonable_func,expected", data)
def test_dict_to_jsonable(x, jsonable_func, expected):
    assert dict_to_jsonable(x, jsonable_func) == expected


class TestEnum(EnumCheckMixin, Enum):
    __test__ = False

    FIRST = 1
    SECOND = "two"


class TestEnumCheckMixin:
    def test_obj_to_enum_with_valid_enum_member(self):
        assert TestEnum.obj_to_enum(TestEnum.FIRST) == TestEnum.FIRST

    def test_obj_to_enum_with_valid_string(self):
        assert TestEnum.obj_to_enum("SECOND") == TestEnum.SECOND

    def test_obj_to_enum_with_valid_int(self):
        assert TestEnum.obj_to_enum(1) == TestEnum.FIRST

    def test_obj_to_enum_with_invalid_string(self):
        with pytest.raises(ValueError):
            TestEnum.obj_to_enum("INVALID")

    def test_obj_to_enum_with_invalid_int(self):
        with pytest.raises(ValueError):
            TestEnum.obj_to_enum(3)


@dataclass
class TestDataclass(DataclassCopyMixin):
    __test__ = False

    int_field: int
    list_field: list[Any]


class TestDataclassCopyMixin:
    @pytest.fixture
    def test_dataclass_instance(self):
        return TestDataclass(10, [1, 2, 3])

    def test_shallow_copy(self, test_dataclass_instance):
        copy_instance = test_dataclass_instance.__copy__()
        assert copy_instance is not test_dataclass_instance
        assert copy_instance.int_field == test_dataclass_instance.int_field
        assert copy_instance.list_field is test_dataclass_instance.list_field

    def test_deep_copy(self, test_dataclass_instance):
        deepcopy_instance = deepcopy(test_dataclass_instance)
        assert deepcopy_instance is not test_dataclass_instance
        assert deepcopy_instance.int_field == test_dataclass_instance.int_field
        assert (
            deepcopy_instance.list_field
            is not test_dataclass_instance.list_field
        )
        assert (
            deepcopy_instance.list_field == test_dataclass_instance.list_field
        )


@dataclass
class TestDataclass2(DataclassToJsonMixin):
    __test__ = False

    box: cb.Box
    boxes: cb.Boxes
    keypoints: cb.Keypoints
    keypoints_list: cb.KeypointsList
    polygon: cb.Polygon
    polygons: cb.Polygons
    np_bool: np.bool
    np_float: np.float64
    np_number: np.ndarray
    np_array: np.ndarray
    np_array_to_b64str: np.ndarray
    image: np.ndarray
    py_dict: dict
    py_str: str
    py_int: int
    py_float: float
    py_tuple: tuple
    py_pow: float

    # based on mixin
    jsonable_func: ClassVar[dict[str, Any]] = {
        "image": lambda x: cb.img_to_b64str(x, cb.IMGTYP.PNG),
        "np_array_to_b64str": lambda x: cb.npy_to_b64str(x),
    }


np_array_to_b64str = np.array([1.1, 2.1], dtype="float32")
np_array_to_b64str_b64 = "zcyMP2ZmBkA="


class TestDataclassToJsonMixin:
    @pytest.fixture
    def test_dataclass_instance(self):
        data = TestDataclass2(
            box=cb.Box((0, 0, 1, 1)),
            boxes=cb.Boxes([(0, 0, 1, 1)]),
            keypoints=cb.Keypoints([(0, 1), (1, 0)]),
            keypoints_list=cb.KeypointsList(
                [[(0, 1), (1, 0)], [(0, 1), (2, 0)]]
            ),
            polygon=cb.Polygon([(0, 0), (1, 0), (1, 1)]),
            polygons=cb.Polygons([[(0, 0), (1, 0), (1, 1)]]),
            np_bool=np.bool_(True),
            np_float=np.float64(1),
            np_number=np.array(1),
            np_array=np.array([1, 2]),
            image=MOCK_IMAGE,
            np_array_to_b64str=np_array_to_b64str,
            py_dict={"box": cb.Box((0, 0, 1, 1))},
            py_str="test",
            py_int=1,
            py_float=0.6,
            py_tuple=(1, 1),
            py_pow=1e10,
        )
        return data

    @pytest.fixture
    def test_expected(self):
        return {
            "box": [0, 0, 1, 1],
            "boxes": [[0, 0, 1, 1]],
            "keypoints": [[0, 1], [1, 0]],
            "keypoints_list": [[[0, 1], [1, 0]], [[0, 1], [2, 0]]],
            "polygon": [[0, 0], [1, 0], [1, 1]],
            "polygons": [[[0, 0], [1, 0], [1, 1]]],
            "np_bool": True,
            "np_float": 1.0,
            "np_number": 1,
            "np_array": [1, 2],
            "image": BASE64_PNG_IMAGE,
            "np_array_to_b64str": np_array_to_b64str_b64,
            "py_dict": {"box": [0, 0, 1, 1]},
            "py_str": "test",
            "py_int": 1,
            "py_float": 0.6,
            "py_tuple": [1, 1],
            "py_pow": 1e10,
        }

    def test_be_jsonable(self, test_dataclass_instance, test_expected):
        assert test_dataclass_instance.be_jsonable() == test_expected


def test_dict_to_jsonable_converts_enum_to_name():
    class LocalEnum(Enum):
        A = 1

    out = dict_to_jsonable({"e": LocalEnum.A})
    assert out["e"] == "A"


def test_dict_to_jsonable_warns_when_output_is_not_jsonable():
    with pytest.warns(UserWarning, match="not JSON serializable"):
        out = dict_to_jsonable({"bad": {1, 2}})
    assert out["bad"] == {1, 2}


def test_dataclass_copy_mixin_requires_dataclass_instance():
    class NotADataclass(DataclassCopyMixin):
        def __init__(self) -> None:
            self.x = 1

    with pytest.raises(TypeError, match="not a dataclass"):
        NotADataclass().__copy__()
