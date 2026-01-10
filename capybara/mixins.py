import json
from collections import OrderedDict
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import asdict
from enum import Enum
from typing import Any, TypeVar, cast
from warnings import warn

import numpy as np
from dacite import from_dict

from .structures import Box, Boxes, Keypoints, KeypointsList, Polygon, Polygons

__all__ = [
    "DataclassCopyMixin",
    "DataclassToJsonMixin",
    "EnumCheckMixin",
    "dict_to_jsonable",
]


def dict_to_jsonable(
    d: Mapping[str, Any],
    jsonable_func: Mapping[str, Callable[[Any], Any]] | None = None,
    dict_factory: Callable[[], MutableMapping[str, Any]] = OrderedDict,
) -> MutableMapping[str, Any]:
    out = dict_factory()
    for k, v in d.items():
        if jsonable_func is not None and k in jsonable_func:
            out[k] = jsonable_func[k](v)
        else:
            if isinstance(v, (Box, Boxes)):
                out[k] = (
                    v.convert("XYXY").numpy().astype(float).round().tolist()
                )
            elif isinstance(v, (Keypoints, KeypointsList, Polygon, Polygons)):
                out[k] = v.numpy().astype(float).round().tolist()
            elif isinstance(v, (np.ndarray, np.generic)):
                # include array and scalar, if you want jsonable image please use jsonable_func
                out[k] = v.tolist()
            elif isinstance(v, (list, tuple)):
                out[k] = [
                    dict_to_jsonable(x, jsonable_func)
                    if isinstance(x, dict)
                    else x
                    for x in v
                ]
            elif isinstance(v, Enum):
                out[k] = v.name
            elif isinstance(v, Mapping):
                out[k] = dict_to_jsonable(v, jsonable_func)
            else:
                out[k] = v

    try:
        json.dumps(out)
    except Exception as e:
        warn(str(e), stacklevel=2)

    return out


_EnumT = TypeVar("_EnumT", bound="EnumCheckMixin")


class EnumCheckMixin:
    @classmethod
    def obj_to_enum(cls: type[_EnumT], obj: Any) -> _EnumT:
        if isinstance(obj, str):
            try:
                return cast(_EnumT, getattr(cls, obj))
            except AttributeError:
                pass
        elif isinstance(obj, cls):
            return obj
        elif isinstance(obj, int):
            try:
                return cast(_EnumT, cast(Any, cls)(obj))
            except ValueError:
                pass

        raise ValueError(f"{obj} is not correct for {cls.__name__}")


class DataclassCopyMixin:
    def __copy__(self):
        dataclass_fields = getattr(self, "__dataclass_fields__", None)
        if dataclass_fields is None:
            raise TypeError(
                f"{self.__class__.__name__} is not a dataclass instance."
            )
        field_names = cast(dict[str, Any], dataclass_fields).keys()
        return self.__class__(
            **{field: getattr(self, field) for field in field_names}
        )

    def __deepcopy__(self, memo):
        out = asdict(cast(Any, self), dict_factory=OrderedDict)
        return from_dict(data_class=self.__class__, data=out)


class DataclassToJsonMixin:
    jsonable_func = None

    def be_jsonable(self, dict_factory=OrderedDict):
        d = asdict(cast(Any, self), dict_factory=dict_factory)
        return dict_to_jsonable(
            d, jsonable_func=self.jsonable_func, dict_factory=dict_factory
        )
