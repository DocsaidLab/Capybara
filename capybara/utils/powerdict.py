from collections.abc import Mapping
from pprint import pprint
from typing import Any

from .files_utils import (
    dump_json,
    dump_pickle,
    dump_yaml,
    load_json,
    load_pickle,
    load_yaml,
)

__all__ = ["PowerDict"]

_MISSING = object()


class PowerDict(dict):
    def __init__(self, d=None, **kwargs):
        """
        This class is used to create a namespace dictionary with freeze and melt functions.

        Args:
            d (dict, optional): dictionary to cfg. Defaults to None.
        """
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

        self._frozen = False

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __set(self, key, value):
        if not self.is_frozen:
            if isinstance(value, Mapping) and not isinstance(
                value, self.__class__
            ):
                value = self.__class__(value)
            if isinstance(value, (list, tuple)):
                value = [
                    self.__class__(v) if isinstance(v, dict) else v
                    for v in value
                ]
            super().__setattr__(key, value)
            super().__setitem__(key, value)
        else:
            raise ValueError(f"PowerDict is frozen. '{key}' cannot be set.")

    def __del(self, key):
        if not self.is_frozen:
            super().__delattr__(key)
            super().__delitem__(key)
        else:
            raise ValueError(f"PowerDict is frozen. '{key}' cannot be del.")

    def __setattr__(self, key, value):
        if key == "_frozen":
            super().__setattr__(key, value)
        else:
            self.__set(key, value)

    def __delattr__(self, key):
        if key == "_frozen":
            raise KeyError("Can not del '_frozen'.")
        else:
            self.__del(key)

    def __setitem__(self, key, value):
        if key == "_frozen":
            raise KeyError("Can not set '_frozen' as an item.")
        else:
            self.__set(key, value)

    def __delitem__(self, key):
        if key == "_frozen":
            raise KeyError("There is not _frozen in items.")
        else:
            self.__del(key)

    def update(self, e=None, **f):
        if self.is_frozen:
            raise ValueError("PowerDict is frozen. Update is not allowed.")

        if e is None:
            d: dict = {}
        else:
            d = dict(e)
        d.update(f)
        for key, value in d.items():
            setattr(self, key, value)

    def pop(self, key, default=_MISSING):
        if self.is_frozen:
            raise ValueError("PowerDict is frozen. Pop is not allowed.")

        if key in self:
            value = self[key]
            del self[key]
            return value

        if default is _MISSING:
            raise KeyError(key)
        return default

    def freeze(self):
        self._frozen = True
        for v in dict.values(self):
            if isinstance(v, PowerDict):
                v.freeze()
            if isinstance(v, (list, tuple)):
                for vv in v:
                    if isinstance(vv, PowerDict):
                        vv.freeze()

    def melt(self):
        self._frozen = False
        for v in dict.values(self):
            if isinstance(v, PowerDict):
                v.melt()
            if isinstance(v, (list, tuple)):
                for vv in v:
                    if isinstance(vv, PowerDict):
                        vv.melt()

    @property
    def is_frozen(self):
        return getattr(self, "_frozen", False)

    def __deepcopy__(self, memo):
        if self._frozen:
            raise Warning("PowerDict is frozen and cannot be copy.")
        return self.__class__(self)

    def to_dict(self):
        out = {}
        for k, v in dict.items(self):
            if isinstance(v, PowerDict):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [
                    x.to_dict() if isinstance(x, PowerDict) else x for x in v
                ]
            else:
                out[k] = v

        return out

    def to_json(self, path):
        d = self.to_dict()
        return dump_json(d, path)

    def to_yaml(self, path):
        d = self.to_dict()
        return dump_yaml(d, path)

    def to_txt(self, path):
        d = self.to_dict()
        with open(path, "w") as f:
            pprint(d, f)

    def to_pickle(self, path):
        d = self.to_dict()
        return dump_pickle(d, path)

    @classmethod
    def load_json(cls, path):
        return cls(load_json(path))

    @classmethod
    def load_pickle(cls, path):
        return cls(load_pickle(path))

    @classmethod
    def load_yaml(cls, path):
        return cls(load_yaml(path))
