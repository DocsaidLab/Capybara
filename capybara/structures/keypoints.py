from typing import Any, List, Tuple, Union
from warnings import warn

import matplotlib
import numpy as np

from ..typing import _Number
from .boxes import Box, Boxes

__all__ = ['Keypoints', 'KeypointsList']


_Keypoints = Union[
    np.ndarray,
    List[np.ndarray],
    List[Tuple[_Number, _Number]],
    List[Tuple[_Number, _Number, _Number]],
    "Keypoints",
]

_KeypointsList = Union[
    np.ndarray,
    List[_Keypoints],
]


class Keypoints:
    '''
    This structure has shape (K, 3) or (K, 2) where K is the number of keypoints.
    The visibility flag follows the COCO format and must be one of three integers:
    * v=0: not labeled (in which case x=y=0)
    * v=1: labeled but not visible
    * v=2: labeled and visible
    '''

    def __init__(self, array: _Keypoints, cmap='rainbow', is_normalized: bool = False):
        self._array = self._check_valid_array(array)
        steps = np.linspace(0., 1., self._array.shape[-2])
        color_map = matplotlib.colormaps[cmap]
        self._point_colors = np.array(color_map(steps, bytes=True))[..., :3].tolist()
        self._is_normalized = is_normalized

    def __len__(self) -> int:
        return self._array.shape[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self._array)})"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False
        return np.allclose(self._array, value._array)

    def _check_valid_array(self, array: Any) -> np.ndarray:
        cond1 = isinstance(array, np.ndarray)
        cond2 = isinstance(array, list) and all(isinstance(x, (tuple, np.ndarray)) for x in array)
        cond3 = isinstance(array, self.__class__)

        if not (cond1 or cond2 or cond3):
            raise TypeError(f"Input array is not {_Keypoints}, but got {type(array)}.")

        if cond3:
            array = array.numpy()
        else:
            array = np.array(array, dtype='float32')

        if not array.ndim == 2:
            raise ValueError(f"Input array ndim = {array.ndim} is not 2, which is invalid.")

        if not array.shape[-1] in [2, 3]:
            raise ValueError(f"Input array's shape[-1] = {array.shape[-1]} is not in [2, 3], which is invalid.")

        if array.shape[-1] == 3 and not ((array[..., 2] <= 2).all() and (array[..., 2] >= 0).all()):
            raise ValueError('Given array is invalid because of its labels. (array[..., 2])')
        return array.copy()

    def numpy(self) -> np.ndarray:
        return self._array.copy()

    def copy(self) -> Any:
        return self.__class__(self._array)

    def shift(self, shift_x: float, shift_y: float) -> "Keypoints":
        array = self._array.copy()
        array[..., :2] += (shift_x, shift_y)
        return self.__class__(array)

    def scale(self, fx: float, fy: float) -> "Keypoints":
        array = self._array.copy()
        array[..., :2] *= (fx, fy)
        return self.__class__(array)

    def normalize(self, w: float, h: float) -> "Keypoints":
        if self.is_normalized:
            warn(f'Normalized keypoints are forced to do normalization.')
        arr = self._array.copy()
        arr[..., :2] = arr[..., :2] / (w, h)
        kpts = self.__class__(arr)
        kpts._is_normalized = True
        return kpts

    def denormalize(self, w: float, h: float) -> "Keypoints":
        if not self.is_normalized:
            warn(f'Non-normalized keypoints is forced to do denormalization.')
        arr = self._array.copy()
        arr[..., :2] = arr[..., :2] * (w, h)
        kpts = self.__class__(arr)
        kpts._is_normalized = False
        return kpts

    @ property
    def is_normalized(self) -> bool:
        return self._is_normalized

    @ property
    def point_colors(self) -> List[Tuple[int, int, int]]:
        return [
            tuple([int(x) for x in cs])
            for cs in self._point_colors
        ]

    @ point_colors.setter
    def set_point_colors(self, cmap: str):
        steps = np.linspace(0., 1., self._array.shape[-2])
        self._point_colors = matplotlib.colormaps[cmap](steps, bytes=True)


class KeypointsList:

    def __init__(self, array: _KeypointsList, cmap='rainbow', is_normalized: bool = False) -> None:
        self._array = self._check_valid_array(array).copy()
        self._is_normalized = is_normalized
        if len(self._array):
            steps = np.linspace(0., 1., self._array.shape[-2])
            self._point_colors = matplotlib.colormaps[cmap](steps, bytes=True)
        else:
            self._point_colors = None

    def __len__(self) -> int:
        return self._array.shape[0]

    def __getitem__(self, item) -> Any:
        if isinstance(item, int):
            return Keypoints(self._array[item], is_normalized=self.is_normalized)
        return KeypointsList(self._array[item], is_normalized=self.is_normalized)

    def __setitem__(self, item, value):
        if not isinstance(value, (Keypoints, KeypointsList)):
            raise TypeError(f'Input value is not a keypoint or keypoints')

        if isinstance(item, (int, np.ndarray, list, slice)):
            self._array[item] = value._array

    def __iter__(self) -> Any:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self._array)})"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            return False
        return np.allclose(self._array, value._array)

    def _check_valid_array(self, array: Any) -> np.ndarray:
        cond1 = isinstance(array, np.ndarray)
        cond2 = isinstance(array, list) and len(array) == 0
        cond3 = isinstance(array, list) and \
            all(isinstance(x, (np.ndarray, Keypoints)) for x in array) or \
            all(isinstance(y, tuple) for x in array for y in x)
        cond4 = isinstance(array, self.__class__)

        if not (cond1 or cond2 or cond3 or cond4):
            raise TypeError(f"Input array is not {_KeypointsList}, but got {type(array)}.")

        if cond4:
            array = array.numpy()
        elif len(array) and isinstance(array[0], Keypoints):
            array = np.array([x.numpy() for x in array], dtype='float32')
        else:
            array = np.array(array, dtype='float32')

        if len(array) == 0:
            return array

        if array.ndim != 3:
            raise ValueError(f"Input array's ndim = {array.ndim} is not 3, which is invalid.")

        if array.shape[-1] not in [2, 3]:
            raise ValueError(f"Input array's shape[-1] = {array.shape[-1]} is not 2 or 3, which is invalid.")

        if array.shape[-1] == 3 and not ((array[..., 2] <= 2).all() and (array[..., 2] >= 0).all()):
            raise ValueError('Given array is invalid because of its labels. (array[..., 2])')

        return array

    def numpy(self) -> np.ndarray:
        return self._array.copy()

    def copy(self) -> Any:
        return self.__class__(self._array)

    def shift(self, shift_x: float, shift_y: float) -> Any:
        array = self._array.copy()
        array[..., :2] += (shift_x, shift_y)
        return self.__class__(array)

    def scale(self, fx: float, fy: float) -> Any:
        array = self._array.copy()
        array[..., :2] *= (fx, fy)
        return self.__class__(array)

    def normalize(self, w: float, h: float) -> "KeypointsList":
        if self.is_normalized:
            warn(f'Normalized keypoints_list is forced to do normalization.')
        arr = self._array.copy()
        arr[..., :2] = arr[..., :2] / (w, h)
        kpts_list = self.__class__(arr)
        kpts_list._is_normalized = True
        return kpts_list

    def denormalize(self, w: float, h: float) -> "KeypointsList":
        if not self.is_normalized:
            warn(f'Non-normalized box is forced to do denormalization.')
        arr = self._array.copy()
        arr[..., :2] = arr[..., :2] * (w, h)
        kpts_list = self.__class__(arr)
        kpts_list._is_normalized = False
        return kpts_list

    @ property
    def is_normalized(self) -> bool:
        return self._is_normalized

    @ property
    def point_colors(self):
        return [tuple(c) for c in self._point_colors[..., :3].tolist()]

    @ point_colors.setter
    def set_point_colors(self, cmap: str):
        steps = np.linspace(0., 1., self._array.shape[-2])
        self._point_colors = matplotlib.colormaps[cmap](steps, bytes=True)

    @ classmethod
    def cat(cls, keypoints_lists: List["KeypointsList"]) -> "KeypointsList":
        '''
        Concatenates a list of KeypointsList into a single KeypointsList

        Raises:
            TypeError: Check keypoints_list is a list.
            ValueError: check keypoints_list is not empty.
            TypeError: check elements in keypoints_list are Keypoints.

        Returns:
            Keypoints: the concatenated Keypoints
        '''
        if not isinstance(keypoints_lists, list):
            raise TypeError('Given keypoints_list should be a list.')

        if len(keypoints_lists) == 0:
            raise ValueError('Given keypoints_list is empty.')

        if not all(isinstance(keypoints_list, KeypointsList) for keypoints_list in keypoints_lists):
            raise TypeError('All type of elements in keypoints_lists must be KeypointsList.')

        return cls(np.concatenate([keypoints_list.numpy() for keypoints_list in keypoints_lists], axis=0))
