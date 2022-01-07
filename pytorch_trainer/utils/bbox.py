from copy import copy
from pathlib import Path
from re import sub
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

from bbox.utils import Pairable, pair


class BBox:
    origin = 1

    def __init__(
        self,
        arr: ArrayLike,
        mode: str = "xyxy",
        origin: float = 1,
        copy: bool = False,
    ) -> None:
        """mode = 'xyxy' | 'xywh' | 'ccwh'"""

        if mode not in ("xyxy", "xywh", "ccwh"):
            raise ValueError("mode must be 'xyxy', 'xywh', or 'ccwh'")

        if isinstance(arr, np.ndarray):
            if copy:
                arr = arr.copy()
        else:
            arr = np.array(arr)

        if arr.shape[-1] != 4:
            raise ValueError(
                "The last dimension of the array must be of size 4, "
                "but the given array shape is " + str(arr.shape)
            )

        if mode == "xywh":
            arr[..., 2:] += arr[..., :2] - 1
        elif mode == "ccwh":
            arr[..., :2] -= (arr[..., 2:] - 1) / 2
            arr[..., 2:] += arr[..., :2] - 1

        self._xyxy = arr
        if origin != self.origin:
            self._xyxy += self.origin - origin

    @property
    def ndim(self) -> int:
        return self._xyxy.ndim - 1

    @property
    def shape(self) -> tuple[int, ...]:
        return self._xyxy.shape[:-1]

    @property
    def _x0(self) -> np.ndarray:
        return self._xyxy[..., 0]

    @property
    def _y0(self) -> np.ndarray:
        return self._xyxy[..., 1]

    @property
    def _x1(self) -> np.ndarray:
        return self._xyxy[..., 2]

    @property
    def _y1(self) -> np.ndarray:
        return self._xyxy[..., 3]

    @property
    def _center(self) -> np.ndarray:
        return (self._xyxy[..., :2] + self._xyxy[..., 2:]) / 2

    @property
    def center(self) -> tuple[np.ndarray, np.ndarray]:
        cxcy = self._center
        return cxcy[..., 0], cxcy[..., 1]

    @property
    def _size(self) -> np.ndarray:
        return self._xyxy[..., 2:] - self._xyxy[..., :2] + 1

    @property
    def size(self) -> tuple[np.ndarray, np.ndarray]:
        wh = self._size
        return wh[..., 0], wh[..., 1]

    @property
    def area(self) -> np.ndarray:
        width, height = self.size
        return width * height * self.is_valid()

    def __repr__(self) -> str:
        return f"BBox({repr(self._xyxy)})"

    def __str__(self) -> str:
        return f"BBox(x0={self._x0}, y0={self._y0}, x1={self._x1}, y1={self._y1})"

    def __copy__(self) -> "BBox":
        return BBox(self._xyxy, copy=True)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key) -> "BBox":
        return BBox(self._xyxy[key])

    def __setitem__(self, key, value) -> None:
        self._xyxy[key] = value

    def __and__(self, other: "BBox") -> "BBox":
        lt = np.maximum(self._xyxy[..., :2], other._xyxy[..., :2])
        rb = np.minimum(self._xyxy[..., 2:], other._xyxy[..., 2:])
        return BBox(np.concatenate((lt, rb), -1))

    def __round__(self) -> "BBox":
        return BBox(self._xyxy.round())

    def __iadd__(self, point: Pairable[float]) -> "BBox":
        dx, dy = pair(point)
        self._xyxy[..., 0::2] += dx
        self._xyxy[..., 1::2] += dy
        return self

    def __add__(self, point: Pairable[float]) -> "BBox":
        bb = copy(self)
        bb += point
        return bb

    def __isub__(self, point: Pairable[float]) -> "BBox":
        dx, dy = pair(point)
        self._xyxy[..., 0::2] -= dx
        self._xyxy[..., 1::2] -= dy
        return self

    def __sub__(self, point: Pairable[float]) -> "BBox":
        bb = copy(self)
        bb -= point
        return bb

    def is_valid(self) -> np.ndarray:
        w, h = self.size
        return (w > 0) & (h > 0)


class AxisError(ValueError, IndexError):
    def __init__(self, axis: int, ndim: int) -> None:
        super().__init__(f"axis {axis} is out of bounds for BBox of dimension {ndim}")


def stack(bboxes: Sequence[BBox], axis: int = 0) -> BBox:
    ndim = bboxes[0].ndim
    if axis not in range(-ndim - 1, ndim + 1):
        raise AxisError(axis, ndim)
    if axis < 0:
        axis -= 1
    arr = np.stack([bbox._xyxy for bbox in bboxes], axis)
    return BBox(arr)


def loadtxt(
    fname: Union[str, Path],
    mode: str = "xywh",
    origin: float = 1,
    dtype: type = float,
    delimiter: Optional[str] = ",",
    usecols: Optional[Union[int, Sequence[int]]] = None,
    ndmin: Optional[int] = 2,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: Optional[int] = None,
):
    if delimiter:
        arr = np.loadtxt(
            fname, dtype, delimiter=delimiter, usecols=usecols, ndmin=ndmin
        )
    else:
        with open(fname) as f:
            arr_str = [sub(",|\t", " ", line) for line in f]
        arr = np.loadtxt(arr_str, dtype, usecols=usecols, ndmin=ndmin)
    arr = arr[start:stop:step]
    return BBox(arr, mode, origin)
