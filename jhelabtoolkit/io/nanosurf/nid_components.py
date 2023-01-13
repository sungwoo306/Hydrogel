from __future__ import annotations
from typing import Literal, BinaryIO, ByteString

import pydantic
from pydantic.dataclasses import dataclass
import numpy as np
import xarray as xr


@dataclass(frozen=True)
class ChannelHeader:
    version: int
    points: int
    lines: int
    frame: pydantic.constr(to_lower=True)
    curline: int

    dim0name: pydantic.constr(to_lower=True) = "dim0"
    dim0unit: str | None = None
    dim0range: float = 0.0
    dim0min: float = 0.0

    dim1name: pydantic.constr(to_lower=True) = "dim1"
    dim1unit: str | None = None
    dim1range: float = 0.0
    dim1min: float = 0.0

    dim2name: pydantic.constr(to_lower=True) = "dim2"
    dim2unit: str | None = None
    dim2range: float = 0.0
    dim2min: float = 0.0

    savemode: Literal["Binary"] = "Binary"
    savebits: pydantic.conint(multiple_of=8) = 16
    savesign: Literal["Signed", "Unsigned"] = "Signed"
    saveorder: Literal["Intel"] = "Intel"

    @property
    def measurement_type(self) -> str:
        return self.frame

    @property
    def data_name(self) -> str:
        return self.dim2name

    @property
    def data_unit(self) -> str:
        return self.dim2unit

    @property
    def data_shape(self) -> tuple[int, int]:
        return (self.points, self.lines)

    @property
    def data_dtype(self) -> np.dtype:
        endian = "<" if self.saveorder == "Intel" else ">"
        data_format = "i" if self.savesign == "Signed" else "u"
        num_bytes = self.savebits // 8
        return np.dtype(f"{endian}{data_format}{num_bytes}")

    @property
    def data_num_bytes(self) -> int:
        dtype = self.data_dtype
        return self.points * self.lines * dtype.itemsize

    @property
    def data_min(self) -> float:
        return self.dim2min

    @property
    def data_range(self) -> float:
        return self.dim2range


class Channel:
    def __init__(self, header: ChannelHeader, data: np.ndarray | None = None):
        self.header = header
        self.data = data

    def load_data_from_stream(self, binary_stream: BinaryIO) -> Channel:
        binary_data = binary_stream.read(self.header.data_num_bytes)
        raw_data = self._decode_binary_data(binary_data)
        data = self._scale_data(raw_data)
        return Channel(self.header, data)

    def _decode_binary_data(self, binary_data: ByteString) -> np.ndarray:
        return np.frombuffer(binary_data, dtype=self.header.data_dtype).reshape(
            self.header.data_shape
        )

    def _scale_data(self, raw_data: np.ndarray) -> np.ndarray:
        dtype_info = np.iinfo(self.header.data_dtype)
        data_max, data_min = dtype_info.max, dtype_info.min
        normalized_data = (raw_data - data_min) / (data_max - data_min)
        return normalized_data * self.header.data_range + self.header.data_min

    def to_dataarray(self) -> xr.DataArray:
        coords = self._make_dataarray_coords()
        dataarray = xr.DataArray(
            self.data,
            coords=list(coords),
            name=self.header.data_name,
            attrs={"unit": self.header.data_unit},
        )
        return dataarray

    def _make_dataarray_coords(self) -> tuple[xr.Variable, xr.Variable]:
        header = self.header
        data0 = np.linspace(0, header.dim0range, header.points) + header.dim0min
        data1 = np.linspace(0, header.dim1range, header.lines) + header.dim1min
        coord0 = xr.Variable(
            dims=header.dim0name, data=data0, attrs={"unit": header.dim0unit}
        )
        coord1 = xr.Variable(
            dims=header.dim1name, data=data1, attrs={"unit": header.dim1unit}
        )
        return coord0, coord1
