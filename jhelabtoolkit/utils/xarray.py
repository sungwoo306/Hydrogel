from typing import Iterable
from itertools import chain

import xarray as xr
import numpy as np


def dataset_from_dataarrays(
    dataarrays: Iterable[xr.DataArray], name: str | None = None
) -> xr.Dataset:
    ds_dict = {da.name: da for da in dataarrays}
    return xr.Dataset(data_vars=ds_dict, attrs={"name": name})


def is_empty_dataset(dataset: xr.Dataset) -> bool:
    return len(dataset) == 0


def is_zero_dataset(dataset: xr.Dataset) -> bool:
    data, coords = dataset.data_vars.values(), dataset.coords.values()
    is_zero = all(is_zero_array(v.to_numpy()) for v in chain(data, coords))
    return is_zero


def is_zero_array(array: np.ndarray) -> bool:
    return np.all(array == 0.0)
