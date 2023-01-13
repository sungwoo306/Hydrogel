from pathlib import Path
import mmap
from configparser import ConfigParser, SectionProxy
import re

import numpy as np


def get_dataset_sections(config: ConfigParser):
    section_regex = re.compile("DataSet-\d+:\d+")
    return filter(
        lambda name: re.match(section_regex, name) is not None, config.sections()
    )


def get_dtype(config: ConfigParser, name: str) -> np.dtype:
    section = config[name]
    assert (
        section["SaveMode"] == "Binary"
    ), "Reading non-binary data format is not supported!"
    endian = "<" if section["SaveOrder"] == "Intel" else ">"
    data_format = "i" if section["SaveSign"] == "Signed" else "u"
    num_bytes = int(section["SaveBits"]) // 8
    return np.dtype(f"{endian}{data_format}{num_bytes}")


def get_read_shape(config: ConfigParser, name: str) -> tuple[int, int]:
    section = config[name]
    dim0, dim1 = section["Points"], section["Lines"]
    return (int(dim0), int(dim1))


def get_scale_params(config: ConfigParser, name: str) -> tuple[float, float]:
    section = config[name]
    range_, offset = section["Dim2Range"], section["Dim2Min"]
    return float(range_), float(offset)


def get_read_length(shape: tuple[int, int], dtype: np.dtype) -> int:
    return shape[0] * shape[1] * dtype.itemsize


def scale_data(data: np.ndarray, datarange: float, dataoffset: float) -> np.ndarray:
    min_val = np.iinfo(data.dtype).min
    normalized_data = 1.0 + data / np.abs(min_val)
    return normalized_data * datarange + dataoffset


def get_data_name(config: ConfigParser, name: str) -> str:
    section = config[name]
    name, frame = section["Dim2Name"], section["Frame"]
    return f"{name.lower()}_{frame.lower()}"


def read_nid(filepath: str | Path):
    config_parser = ConfigParser(allow_no_value=True)
    with open(filepath, "rb") as file:
        with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as memorymap:
            data_start_ind = memorymap.find(b"#!")
            header = memorymap[:data_start_ind].decode()
            config_list = header.split("\r\n")
            config_parser.read_file(config_list)

            shapes = [
                get_read_shape(config_parser, name)
                for name in get_dataset_sections(config_parser)
            ]
            dtypes = [
                get_dtype(config_parser, name)
                for name in get_dataset_sections(config_parser)
            ]
            read_lengths = [get_read_length(s, d) for s, d in zip(shapes, dtypes)]

            memorymap.seek(data_start_ind + 2)
            raw_data = [
                np.frombuffer(memorymap.read(n), d)
                for n, d in zip(read_lengths, dtypes)
            ]
            scale_params = [
                get_scale_params(config_parser, name)
                for name in get_dataset_sections(config_parser)
            ]
            data = [
                scale_data(raw, *p).reshape(s)
                for raw, p, s in zip(raw_data, scale_params, shapes)
            ]
            names = [
                get_data_name(config_parser, name)
                for name in get_dataset_sections(config_parser)
            ]

    return config_parser, dict(zip(names, data))
