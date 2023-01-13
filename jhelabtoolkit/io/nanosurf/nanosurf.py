from pathlib import Path
from typing import ByteString, BinaryIO, Mapping
import mmap
from io import BytesIO
from configparser import ConfigParser
import re

import xarray as xr

from jhelabtoolkit.io.nanosurf.nid_components import ChannelHeader, Channel
from jhelabtoolkit.utils.xarray import (
    dataset_from_dataarrays,
    is_empty_dataset,
    is_zero_dataset,
)
from jhelabtoolkit.utils.misc import bin_key_value_list, filter_dict


def read_nid(filepath: str | Path) -> tuple[ConfigParser, dict[str, xr.Dataset]]:
    with open(filepath, "rb") as file:
        with mmap.mmap(file.fileno(), length=0, access=mmap.ACCESS_READ) as memorymap:
            config_binary, data_binary = split_config_and_data(memorymap)
            config = parse_config(config_binary)
            channel_headers = get_channel_headers(config)
            data = parse_data(data_binary, channel_headers)
    data = filter_dict(data, rejection_criteria=(is_empty_dataset, is_zero_dataset))
    return config, data


def split_config_and_data(
    binary_fileio: BinaryIO, data_delimiter: ByteString = b"#!"
) -> tuple[ByteString, ByteString]:
    config_end = binary_fileio.find(data_delimiter)
    config = binary_fileio[:config_end]
    data = binary_fileio[config_end + len(data_delimiter) :]
    return config, data


def parse_config(config_binary: ByteString) -> ConfigParser:
    config_list = config_binary.decode().split("\r\n")
    config_parser = ConfigParser(allow_no_value=True)
    config_parser.read_file(config_list)
    return config_parser


def get_channel_headers(config: ConfigParser) -> list[ChannelHeader]:
    header_regex = re.compile("DataSet-\d+:\d+")
    headers = [
        ChannelHeader(**section)
        for name, section in config.items()
        if re.match(header_regex, name) is not None
    ]
    return headers


def parse_data(
    data_binary: ByteString, channel_headers: list[ChannelHeader]
) -> dict[str, xr.Dataset]:
    data_stream = BytesIO(data_binary)
    parsed_channels = [parse_channel(data_stream, hdr) for hdr in channel_headers]
    channel_group_dict = bin_key_value_list(parsed_channels)
    data = {
        group_name: dataset_from_dataarrays(group_data)
        for group_name, group_data in channel_group_dict.items()
    }
    return data


def parse_channel(
    binary_stream: BinaryIO, header: ChannelHeader
) -> tuple[str, xr.DataArray]:
    meas_type = header.measurement_type
    channel_empty = Channel(header)
    channel = channel_empty.load_data_from_stream(binary_stream)
    return meas_type, channel.to_dataarray()


def filter_dataset_info(dataset_info: Mapping[str, str]):
    dummy_key_regex = re.compile("-- \s+ --")
    return {
        k: v
        for k, v in dataset_info.items()
        if re.match(dummy_key_regex, k) is not None
    }
