from pathlib import Path
import re

import numpy as np
import xarray as xr


def read_scope_file(filepath: Path | str) -> xr.DataArray:
    comments, data = read_csv_with_comments(filepath, comments="%", delimiter=";")
    label_list = comments[-1]
    names, units = list(zip(*[match_name_and_unit(l) for l in label_list]))
    output = xr.DataArray(
        data=data[:, 1],
        coords=[(names[0], data[:, 0], {"unit": units[0]})],
        name=names[1],
        attrs={"unit": units[1]},
    )
    return comments[:-1], output


def read_csv_with_comments(
    filepath: Path | str, comments: str = "%", delimiter: str = ";"
):
    with open(filepath, "r") as f:
        metadata = []
        for line in f:
            if not line.startswith(comments):
                break
            metadata.append(line.lstrip(comments).rstrip("\n").split(delimiter))
        body = np.loadtxt(f, dtype="float", comments=comments, delimiter=delimiter)
    return metadata, body


def match_name_and_unit(label: str) -> tuple[str, str]:
    match_result = re.match(r"\s*([\w+\s*]+) \((\S+)\)", label)
    name, unit = match_result.groups()
    return name.lower(), unit
