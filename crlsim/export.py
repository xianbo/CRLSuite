import json
from pathlib import Path
from typing import Any, Union

import h5py
import numpy as np

from .models import SimulationResult
from .wavefront import WavefrontResult


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default)


def _write_mapping(group: h5py.Group, values: dict[str, Any]) -> None:
    for key, value in values.items():
        name = str(key).replace("/", "_")
        if isinstance(value, dict):
            _write_mapping(group.create_group(name), value)
        elif isinstance(value, (str, bytes)):
            group.attrs[name] = value
        elif value is None:
            group.attrs[name] = "null"
        else:
            array = np.asarray(value)
            if array.dtype.kind in "biufc":
                group.create_dataset(name, data=array)
            else:
                group.attrs[name] = _json_dumps(value)


def _write_result(
    group: Union[h5py.File, h5py.Group],
    result: Union[SimulationResult, WavefrontResult],
    profile_name: str,
) -> None:
    if isinstance(result, SimulationResult):
        _write_mapping(group.create_group("envelope"), result.envelope)
        _write_mapping(group.create_group("focus"), result.focus)
    else:
        group.create_dataset("coordinates_x_m", data=result.coordinates_x)
        if result.coordinates_y is not None:
            group.create_dataset("coordinates_y_m", data=result.coordinates_y)
        group.attrs["transmission"] = result.transmission
    _write_mapping(group.create_group("metadata"), result.metadata)
    if result.stats is not None:
        _write_mapping(group.create_group("statistics"), result.stats)
    if result.profile is not None:
        group.create_dataset(profile_name, data=result.profile)
    if result.profile_coordinates is not None:
        group.create_dataset("profile_coordinates_m", data=result.profile_coordinates)
    if result.profile_extent is not None:
        group.create_dataset("profile_extent_m", data=result.profile_extent)


def save_hdf5(
    path: Path,
    configuration: dict[str, Any],
    result: Union[SimulationResult, WavefrontResult],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "CRLSuite results"
        handle.attrs["schema_version"] = 1
        handle.attrs["configuration_json"] = _json_dumps(configuration)
        _write_mapping(handle.create_group("configuration"), configuration)
        _write_result(handle, result, "detector_profile")


def save_hdf5_multiple(
    path: Path,
    configuration: dict[str, Any],
    results: list[Union[SimulationResult, WavefrontResult]],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "CRLSuite multi-detector results"
        handle.attrs["schema_version"] = 1
        handle.attrs["configuration_json"] = _json_dumps(configuration)
        _write_mapping(handle.create_group("configuration"), configuration)
        detectors = handle.create_group("detectors")
        for index, result in enumerate(results):
            group = detectors.create_group(str(index + 1))
            group.attrs["z_m"] = result.detector_z
            _write_result(group, result, "profile")
