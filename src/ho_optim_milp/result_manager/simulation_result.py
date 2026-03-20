"""Module to load and handle simulation results from an HDF5 file."""

import json
import os
import warnings
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from .episode_result import EpisodeResult


def _decode_scalar(value: Any) -> Any:
    """Decode HDF5 scalar values to plain Python objects where possible."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        return value.item()
    raise ValueError(f"{value} has no 'item'")


def _group_attrs_to_dict(group: h5py.Group) -> dict[str, Any]:
    """Convert HDF5 group attributes to a plain Python dict."""
    return {key: _decode_scalar(value) for key, value in group.attrs.items()}


def _read_json_dataset(group: h5py.Group, name: str) -> Any:
    """Read a JSON dataset and parse it."""
    obj = group[name]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f'"{group.name}/{name}" is not an HDF5 dataset')

    raw = obj[()]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    elif not isinstance(raw, str):
        raise TypeError(
            f'"{group.name}/{name}" must contain str or bytes, '
            f"got {type(raw).__name__}"
        )

    return json.loads(raw)


class SimulationResults:
    """Class to load and handle merged simulation results from an HDF5 file."""

    def __init__(self, env: Any = None) -> None:
        """Initialise an empty ``SimulationResults`` container.

        Parameters
        ----------
        env:
            Optional environment object retained for back-reference.
        """
        self.env = env

        self.n_eps_loaded = 0
        self.episode_results: list[EpisodeResult] = []

        self.meta_attrs: dict[str, Any] = {}
        self.config_common: dict[str, Any] = {}
        self.included_keys: list[str] = []

        self._file_path: str | None = None
        self._file_info: dict[str, Any] | None = None

    @staticmethod
    def _episode_names(f: h5py.File) -> list[str]:
        """Return sorted episode group names."""
        return sorted(
            name
            for name, obj in f.items()
            if isinstance(obj, h5py.Group) and name.startswith("ep_")
        )

    @staticmethod
    def _summarize_dataset(ds: h5py.Dataset) -> dict[str, Any]:
        """Return basic metadata for one dataset."""
        if ds.size is None:
            raise TypeError(f'Dataset "{ds.name}" has size None')

        return {
            "shape": tuple(int(x) for x in ds.shape),
            "dtype": str(ds.dtype),
            "size": int(ds.size),
        }

    @classmethod
    def _validate_file(cls, file_path: str) -> dict[str, Any]:
        """Validate basic structure of the merged HDF5 file."""
        try:
            with h5py.File(file_path, "r") as f:
                info: dict[str, Any] = {
                    "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
                    "top_level_groups": list(f.keys()),
                }

                if "meta" not in f or not isinstance(f["meta"], h5py.Group):
                    raise ValueError('Missing top-level group "meta"')

                meta = f["meta"]
                if not isinstance(meta, h5py.Group):
                    raise TypeError('"meta" is not an HDF5 group')

                info["meta_attrs"] = _group_attrs_to_dict(meta)
                info["meta_datasets"] = list(meta.keys())

                if "config_common" not in meta:
                    raise ValueError('Missing dataset "meta/config_common"')
                if "included_keys" not in meta:
                    raise ValueError('Missing dataset "meta/included_keys"')

                config_common = _read_json_dataset(meta, "config_common")
                included_keys = _read_json_dataset(meta, "included_keys")

                if not isinstance(config_common, dict):
                    raise TypeError('"meta/config_common" must decode to dict')
                if not isinstance(included_keys, list):
                    raise TypeError('"meta/included_keys" must decode to list')

                info["config_common_keys"] = list(config_common.keys())
                info["included_keys"] = included_keys

                episode_names = cls._episode_names(f)
                if not episode_names:
                    raise ValueError(f"No valid episode groups found in {file_path}")

                info["n_episodes"] = len(episode_names)
                info["episodes"] = episode_names

                first_ep = episode_names[0]
                first_group = f[first_ep]
                if not isinstance(first_group, h5py.Group):
                    raise ValueError(f'Episode "{first_ep}" is not a valid group')

                available_keys = sorted(
                    key
                    for key, obj in first_group.items()
                    if isinstance(obj, h5py.Dataset)
                )
                info["available_keys_first_episode"] = available_keys
                info["missing_keys_first_episode"] = sorted(
                    set(included_keys) - set(available_keys)
                )
                info["extra_keys_first_episode"] = sorted(
                    set(available_keys) - set(included_keys)
                )

                if info["missing_keys_first_episode"]:
                    warnings.warn(
                        "Missing expected keys in first episode: "
                        f"{info['missing_keys_first_episode']}"
                    )
                if info["extra_keys_first_episode"]:
                    warnings.warn(
                        "Additional keys found in first episode: "
                        f"{info['extra_keys_first_episode']}"
                    )

                per_episode_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
                for ep_name in episode_names:
                    ep_group = f[ep_name]
                    if not isinstance(ep_group, h5py.Group):
                        continue

                    ep_shapes: dict[str, tuple[int, ...]] = {}
                    for key in included_keys:
                        if key not in ep_group:
                            continue
                        obj = ep_group[key]
                        if not isinstance(obj, h5py.Dataset):
                            raise TypeError(f'"{ep_name}/{key}" is not a dataset')
                        ep_shapes[key] = tuple(int(x) for x in obj.shape)

                    per_episode_shapes[ep_name] = ep_shapes

                info["per_episode_shapes"] = per_episode_shapes

                return info

        except Exception as exc:
            raise ValueError(
                f"Failed to validate HDF5 file {file_path}: {exc}"
            ) from exc

    @classmethod
    def load(
        cls,
        path: str,
        *,
        max_eps: int | None = None,
        max_steps: int | None = None,
        keys: list[str] | None = None,
        validate_data: bool = False,
        strict: bool = True,
    ) -> "SimulationResults":
        """
        Load results from a HDF5 file.

        Parameters
        ----------
        path:
            Path to the HDF5 file.
        max_eps:
            Maximum number of episodes to load. Use None to load all episodes.
        max_steps:
            Maximum number of time steps to load from each dataset along axis 0.
            Use None to load full datasets.
        keys:
            Subset of keys to load. If None, load all keys from meta/included_keys.
        validate_data:
            If True, run file validation before loading.
        strict:
            If True, require every selected key to exist in every episode and enforce
            consistent time dimension within an episode. If False, skip missing keys.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")

        instance = cls(env=None)
        instance._file_path = path

        if validate_data:
            instance._file_info = cls._validate_file(path)

        with h5py.File(path, "r") as f:
            if "meta" not in f or not isinstance(f["meta"], h5py.Group):
                raise ValueError('Missing top-level group "meta"')

            meta = f["meta"]
            if not isinstance(meta, h5py.Group):
                raise TypeError('"meta" is not an HDF5 group')

            instance.meta_attrs = _group_attrs_to_dict(meta)
            instance.config_common = _read_json_dataset(meta, "config_common")
            instance.included_keys = _read_json_dataset(meta, "included_keys")

            if not isinstance(instance.config_common, dict):
                raise TypeError('"meta/config_common" must decode to dict')
            if not isinstance(instance.included_keys, list):
                raise TypeError('"meta/included_keys" must decode to list')

            if keys is not None:
                load_keys = list(keys)
                invalid_keys = sorted(set(load_keys) - set(instance.included_keys))
                if invalid_keys:
                    raise ValueError(
                        f"Invalid keys requested: {invalid_keys}. "
                        f"Available keys: {instance.included_keys}"
                    )
            else:
                load_keys = list(instance.included_keys)

            episode_names = cls._episode_names(f)
            if max_eps is not None:
                episode_names = episode_names[:max_eps]

            for ep_name in episode_names:
                ep_group = f[ep_name]
                if not isinstance(ep_group, h5py.Group):
                    continue

                available_keys = {
                    key
                    for key, obj in ep_group.items()
                    if isinstance(obj, h5py.Dataset)
                }

                missing_keys = sorted(set(load_keys) - available_keys)
                if missing_keys and strict:
                    raise ValueError(
                        f'Episode "{ep_name}" is missing required keys: {missing_keys}'
                    )

                selected_keys = [key for key in load_keys if key in available_keys]
                if not selected_keys:
                    if strict:
                        raise ValueError(
                            f'Episode "{ep_name}" has no selected datasets'
                        )
                    continue

                episode_data: dict[str, NDArray[Any]] = {}
                n_steps_ref: int | None = None

                for key in selected_keys:
                    obj = ep_group[key]
                    if not isinstance(obj, h5py.Dataset):
                        raise TypeError(f'"{ep_name}/{key}" is not a dataset')

                    arr = np.asarray(obj)
                    if max_steps is not None:
                        arr = arr[:max_steps]

                    episode_data[key] = arr

                    if arr.ndim >= 1:
                        if n_steps_ref is None:
                            n_steps_ref = int(arr.shape[0])
                        elif strict and int(arr.shape[0]) != n_steps_ref:
                            raise ValueError(
                                f'Inconsistent time dimension in "{ep_name}": '
                                f'key "{key}" has shape {arr.shape}, expected first '
                                f"dimension {n_steps_ref}"
                            )

                ep_meta_attrs: dict[str, Any] = {}
                ep_meta: dict[str, Any] = {}

                if "meta" in ep_group:
                    ep_meta_group = ep_group["meta"]
                    if not isinstance(ep_meta_group, h5py.Group):
                        raise TypeError(f'"{ep_name}/meta" is not an HDF5 group')

                    ep_meta_attrs = _group_attrs_to_dict(ep_meta_group)

                    for dataset_name, obj in ep_meta_group.items():
                        if dataset_name in {"config_full", "config_varying"}:
                            ep_meta[dataset_name] = _read_json_dataset(
                                ep_meta_group, dataset_name
                            )
                        elif isinstance(obj, h5py.Dataset):
                            ep_meta[dataset_name] = np.asarray(obj)

                instance.episode_results.append(
                    EpisodeResult.from_dict(
                        ep_name,
                        episode_data,
                        meta_attrs=ep_meta_attrs,
                        meta=ep_meta,
                        strict=False,
                    )
                )
                instance.episode_results[-1].validate_shapes()
                instance.n_eps_loaded += 1

        return instance

    @property
    def file_info(self) -> dict[str, Any] | None:
        """Return cached validation info, if available."""
        return self._file_info

    @property
    def file_path(self) -> str | None:
        """Return path of the loaded file, if available."""
        return self._file_path

    def keys(self) -> list[str]:
        """Return included dataset keys from file metadata."""
        return list(self.included_keys)

    def episode_names(self) -> list[str]:
        """Return loaded episode names."""
        return [ep.ep_name for ep in self.episode_results]

    def get_episode(self, index: int) -> EpisodeResult:
        """Return one loaded episode by index."""
        return self.episode_results[index]


if __name__ == "__main__":
    # Example usage
    results = SimulationResults.load(
        path="dataset_root/network_data/network_results.h5",
        max_eps=2,
        max_steps=500,
        keys=["rsrp", "sinr", "ue_pos"],
        validate_data=True,
    )

    print(results.config_common["n_ue"])
    print(results.keys())
    print(results.episode_names())

    rsrp_ep0 = results.episode_results[0].rsrp
