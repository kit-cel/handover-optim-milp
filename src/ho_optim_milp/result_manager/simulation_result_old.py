"""Module to load and handle simulation results from an HDF5 file."""

import os
from typing import Any
import warnings
import json

import h5py
import numpy as np

from .episode_result import EpisodeResult


class SimulationResults:
    """Class to load and handle simulation results from an HDF5 file."""

    EXPECTED_RESULT_KEYS = set(EpisodeResult.keys())

    def __init__(self, env) -> None:
        self.env = env

        self.n_eps_loaded = 0
        self.ep_network_results: list[EpisodeResult] = []

        self.meta_attrs: dict[str, Any] = {}
        self.config: dict[str, Any] = {}

        self._file_path: str | None = None
        self._file_info: dict | None = None

    @classmethod
    def _validate_file(cls, file_path: str) -> dict:
        assert os.path.isfile(file_path), f"HDF5 file not found: {file_path}"

        try:
            with h5py.File(file_path, "r") as f:
                episodes = list(f.keys())

                if not episodes:
                    raise ValueError(f"No valid episode groups found in {file_path}")

                first_ep = episodes[0]
                first_group = f[first_ep]
                if isinstance(first_group, h5py.Group):
                    available_keys = set(first_group.keys())
                else:
                    raise ValueError(f"Episode '{first_ep}' is not a valid group")

                file_info = {
                    "n_episodes": len(episodes),
                    "episodes": episodes,
                    "available_keys": available_keys,
                    "missing_keys": cls.EXPECTED_RESULT_KEYS - available_keys,
                    "extra_keys": available_keys - cls.EXPECTED_RESULT_KEYS,
                    "file_size_mb": os.stat(file_path).st_size / (1024 * 1024),
                }

                if "meta" in f:
                    meta_group = f["meta"]
                    if not isinstance(meta_group, h5py.Group):
                        raise TypeError('"meta" is not an HDF5 group')
                    file_info["has_meta"] = True
                    file_info["meta_attrs_keys"] = list(f["meta"].attrs.keys())
                    file_info["meta_datasets"] = list(meta_group.keys())
                else:
                    file_info["has_meta"] = False

                if file_info["missing_keys"]:
                    warnings.warn(f"Missing expected keys: {file_info['missing_keys']}")
                if file_info["extra_keys"]:
                    warnings.warn(f"Additional keys found: {file_info['extra_keys']}")

                return file_info

        except Exception as e:
            raise ValueError(f"Failed to validate HDF5 file {file_path}: {e}") from e

    @classmethod
    def load(
        cls,
        path: str,
        max_eps: int = 1,
        max_samples: int = 1_000,
        keys: list[str] | None = None,
        validate_data: bool = False,
    ) -> "SimulationResults":
        """Load results from an HDF5 file."""
        instance = cls(env=None)

        if validate_data:
            instance._file_info = cls._validate_file(path)
            instance._file_path = path

        with h5py.File(path, "r") as f:
            meta_config = f["meta/config"]
            if not isinstance(meta_config, h5py.Dataset):
                raise TypeError('"meta/config" is not an HDF5 dataset')

            raw = meta_config[()]

            if isinstance(raw, bytes):
                config_dict = json.loads(raw.decode("utf-8"))
            elif isinstance(raw, str):
                config_dict = json.loads(raw)
            else:
                raise TypeError(
                    f'"meta/config" must contain str or bytes, got {type(raw).__name__}'
                )
            instance.config = config_dict

            episodes = list(f.keys())[:max_eps]
            for ep in episodes:
                maybe_grp = f[ep]
                if not isinstance(maybe_grp, h5py.Group):
                    continue

                if keys is not None:
                    load_keys = set(keys)
                    if not load_keys.issubset(cls.EXPECTED_RESULT_KEYS):
                        raise ValueError(f"Invalid keys: {keys}")
                else:
                    load_keys = cls.EXPECTED_RESULT_KEYS

                grp_keys = set(maybe_grp.keys())
                if not load_keys.issubset(grp_keys):
                    raise ValueError(f"Expected: {load_keys}, found: {grp_keys}")

                d = {k: np.asarray(maybe_grp[k])[:max_samples] for k in load_keys}
                instance.ep_network_results.append(EpisodeResult.from_dict(d))
                instance.n_eps_loaded += 1

        return instance
