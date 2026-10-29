"""Module for storing network simulation results."""

from copy import deepcopy
from dataclasses import dataclass, field, fields
from typing import Any
import warnings

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class EpisodeResult:
    """Store episode arrays and metadata.

    Positions have shape (steps, UEs, 2); radio arrays (steps, UEs, cells).
    """

    ep_name: str
    ue_pos: NDArray[np.float32]
    distances_2d: NDArray[np.float32]
    distances_3d: NDArray[np.float32]
    rsrp: NDArray[np.float32]
    sinr: NDArray[np.float32]
    ue_move_angle: NDArray[np.float32]

    meta_attrs: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    config_common: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, NDArray[np.float32] | str]:
        """Return the core fields as a dictionary, excluding metadata and config."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in {"meta_attrs", "meta", "config_common", "config"}
        }

    def extract_ue_speed_kph(self) -> float | None:
        """Extract UE speed in km/h from metadata if available."""
        ue_config = self.config.get("ue", None)
        if ue_config is None:
            warnings.warn("UE configuration not found in episode config.")
            return None

        ue_speed = None
        if isinstance(ue_config, list):
            if len(ue_config) > 1:
                raise ValueError("UE config list has more than one entry.")
            if isinstance(ue_config[0], dict) and "speed_kmh" in ue_config[0]:
                ue_speed = ue_config[0]["speed_kmh"]
            else:
                warnings.warn("UE config list does not contain 'speed_kmh' key.")

        return ue_speed

    @classmethod
    def from_dict(
        cls,
        ep_name: str,
        data: dict[str, NDArray | np.ma.MaskedArray],
        *,
        meta_attrs: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        config_common: dict[str, Any] | None = None,
        strict: bool = True,
        warn: bool = True,
    ) -> "EpisodeResult":
        """Build an episode; strict mode rejects missing array fields."""
        update_dict: dict[str, Any] = {}

        meta_attrs = meta_attrs or {}
        meta = meta or {}
        config_common = config_common or {}

        for f in fields(cls):
            if f.name in {
                "ep_name",
                "meta_attrs",
                "meta",
                "config_common",
                "config",
            }:
                continue

            if f.name in data:
                update_dict[f.name] = np.asarray(data[f.name], dtype=np.float32)
            else:
                if strict:
                    raise ValueError(f"Missing field '{f.name}' in data.")
                if warn:
                    warnings.warn(
                        f"Missing field '{f.name}' in data. Continue with empty array."
                    )
                update_dict[f.name] = np.empty(0, dtype=np.float32)

        config = cls._build_effective_config(
            config_common=config_common,
            meta=meta,
        )

        return cls(
            ep_name=ep_name,
            **update_dict,
            meta_attrs=meta_attrs,
            meta=meta,
            config_common=deepcopy(config_common),
            config=config,
        )

    @classmethod
    def keys(cls) -> list[str]:
        """Return the dataset field names excluding metadata and config fields."""
        return [
            f.name
            for f in fields(cls)
            if f.name
            not in {"ep_name", "meta_attrs", "meta", "config_common", "config"}
        ]

    @staticmethod
    def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        """Merge nested dictionaries without changing either input."""
        result = deepcopy(base)
        for key, value in update.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = EpisodeResult._deep_update(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    @classmethod
    def _build_effective_config(
        cls,
        config_common: dict[str, Any],
        meta: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge common, varying, then full episode configuration."""
        config = deepcopy(config_common)

        config_varying = meta.get("config_varying")
        if isinstance(config_varying, dict):
            config = cls._deep_update(config, config_varying)

        config_full = meta.get("config_full")
        if isinstance(config_full, dict):
            config = cls._deep_update(config, config_full)

        return config

    def validate_shapes(self) -> None:
        """Check that arrays share time and UE dimensions."""
        shapes: dict[str, tuple[int, ...]] = {}
        for key, value in self.to_dict().items():
            if key == "ep_name" or not isinstance(value, np.ndarray):
                continue

            shape = value.shape
            if shape == (0,):
                warnings.warn(f"Shape of '{key}' is {shape}.")
                continue

            shapes[key] = shape

        time_dims = {k: s[0] for k, s in shapes.items() if len(s) >= 1}
        if len(set(time_dims.values())) != 1:
            raise ValueError(f"Inconsistent time dimensions: {time_dims}")

        ue_dims = {k: s[1] for k, s in shapes.items() if len(s) >= 2}
        if ue_dims and len(set(ue_dims.values())) != 1:
            raise ValueError(f"Inconsistent UE dimensions: {ue_dims}")
