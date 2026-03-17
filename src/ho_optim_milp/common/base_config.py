"""Base configuration model using Pydantic for validation and serialization."""

from abc import abstractmethod, ABC
from datetime import datetime
import logging
import shutil
from typing import Any
import yaml

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictBool,
    StrictInt,
    StrictStr,
    Field,
    field_validator,
    model_validator,
)

from . import utils as ut


def propagate(obj: Any, attr_name: str, value_to_propagate: Any):
    """
    Recursively propagate a value to all attributes of an object, list, or dictionary.

    Parameters
    ----------
    obj : Any
        The object, list, or dictionary to propagate the value to.
    attr_name : str
        The name of the attribute to set or check.
    value_to_propagate : Any
        The value to propagate to the specified attribute.
    """
    if hasattr(obj, "__dict__") and hasattr(obj, attr_name):
        current_value = getattr(obj, attr_name)
        if current_value is None:
            setattr(obj, attr_name, value_to_propagate)
            value_to_propagate = getattr(obj, attr_name)
        for v in obj.__dict__.values():
            propagate(v, attr_name, value_to_propagate)
    elif isinstance(obj, list):
        for item in obj:
            propagate(item, attr_name, value_to_propagate)
    elif isinstance(obj, dict):
        for val in obj.values():
            propagate(val, attr_name, value_to_propagate)


def default_simulation_id() -> str:
    """
    Generate a default simulation ID based on the current timestamp.

    Returns
    -------
    str
        A string representing the current date and time.
    """
    return datetime.now().strftime("%Y%m%d%H%M%S%f")


class BaseConfig(BaseModel, ABC):
    """Base class for all configuration models using Pydantic."""

    path_to_config: StrictStr | None = Field(
        default=None,
        description="Path to the configuration file, if loaded from a yaml file.",
        exclude=True,
    )

    base_dir: str = Field(
        default_factory=ut.find_project_root,
        description="Root directory of the project.",
    )
    log_dir: StrictStr = Field(
        default="logs",
        description="Directory to save logs, relative to base_dir.",
    )

    debug: StrictBool | None = Field(default=None, description="Debug mode flag.")
    log_level: StrictInt | None = logging.INFO
    simulation_id: StrictStr = Field(
        default_factory=default_simulation_id,
        description="Unique identifier for the simulation run.",
    )

    model_config = ConfigDict(extra="forbid")  # Forbid unknown keys

    def __str__(self):
        lmax = max(
            (len(l) for l in ut.nested_dict_to_list(self.model_dump(), indent=2))
        )
        terminal_width = shutil.get_terminal_size((80, 20)).columns
        pad = max(
            0,
            min(lmax, terminal_width) - 1 - len(self.__class__.__name__),
        )
        header = f"{self.__class__.__name__} {'=' * pad}\n"
        if hasattr(self, "path_to_config") and self.path_to_config is not None:
            header += f"Path to config: {self.path_to_config}\n"
        return header + ut.nested_dict_to_str(self.model_dump(), indent=2)

    def update(self, updates: dict[str, Any]) -> None:
        """
        Update the configuration with a dictionary of key-value pairs.

        Parameters
        ----------
        updates : dict[str, Any]
            A dictionary containing the keys and values to update in the configuration.

        Raises
        ------
        KeyError
            If an invalid key is provided that does not exist in the configuration.
        """
        for key, value in updates.items():
            if not hasattr(self, key):
                raise KeyError(f"Invalid configuration key: {key}")
            if isinstance(getattr(self, key), BaseConfig) and isinstance(value, dict):
                # If the attribute is a BaseConfig, update it recursively
                current_value = getattr(self, key)
                if isinstance(current_value, BaseConfig):
                    current_value.update(value)
                    continue
            print(
                f"Updating '{self.__class__.__name__}.{key}': "
                f"{repr(getattr(self, key))} -> {repr(value)}"
            )
            setattr(self, key, value)

    def update_recursively(self, updates: dict[str, Any]) -> None:
        """
        Recursively update the configuration with a dictionary of key-value pairs.

        Parameters
        ----------
        updates : dict[str, Any]
            A dictionary containing the keys and values to update in the configuration.
        """
        for key in self.to_dict().keys():
            value = getattr(self, key)
            if isinstance(value, BaseConfig):
                print(f"Recursively updating '{self.__class__.__name__}.{key}'...")
                value.update_recursively(updates)
            else:
                for update_key, update_value in updates.items():
                    if key == update_key:
                        print(
                            f"Update '{self.__class__.__name__}.{key}': "
                            f"{repr(value)} -> {repr(update_value)}"
                        )
                        setattr(self, key, update_value)
                        break

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the configuration.
        """
        return self.model_dump()

    def export_as_yaml(self, path: str) -> None:
        """
        Save the configuration to a YAML file.

        Parameters
        ----------
        path : str
            The file path where the configuration will be saved.
        """
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, allow_unicode=True)
        print(f"Configuration saved to: {path}")

    @classmethod
    @abstractmethod
    def from_yaml(cls, path: str) -> Any:
        """
        Load the configuration from a YAML file.

        Parameters
        ----------
        path : str
            The file path from which the configuration will be loaded.

        Returns
        -------
        BaseConfig
            An instance of the configuration class populated with the data from the YAML file.
        """

    @staticmethod
    def _from_yaml(path: str) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: int | str | None) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            value_upper = value.upper()
            if hasattr(logging, value_upper):
                return getattr(logging, value_upper)
        raise ValueError("log_level must be an int or a valid logging level string.")

    @model_validator(mode="after")
    def _propagate_simulation_id(self) -> "BaseConfig":
        """Propagate simulation_id to all nested BaseConfig subclasses."""
        propagate(self, "simulation_id", self.simulation_id)
        return self

    @model_validator(mode="after")
    def _propagate_log_dir(self) -> "BaseConfig":
        """Propagate log_dir to all nested BaseConfig subclasses."""
        propagate(self, "log_dir", self.log_dir)
        return self

    @model_validator(mode="after")
    def _propagate_debug(self) -> "BaseConfig":
        """Propagate debug and log_level to all nested BaseConfig subclasses."""
        propagate(self, "debug", self.debug)
        return self

    @model_validator(mode="after")
    def _propagate_log_level(self) -> "BaseConfig":
        """Propagate log_level to all nested BaseConfig subclasses."""
        propagate(self, "log_level", self.log_level)
        return self
