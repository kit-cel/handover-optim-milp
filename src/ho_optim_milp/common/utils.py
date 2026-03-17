"""Common utility functions."""

from pathlib import Path
from typing import Any


def find_project_root(start: Path = Path(__file__).resolve()) -> str:
    """
    Find the root directory of the project by looking for the nearest .git directory.

    Parameters
    ----------
    start : Path, optional
        The starting directory to search from, by default the directory of this file.

    Returns
    -------
    str
        The path to the root directory of the project.
    """
    for parent in start.parents:
        if (parent / ".git").exists():
            return str(parent)
    return str(start.parent)  # fallback


def nested_dict_to_list(
    d: dict[Any, Any], indent: int = 2, float_precision: int = 3, _lvl: int = 0
) -> list[str]:
    """
    Convert a nested dictionary to a list of strings with indentation.

    Parameters
    ----------
    d : dict
        Dictionary to convert
    indent : int, optional
        Indentation level, by default 2
    float_precision : int, optional
        Precision for float values, by default 3
    _lvl : int, optional
        Current level of indentation, by default 0

    Returns
    -------
    list[str]: List of strings representing the nested dictionary
    """
    lines = []
    pad = " " * indent * _lvl

    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{pad}{key}:")
            lines += nested_dict_to_list(value, indent, float_precision, _lvl + 1)
        elif isinstance(value, list):
            lines.append(f"{pad}{key}: [")
            for item in value:
                if isinstance(item, dict):
                    __lvl = _lvl + 2 if len(value) > 1 else _lvl + 1
                    lines += nested_dict_to_list(item, indent, float_precision, __lvl)
                else:
                    lines.append(f"{' ' * (indent + 2)}{item}")
                lines[-1] += ","
            lines.append(f"{pad}]")
        else:
            if isinstance(value, float):
                value = f"{value:.{float_precision}f}"
            lines.append(f"{pad}{key}: {value}")
    return lines


def nested_dict_to_str(
    d: dict[Any, Any], indent: int = 2, float_precision: int = 3, _lvl: int = 0
) -> str:
    """
    Format nested dictionary as a string with indentation.

    Parameters
    ----------
    d : dict
        Dictionary to format
    indent : int, optional
        Indentation level, by default 2
    float_precision : int, optional
        Precision for float values, by default 3

    Returns:
        str: Formatted string
    """
    return "\n".join(nested_dict_to_list(d, indent, float_precision, _lvl))


def format_time(seconds: float) -> str:
    """Return seconds as a formatted string."""
    if seconds < 60:
        return f"{seconds:.3f} s"
    if seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}:{seconds:.3f} min"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}:{int(minutes):02}:{seconds:.3f} h"
