"""Load shared YAML configuration files."""

import yaml


def read_config(path: str) -> dict:
    """Load a YAML configuration mapping from ``path``.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration mapping.

    Raises
    ------
    FileExistsError
        If opening or parsing fails. The original exception is replaced with
        this project-level error.
    """
    try:
        with open(path, "r") as file:
            config = yaml.safe_load(file)
        return config
    except:
        raise FileExistsError("Invalid configuration file in address: " + path)
