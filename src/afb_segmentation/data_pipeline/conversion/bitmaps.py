"""Convert individual bitmap files to PNG."""

import os
from typing import Tuple

from PIL import Image


def convert_single_bmp(bmp_path: str) -> Tuple[bool, str]:
    """Convert a single BMP file to PNG format.

    Parameters
    ----------
    bmp_path : str
        Path to the BMP file to convert.

    Returns
    -------
    Tuple[bool, str]
        A success flag and a status message.
    """
    try:
        png_path = os.path.splitext(bmp_path)[0] + ".png"

        if os.path.exists(png_path):
            return True, f"PNG already exists for {bmp_path}"

        with Image.open(bmp_path) as img:
            img.save(png_path, "PNG")

        os.remove(bmp_path)
        return True, f"Successfully converted {bmp_path}"

    except Exception as e:
        return False, f"Error converting {bmp_path}: {str(e)}"
