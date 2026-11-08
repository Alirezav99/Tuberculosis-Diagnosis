"""File and directory helpers for dataset preparation."""

import logging
import os
import re
import shutil
from typing import Dict, Optional, Tuple


def get_set_name(folder_name: str, mappings: Dict[str, str]) -> Optional[str]:
    """Get the mapped set name from folder name.

    Parameters
    ----------
    folder_name : str
        Original folder name to map.
    mappings : dict[str, str]
        Mapping from original folder-name fragments to standardized names.

    Returns
    -------
    str or None
        Mapped name when a key occurs in ``folder_name``; otherwise ``None``.
    """
    for old_set, new_set in mappings.items():
        if old_set in folder_name:
            return new_set
    return None


def get_last_index(target_dir: str) -> int:
    """Get the highest index number from existing MSC files.

    Parameters
    ----------
    target_dir : str
        Directory to scan for existing files.

    Returns
    -------
    int
        Highest matching index, or 0 when the directory is absent or has no
        matching files.
    """
    if not os.path.exists(target_dir):
        return 0

    existing_files = [f for f in os.listdir(target_dir) if re.match(r"MSC\d+\.png", f)]
    if not existing_files:
        return 0
    return max(int(f[3:-4]) for f in existing_files)


def get_unique_filename(
    target_dir: str, file_name: str, last_num: int
) -> Tuple[str, int]:
    """Generate a unique filename if the target already exists.

    Parameters
    ----------
    target_dir : str
        Target directory path.
    file_name : str
        Original filename.
    last_num : int
        Current highest index number.

    Returns
    -------
    tuple[str, int]
        Destination file path and updated highest index.
    """
    dst_file = os.path.join(target_dir, file_name)
    if os.path.exists(dst_file):
        last_num += 1
        new_file_name = f"MSC{last_num}.png"
        return os.path.join(target_dir, new_file_name), last_num
    return dst_file, last_num


def process_files_in_directory(
    src_path: str, target_dir: str, last_num: int
) -> Tuple[int, int]:
    """Process all valid files in a source directory.

    Parameters
    ----------
    src_path : str
        Source directory path.
    target_dir : str
        Target directory path.
    last_num : int
        Starting index number.

    Returns
    -------
    tuple[int, int]
        Number of copied files and updated highest index.
    """
    files_processed = 0

    for file_name in os.listdir(src_path):
        if not re.match(r"MSC\d+\.png", file_name):
            continue

        src_file = os.path.join(src_path, file_name)
        dst_file, last_num = get_unique_filename(target_dir, file_name, last_num)

        try:
            shutil.copy(src_file, dst_file)
            files_processed += 1
            logging.debug(f"Copied {src_file} -> {dst_file}")
        except Exception as e:
            logging.error(f"Failed to copy {src_file}: {str(e)}")

    return files_processed, last_num


def process_split_files(
    split_data: Tuple[str, str, str, str, str],
) -> Tuple[str, int, int]:
    """Process files for a single split (train/validation/test).

    Parameters
    ----------
    split_data : tuple[str, str, str, str, str]
        Tuple containing the split name, source image and mask directories,
        and destination image and label directories.

    Returns
    -------
    tuple[str, int, int]
        Split name, number of copied images, and number of converted labels.
    """
    from afb_segmentation.data_pipeline.conversion.yolo_labels import (
        convert_mask_to_polygons,
    )

    split_name, old_images, old_labels, new_images, new_labels = split_data
    images_processed = 0
    labels_processed = 0

    if os.path.exists(old_images):
        for f in os.listdir(old_images):
            src = os.path.join(old_images, f)
            dst = os.path.join(new_images, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                try:
                    shutil.copy(src, dst)
                    images_processed += 1
                except Exception as e:
                    logging.error(f"Failed to copy image {src}: {str(e)}")

    if os.path.exists(old_labels):
        for f in os.listdir(old_labels):
            src = os.path.join(old_labels, f)
            dst = os.path.join(new_labels, f).replace(".png", ".txt")

            if os.path.isfile(src) and not os.path.exists(dst):
                try:
                    text = convert_mask_to_polygons(src)
                    with open(dst, "w") as file:
                        file.write(text)
                    labels_processed += 1
                except Exception as e:
                    logging.error(f"Failed to process mask {src}: {str(e)}")

    return split_name, images_processed, labels_processed
