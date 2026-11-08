"""Build the YOLO image, label, and configuration layout."""

import logging
import os

import dask
import yaml
from dask.diagnostics import ProgressBar

from afb_segmentation.data_pipeline.preparation import files as file_management


def create_yolo_dataset(
    target_folder_name: str = "dds3-yolo",
    src_folder_name: str = "dds3-restructured",
) -> None:
    """Create a YOLO-compatible dataset from restructured dds3 data.

    This function converts the restructured dds3 dataset into YOLO format:

    Source structure:
        dds3-restructured/
            train/images/, train/masks/
            test/images/, test/masks/
            validation/images/, validation/masks/

    Target structure:
        dds3-yolo/
            images/
                train/
                validation/
                test/
            labels/
                train/
                validation/
                test/
            data.yaml

    Key operations:
    1. Creates YOLO directory structure
    2. Copies image files to appropriate directories
    3. Converts mask images to YOLO polygon format (.txt files)
    4. Generates data.yaml configuration file
    5. Uses parallel processing for improved performance

    The function is idempotent - running it multiple times will skip already
    processed files and only process new/changed files.

    Parameters
    ----------
    target_folder_name : str, optional
        Name of the target YOLO dataset folder.
    src_folder_name : str, optional
        Name of the source restructured dataset folder.

    Raises
    ------
    FileNotFoundError
        If the source restructured directory does not exist.
    PermissionError
        If file operations do not have sufficient permissions.
    """
    logging.info("Starting YOLO dataset creation process...")

    src_data = os.getenv("ROOT_DATA")
    src_path = os.path.join(src_data, src_folder_name)
    dataset_root = os.path.join(src_data, target_folder_name)

    if not os.path.exists(src_path):
        logging.error(f"Source directory not found: {src_path}")
        raise FileNotFoundError(f"Source directory not found: {src_path}")

    images_root = os.path.join(dataset_root, "images")
    labels_root = os.path.join(dataset_root, "labels")
    splits = ["train", "validation", "test"]

    yaml_path = os.path.join(dataset_root, "data.yaml")
    if os.path.exists(yaml_path):
        logging.info(f"Target directory already exists: {dataset_root}")

        all_complete = True
        for split in splits:
            images_dir = os.path.join(images_root, split)
            labels_dir = os.path.join(labels_root, split)

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                all_complete = False
                break

            if not any(os.listdir(images_dir)) or not any(os.listdir(labels_dir)):
                all_complete = False
                break

        if all_complete:
            logging.info(
                "YOLO dataset creation skipped - already processed and complete"
            )
            return
        else:
            logging.info(
                "Existing dataset appears incomplete, continuing with processing..."
            )

    logging.info(f"Creating YOLO dataset in: {dataset_root}")

    logging.info("Creating YOLO directory structure...")
    for split in splits:
        os.makedirs(os.path.join(images_root, split), exist_ok=True)
        os.makedirs(os.path.join(labels_root, split), exist_ok=True)

    split_data_list = []
    for split in splits:
        old_images = os.path.join(src_path, split, "images")
        old_labels = os.path.join(src_path, split, "masks")
        new_images = os.path.join(images_root, split)
        new_labels = os.path.join(labels_root, split)

        split_data_list.append((split, old_images, old_labels, new_images, new_labels))

    logging.info(f"Processing {len(splits)} dataset splits...")

    total_images = 0
    total_labels = 0

    tasks = []
    for split, old_images, old_labels, new_images, new_labels in split_data_list:
        task = dask.delayed(file_management.process_split_files)(
            (split, old_images, old_labels, new_images, new_labels)
        )
        tasks.append(task)

    with ProgressBar():
        dask.compute(*tasks, scheduler="processes")

    logging.info("Creating YOLO configuration file...")
    data = {
        "path": os.path.join(src_data, target_folder_name),
        "train": "images/train",
        "test": "images/test",
        "val": "images/validation",
        "names": {0: "basil"},
    }

    try:
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logging.info(f"YOLO configuration saved to: {yaml_path}")
    except Exception as e:
        logging.error(f"Failed to create YOLO configuration file: {str(e)}")
        raise

    logging.info(f"YOLO dataset creation completed successfully:")
    logging.info(f"  - Processed {len(splits)} dataset splits")
    logging.info(f"  - Total images processed: {total_images}")
    logging.info(f"  - Total labels processed: {total_labels}")
    logging.info(f"  - Output directory: {dataset_root}")
    logging.info("YOLO dataset creation process finished.")
