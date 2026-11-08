"""Copy source DDS3 splits into the standardized directory layout."""

import logging
import os

from tqdm import tqdm

from afb_segmentation.data_pipeline.preparation import files as file_management


def restructure_dataset(target_folder_name: str = "dds3-restructured") -> None:
    """Restructure the DDS3 dataset into a standardized format.

    This function reorganizes the original DDS3 dataset structure into a more
    standardized format suitable for machine learning workflows:

    Original structure:
        DDS3/
            TRAINING SET/
                {category}/
                    IMAGE/
                    MASK/
            TEST SET/
                {category}/
                    IMAGE/
                    MASK/
            VALIDATION SET/
                {category}/
                    IMAGE/
                    MASK/

    Target structure:
        dds3-restructured/
            train/
                images/
                masks/
            test/
                images/
                masks/
            validation/
                images/
                masks/

    The function is idempotent - if the target directory already exists,
    it will skip processing to avoid duplicate work.

    Parameters
    ----------
    target_folder_name : str, optional
        Name of the target restructured folder.

    Raises
    ------
    FileNotFoundError
        If the source DDS3 directory does not exist.
    PermissionError
        If file operations do not have sufficient permissions.
    """
    logging.info("Starting DDS3 dataset restructuring process...")

    src = os.path.join(os.getenv("ROOT_DATA"), "DDS3")

    if not os.path.exists(src):
        logging.error(f"Source directory not found: {src}")
        raise FileNotFoundError(f"Source directory not found: {src}")

    set_mappings = {
        "TRAINING SET": "train",
        "TEST SET": "test",
        "VALIDATION SET": "validation",
    }
    subfolder_mappings = {
        "IMAGE": "images",
        "MASK": "masks",
    }

    target_path = os.path.join(os.path.dirname(src), target_folder_name)

    if os.path.exists(target_path):
        logging.info(f"Target directory already exists: {target_path}")
        logging.info("Dataset restructuring skipped - already processed")
        return

    logging.info(f"Creating restructured dataset in: {target_path}")

    items_to_process = []
    for item in os.listdir(src):
        item_path = os.path.join(src, item)
        if os.path.isdir(item_path) and file_management.get_set_name(
            item, set_mappings
        ):
            items_to_process.append((item, item_path))

    if not items_to_process:
        logging.warning("No valid dataset folders found to process")
        return

    logging.info(f"Found {len(items_to_process)} dataset splits to process")

    total_files_processed = 0

    with tqdm(
        total=len(items_to_process), desc="Processing dataset splits", unit="split"
    ) as split_pbar:
        for item, item_path in items_to_process:
            set_name = file_management.get_set_name(item, set_mappings)
            logging.info(f"Processing {set_name} dataset split...")

            set_target = os.path.join(target_path, set_name)
            os.makedirs(set_target, exist_ok=True)

            for category_folder in os.listdir(item_path):
                category_path = os.path.join(item_path, category_folder)
                if not os.path.isdir(category_path):
                    continue

                logging.debug(f"Processing category: {category_folder}")

                for sub_folder in os.listdir(category_path):
                    sub_path = os.path.join(category_path, sub_folder)
                    if not os.path.isdir(sub_path):
                        logging.warning(f"{sub_path} is not a dir!")
                        continue
                    data_type = file_management.get_set_name(
                        sub_folder, subfolder_mappings
                    )
                    if not data_type:
                        logging.warning(f"Skipping unknown subfolder: {sub_folder}")
                        continue

                    target_dir = os.path.join(set_target, data_type)
                    os.makedirs(target_dir, exist_ok=True)

                    last_num = file_management.get_last_index(target_dir)
                    files_processed, _ = file_management.process_files_in_directory(
                        sub_path, target_dir, last_num
                    )

                    total_files_processed += files_processed
                    logging.debug(f"Processed {files_processed} files from {sub_path}")

            split_pbar.update(1)

    logging.info(f"Dataset restructuring completed successfully:")
    logging.info(f"  - Processed {len(items_to_process)} dataset splits")
    logging.info(f"  - Total files processed: {total_files_processed}")
    logging.info(f"  - Output directory: {target_path}")
    logging.info("Dataset restructuring process finished.")
