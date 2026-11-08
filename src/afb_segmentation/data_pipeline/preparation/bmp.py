"""Convert source DDS3 BMP files to PNG."""

import logging
import os

import dask
from dask.diagnostics import ProgressBar

from afb_segmentation.data_pipeline.conversion.bitmaps import convert_single_bmp


def convert_bmp_to_png() -> None:
    """Convert BMP files to PNG format using parallel processing.

    This function performs the following operations:
    1. Scans the DDS3 directory for all BMP files
    2. Checks if files are already processed (PNG exists)
    3. Converts BMP files to PNG format in parallel using Dask
    4. Removes original BMP files after successful conversion
    5. Provides comprehensive logging and progress tracking

    The function is idempotent - running it multiple times will skip already
    processed files and only convert new BMP files.

    Raises
    ------
    FileNotFoundError
        If the DDS3 source directory does not exist.
    Exception
        If a processing error occurs during conversion.
    """
    logging.info("Starting BMP to PNG conversion process...")

    src_dds3 = os.path.join(os.getenv("ROOT_DATA"), "DDS3")

    if not os.path.exists(src_dds3):
        logging.error(f"Source directory not found: {src_dds3}")
        raise FileNotFoundError(f"Source directory not found: {src_dds3}")

    logging.info(f"Scanning for BMP files in: {src_dds3}")

    bmp_files = []
    already_processed = []

    for dirpath, _, filenames in os.walk(src_dds3):
        for filename in filenames:
            if filename.endswith(".bmp"):
                bmp_path = os.path.join(dirpath, filename)
                png_path = os.path.splitext(bmp_path)[0] + ".png"

                if os.path.exists(png_path):
                    already_processed.append(bmp_path)
                else:
                    bmp_files.append(bmp_path)

    total_bmp = len(bmp_files) + len(already_processed)
    logging.info(f"Found {total_bmp} BMP files total:")
    logging.info(f"  - {len(already_processed)} already processed (PNG exists)")
    logging.info(f"  - {len(bmp_files)} need to be converted")

    if len(already_processed) > 0:
        logging.info(
            "Files already processed - skipping conversion for existing PNG files"
        )

    if len(bmp_files) == 0:
        logging.info("No BMP files need conversion. All files are already processed.")
        return

    logging.info(f"Starting parallel conversion of {len(bmp_files)} BMP files...")

    tasks = []
    for path in bmp_files:
        task = dask.delayed(convert_single_bmp)(path)
        tasks.append(task)

    with ProgressBar():
        dask.compute(*tasks, scheduler="processes")

    logging.info(f"Conversion completed:")
    logging.info("BMP to PNG conversion process finished.")
