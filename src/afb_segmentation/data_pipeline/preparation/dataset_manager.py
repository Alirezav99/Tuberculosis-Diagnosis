"""Initialize or switch between working and snapshotted DDS3 datasets."""

import logging
import os
import shutil

from dask.distributed import Client, LocalCluster
from dotenv import load_dotenv

from afb_segmentation.data_pipeline.augmentation import (
    dataset_augmentation,
    spatial_bias,
)
from afb_segmentation.data_pipeline.preparation.bmp import convert_bmp_to_png
from afb_segmentation.data_pipeline.preparation.restructure import restructure_dataset
from afb_segmentation.data_pipeline.preparation.yolo_export import create_yolo_dataset

load_dotenv()

ROOT_DATA = os.getenv("ROOT_DATA")
ROOT_SNAPSHOT_DATA = os.getenv("ROOT_SNAPSHOT_DATA")

logging.basicConfig(level=logging.INFO)


def init_dds3():
    """Create the augmented snapshot and rebuild unaugmented working datasets.

    If ``ROOT_SNAPSHOT_DATA`` does not exist, this prepares DDS3, augments
    training image-mask pairs with paired tile transforms and tile shuffling,
    applies six circular spatial-shift iterations, and exports the YOLO layout.
    It then copies both prepared layouts into the snapshot directory, removes
    the working copies, and rebuilds the unaugmented layouts. If the snapshot
    directory already exists, the function does nothing.

    This operation reads and mutates files beneath ``ROOT_DATA`` and
    ``ROOT_SNAPSHOT_DATA``.
    """
    ROOT_SNAPSHOT_DATA = os.getenv("ROOT_SNAPSHOT_DATA")
    ROOT_DATA = os.getenv("ROOT_DATA")

    if not os.path.exists(os.path.join(ROOT_SNAPSHOT_DATA)):
        cluster = LocalCluster(n_workers=32, processes=True)

        with Client(cluster) as client:
            convert_bmp_to_png()
            restructure_dataset()
            dataset_augmentation.augment_dds3(
                folder2augment="train",
                shuffle_tiles=True,
            )
            spatial_bias.remove_dds3_spatial_bias(n_iters=6)
            create_yolo_dataset()

            os.makedirs(ROOT_SNAPSHOT_DATA)
            shutil.copytree(
                src=os.path.join(ROOT_DATA, "dds3-restructured"),
                dst=os.path.join(ROOT_SNAPSHOT_DATA, "dds3-restructured"),
            )
            shutil.copytree(
                src=os.path.join(ROOT_DATA, "dds3-yolo"),
                dst=os.path.join(ROOT_SNAPSHOT_DATA, "dds3-yolo"),
            )

            shutil.rmtree(os.path.join(ROOT_DATA, "dds3-restructured"))
            shutil.rmtree(os.path.join(ROOT_DATA, "dds3-yolo"))

            convert_bmp_to_png()
            restructure_dataset()
            create_yolo_dataset()


class DatasetManager:
    """Switch working DDS3 directories between rebuilt and snapshotted data."""

    @staticmethod
    def _safe_rmtree(path):
        """Remove ``path`` recursively when it exists; otherwise log and return.

        Parameters
        ----------
        path : str
            Directory path to remove if present.

        Returns
        -------
        None
        """
        if os.path.exists(path):
            shutil.rmtree(path)
            logging.info(f"Removed directory: {path}")
        else:
            logging.info(f"Directory does not exist, skipping: {path}")

    @staticmethod
    def reset_dataset():
        """
        Delete and rebuild the working restructured and YOLO dataset folders.

        Rebuilding converts any remaining BMP files, restructures source splits,
        and exports the YOLO layout without running augmentation or spatial
        bias removal.
        """
        DatasetManager._safe_rmtree(os.path.join(ROOT_DATA, "dds3-restructured"))
        DatasetManager._safe_rmtree(os.path.join(ROOT_DATA, "dds3-yolo"))
        convert_bmp_to_png()
        restructure_dataset()
        create_yolo_dataset()
        logging.info("Dataset reset and re-initialized.")

    def levelup_dataset():
        """
        Replace the working dataset folders with copies from the snapshot.

        The working ``dds3-restructured`` and ``dds3-yolo`` directories are
        removed before the corresponding directories under
        ``ROOT_SNAPSHOT_DATA`` are copied into ``ROOT_DATA``.
        """
        DatasetManager._safe_rmtree(os.path.join(ROOT_DATA, "dds3-restructured"))
        DatasetManager._safe_rmtree(os.path.join(ROOT_DATA, "dds3-yolo"))
        shutil.copytree(
            src=os.path.join(ROOT_SNAPSHOT_DATA, "dds3-restructured"),
            dst=os.path.join(ROOT_DATA, "dds3-restructured"),
        )
        shutil.copytree(
            src=os.path.join(ROOT_SNAPSHOT_DATA, "dds3-yolo"),
            dst=os.path.join(ROOT_DATA, "dds3-yolo"),
        )
        logging.info("Dataset restored from snapshot.")
