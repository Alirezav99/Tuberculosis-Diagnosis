"""Training and test evaluation for configured YOLO segmentation models.

Each configured dataset scenario is prepared, trained, evaluated, and written
under the configured model-results directory.
"""

import os

import pandas as pd
from dotenv import load_dotenv
from ultralytics import YOLO

from afb_segmentation.data_pipeline.preparation.dataset_manager import DatasetManager
from afb_segmentation.utils.config import read_config

load_dotenv()


def train_from_scratch() -> None:
    """Train and test one fresh YOLO model for each configured dataset setting.

    The configuration controls model weights, augmentation scenarios, training
    parameters, and validation parameters. Dataset reset/restore operations are
    performed before each run. Each run writes Ultralytics outputs and a
    ``test/test_results.csv`` file beneath its model results directory.

    Raises
    ------
    EnvironmentError
        If ``ROOT_DATA``, ``ROOT_PROJECT``, or ``ROOT_MODELS``
        is not set.
    """
    ROOT_DATA = os.getenv("ROOT_DATA")
    ROOT_PROJECT = os.getenv("ROOT_PROJECT")
    ROOT_MODELS = os.getenv("ROOT_MODELS")

    if not all([ROOT_DATA, ROOT_PROJECT, ROOT_MODELS]):
        raise EnvironmentError(
            "One or more required environment variables "
            "(ROOT_DATA, ROOT_PROJECT, ROOT_MODELS) are not set."
        )

    config_train_path = os.path.join(
        ROOT_PROJECT, "src/afb_segmentation/models/config.yml"
    )
    config_data_path = os.path.join(ROOT_DATA, "dds3-yolo/data.yaml")

    config_train = read_config(config_train_path)

    for use_augmentation in config_train["yolo"]["augmentation"]:
        if use_augmentation:
            DatasetManager.levelup_dataset()
        else:
            DatasetManager.reset_dataset()

        model_name = config_train["yolo"]["models"]
        model = YOLO(model_name)
        results_dir = os.path.join(ROOT_MODELS, f"{model_name} - {use_augmentation}")
        os.makedirs(results_dir, exist_ok=True)

        model.train(
            data=config_data_path,
            epochs=config_train["yolo"]["epochs"],
            imgsz=config_train["yolo"]["imgsz"],
            device=config_train["yolo"]["device"],
            batch=config_train["yolo"]["batch_size"],
            project=results_dir,
            seed=config_train["yolo"]["seed"],
            single_cls=True,
            cache="disk",
            optimizer=config_train["yolo"]["optimizer"],
            deterministic=True,
            patience=config_train["yolo"]["patience"],
            lr0=config_train["yolo"]["lr0"],
            lrf=config_train["yolo"]["lrf"],
            warmup_epochs=config_train["yolo"]["warmup_epochs"],
            close_mosaic=False,
            dropout=config_train["yolo"]["dropout"],
            verbose=True,
        )

        results = model.val(
            data=config_data_path,
            imgsz=config_train["yolo"]["imgsz"],
            device=config_train["yolo"]["device"],
            batch=config_train["yolo"]["batch_size"],
            single_cls=True,
            split="test",
            project=results_dir,
            verbose=False,
        )

        test_csv_path = os.path.join(results_dir, "test")
        os.makedirs(test_csv_path, exist_ok=True)
        test_csv_path = os.path.join(test_csv_path, "test_results.csv")

        pd.DataFrame(
            list(results.results_dict.values()),
            index=list(results.results_dict.keys()),
        ).T.to_csv(test_csv_path, index=False)

        print(
            "================ Finished training with "
            f"augmentation={use_augmentation} ================\n\n"
        )


if __name__ == "__main__":
    train_from_scratch()
