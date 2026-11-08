"""Run Optuna searches for configured YOLO segmentation models.

Run with ``python -m experiments.tuning.yolo`` from the repository root. The
workflow reads model and dataset settings from the project configuration,
runs one 20-trial study per configured model, and stores trial metrics as CSV
files under ``experiments``. Each trial trains for two epochs and
selects its objective from validation mask mAP50 (or box mAP50 when mask mAP50
is unavailable). Dataset reset and training outputs are side effects of
running the search.
"""

import os

import optuna
from dotenv import load_dotenv
from ultralytics import YOLO

from afb_segmentation.data_pipeline.preparation.dataset_manager import DatasetManager
from afb_segmentation.utils.config import read_config

load_dotenv()

if __name__ == "__main__":
    ROOT_DATA = os.getenv("ROOT_DATA")
    ROOT_PROJECT = os.getenv("ROOT_PROJECT")
    ROOT_MODELS = os.getenv("ROOT_MODELS")
    rest_flag = False
    path_config_train = os.path.join(
        ROOT_PROJECT, "src/afb_segmentation/models/config.yml"
    )
    config_train = read_config(path_config_train)
    results_path = os.path.join(ROOT_MODELS, "optuna_results")
    os.makedirs(results_path, exist_ok=True)
    experiment_results_path = os.path.join(ROOT_PROJECT, "experiments")
    os.makedirs(experiment_results_path, exist_ok=True)

    import csv

    for model_name in config_train["models"]:
        print(f"Starting hyperparameter optimization for model: {model_name}")
        print("-----------------------------------------------")
        trial_results = []

        def objective(trial: optuna.trial.Trial) -> float:
            """Train one sampled configuration and return its validation mAP50.

            The trial samples learning rate, batch size, weight decay,
            optimizer, dropout, and final learning-rate ratio. The dataset is
            reset before the first trial in the script, then the trial trains
            for two epochs and validates the model.
            """
            learning_rate = trial.suggest_loguniform("lr0", 1e-5, 1e-2)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 42])
            weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
            optimizer = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            lrf = trial.suggest_loguniform("lrf", 1e-5, 1e-1)

            path_config_data = os.path.join(ROOT_DATA, "dds3-yolo/data.yaml")
            global rest_flag
            if not rest_flag:
                DatasetManager.reset_dataset()
                rest_flag = True
            weights_path = f"{model_name}.pt"
            if os.path.exists(weights_path):
                os.remove(weights_path)
            model = YOLO(model_name)

            model.train(
                data=path_config_data,
                epochs=2,
                imgsz=config_train["imgsz"],
                device=config_train["device"],
                batch=batch_size,
                seed=config_train["seed"],
                single_cls=True,
                optimizer=optimizer,
                deterministic=True,
                project=os.path.join(ROOT_PROJECT, "artifacts/runs/segment"),
                lr0=learning_rate,
                lrf=lrf,
                close_mosaic=False,
                dropout=dropout,
                verbose=False,
                resume=False,
                weight_decay=weight_decay,
            )

            results = model.val(
                data=path_config_data,
                imgsz=config_train["imgsz"],
                device=config_train["device"],
                batch=batch_size,
                single_cls=True,
                name="val",
                project=results_path,
                verbose=False,
            )

            metrics_dict = {}
            mAP50 = None

            if hasattr(results, "results_dict"):
                metrics_dict = results.results_dict.copy()
                mAP50 = metrics_dict.get("metrics/mAP50(M)", None)
                if mAP50 is None:
                    mAP50 = metrics_dict.get("metrics/mAP50(B)", None)

            trial_results.append(
                {
                    "lr0": learning_rate,
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "optimizer": optimizer,
                    "dropout": dropout,
                    "lrf": lrf,
                    **metrics_dict,
                }
            )

            if mAP50 is None:
                raise ValueError("mAP50 metric not found in results")
            return mAP50

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)

        if trial_results:
            csv_file = os.path.join(experiment_results_path, f"{model_name}_trials.csv")

            all_keys = set(
                ["lr0", "batch_size", "weight_decay", "optimizer", "dropout", "lrf"]
            )
            for row in trial_results:
                all_keys.update(row.keys())
            all_keys = list(all_keys)
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                for row in trial_results:
                    writer.writerow(row)
            print(f"Trial results saved to {csv_file}")

    print(
        "Hyperparameter optimization for all models completed. All trial results saved as CSV files."
    )
