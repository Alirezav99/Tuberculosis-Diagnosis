"""Test-set evaluation for available trained U-Net checkpoints.

Each configured augmentation scenario is evaluated independently when its
``best.pt`` checkpoint exists, and its metrics are written to the scenario
results directory.
"""

import csv
import os

import torch

from ..training.unet_components import get_dataloaders, get_loss_fn, get_model
from ..training.unet_config import load_config


def evaluate_trained_models_on_test() -> None:
    """Load the best checkpoint for each available augmentation scenario and
    evaluate it with ``eval_one_epoch`` on the test split. Scenarios without a
    ``best.pt`` checkpoint are skipped. Results are written to each scenario's
    ``test_results.csv``.

    Environment
    -----------
        Reads ``ROOT_PROJECT``, ``ROOT_MODELS``, and ``ROOT_DATA``.
    """
    from ..training.segmentation_epochs import eval_one_epoch

    project_root = os.getenv("ROOT_PROJECT")
    models_root = os.getenv("ROOT_MODELS")
    data_root = os.getenv("ROOT_DATA")

    cfg = load_config(project_root)

    scenarios = [
        {"name": "without_augmentation", "augmentation": False},
        {"name": "with_augmentation", "augmentation": True},
    ]

    for scenario in scenarios:
        augmentation = scenario["augmentation"]
        scenario_name = scenario["name"]

        print("\n" + "=" * 60)
        print(f"Evaluating UNet model: {scenario_name}")
        print("=" * 60)

        weights_dir = os.path.join(models_root, "unet", scenario_name, "weights")
        results_dir = os.path.join(models_root, "unet", scenario_name, "results")
        os.makedirs(results_dir, exist_ok=True)

        best_model_path = os.path.join(weights_dir, "best.pt")
        if not os.path.exists(best_model_path):
            print(f"Warning: best.pt not found at {best_model_path}, skipping...")
            continue

        root_test = os.path.join(data_root, "dds3-restructured/test")
        test_loader = get_dataloaders(
            root_test, cfg, shuffle=False, augmentation=augmentation
        )

        model = get_model(cfg)
        print(f"Loading model from: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=cfg.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(cfg.device)

        loss_fn = get_loss_fn(cfg)

        print("Running evaluation on test set...")
        test_metrics = eval_one_epoch(
            model=model,
            loss_fn=loss_fn,
            device=cfg.device,
            eval_loader=test_loader,
        )

        print("\nTest Set Evaluation Results:")
        print("-" * 40)
        for key, value in test_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        results_path = os.path.join(results_dir, "test_results.csv")
        with open(results_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(test_metrics.keys()))
            writer.writeheader()
            writer.writerow(test_metrics)
        print(f"\nTest results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("Evaluation complete for all scenarios!")
    print("=" * 60)


__all__ = ["evaluate_trained_models_on_test"]
