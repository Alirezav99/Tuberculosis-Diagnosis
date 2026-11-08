"""Evaluate frozen DDS3 checkpoints and bootstrap image-level uncertainty.

No training, dataset manager, or data preparation is invoked. The original
dataset loader, image order, batch size, model, and component scorer are used.
Outputs require a fresh directory; saved source data and models are read only.
The legacy AP scores are precision-based component scores, not standard AP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import pathlib
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "src/afb_segmentation/models/config.yml"
SPLIT = ROOT / "artifacts/dataset/dds3-restructured/test"
YOLO = ROOT / "artifacts/models/yolo11l-seg - True/train/weights/best.pt"
UNET = ROOT / "artifacts/models/unet/with_augmentation/weights/best.pt"
REFERENCE = ROOT / "artifacts/models/unet/with_augmentation/results/test_results.csv"
SOURCE_FILES = [
    Path(__file__),
    CONFIG,
    ROOT / "src/afb_segmentation/data_pipeline/load/dataset.py",
    ROOT / "src/afb_segmentation/utils/image_filtering.py",
    ROOT / "src/afb_segmentation/utils/yolo.py",
    ROOT / "src/afb_segmentation/models/architectures/unet.py",
    ROOT / "src/afb_segmentation/models/metrics/instance_segmentation.py",
    ROOT / "src/afb_segmentation/models/training/segmentation_epochs.py",
]


def sha256(path: Path) -> str:
    """Hash an input without holding a whole checkpoint in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write a readable, finite JSON record."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries with a fixed schema."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_prediction(path_string: str) -> dict:
    """Apply the unchanged repository scorer to one cached image."""
    import torch

    from afb_segmentation.models.metrics.instance_segmentation import (
        compute_instance_segmentation_metrics,
    )

    torch.set_num_threads(1)
    path = Path(path_string)
    with np.load(path) as sample:
        pred, target = sample["prediction"], sample["target"]
    metrics = compute_instance_segmentation_metrics(pred, target)
    pred, target = pred.astype(bool), target.astype(bool)
    return {
        "image": path.stem + ".png",
        "tp": metrics.get("tp@50", 0),
        "fp": metrics.get("fp@50", 0),
        "fn": metrics.get("fn@50", 0),
        "legacy_ap50": float(metrics["ap@50"]),
        "legacy_ap50_95": float(metrics["ap@50:95"]),
        "num_gt_instances": metrics["num_gt_instances"],
        "num_pred_instances": metrics["num_pred_instances"],
        "pixel_tp": int(np.count_nonzero(pred & target)),
        "pixel_fp": int(np.count_nonzero(pred & ~target)),
        "pixel_fn": int(np.count_nonzero(~pred & target)),
        "pixel_tn": int(np.count_nonzero(~pred & ~target)),
    }


def aggregate(rows: list[dict], indices=None) -> dict:
    """Preserve pooled count ratios and image-mean legacy AP aggregation."""
    if indices is None:
        indices = np.arange(len(rows))
    data = {
        k: np.asarray([r[k] for r in rows])[indices] for k in rows[0] if k != "image"
    }

    def ratio(num, denom):
        return float(num / denom) if denom else 0.0

    tp, fp, fn = (data[k].sum() for k in ("tp", "fp", "fn"))
    ptp, pfp, pfn, ptn = (
        data[k].sum() for k in ("pixel_tp", "pixel_fp", "pixel_fn", "pixel_tn")
    )
    return {
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "legacy_ap50": float(data["legacy_ap50"].mean()),
        "legacy_ap50_95": float(data["legacy_ap50_95"].mean()),
        "pixel_precision": ratio(ptp, ptp + pfp),
        "pixel_recall": ratio(ptp, ptp + pfn),
        "pixel_dice": ratio(2 * ptp, 2 * ptp + pfp + pfn),
        "pixel_iou": ratio(ptp, ptp + pfp + pfn),
        "pixel_accuracy": ratio(ptp + ptn, ptp + pfp + pfn + ptn),
    }


def bootstrap(rows: list[dict], output: Path, seed: int, repeats: int) -> list[dict]:
    """Resample complete images; recalculate pooled metrics in every sample."""
    point = aggregate(rows)
    rng = np.random.default_rng(seed)
    # Each draw selects the original number of complete image records.
    draws = []
    for _ in range(repeats):
        indices = rng.integers(0, len(rows), size=len(rows))
        draws.append(aggregate(rows, indices))
    write_csv(output / "bootstrap_replicates.csv", draws)
    result = []
    for name, value in point.items():
        values = np.asarray([d[name] for d in draws])
        lower, upper = np.quantile(values, [0.025, 0.975])
        result.append(
            {
                "metric": name,
                "estimate": value,
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "bootstrap_standard_error": float(values.std(ddof=1)),
            }
        )
    write_csv(output / "confidence_intervals.csv", result)
    return result


def main() -> None:
    """Run frozen inference, validate historical scores, then bootstrap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--limit", type=int, help="Smoke test only; no publishable intervals"
    )
    args = parser.parse_args()
    if (
        args.resamples < 2
        or args.workers < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("resamples >= 2, workers >= 1, and limit >= 1 are required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "predictions"
    cache.mkdir()
    started = time.perf_counter()
    # Original trusted checkpoints were saved using Python 3.13 pathlib.
    sys.modules.setdefault("pathlib._local", pathlib)
    import cv2
    import torch
    import ultralytics
    import yaml
    from torch.utils.data import DataLoader
    from ultralytics import YOLO as Detector

    from afb_segmentation.data_pipeline.load.dataset import CustomDatasetTBDD3
    from afb_segmentation.models.architectures.unet import AttentionUNetMaskPredictor

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This protocol requires CUDA, as used for the saved evaluation"
        )
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    cfg = yaml.safe_load(CONFIG.read_text())["unet"]
    detector = Detector(str(YOLO)).to("cuda")
    dataset = CustomDatasetTBDD3(str(SPLIT), detector, cfg["size"])
    if len(dataset) != 1000:
        raise RuntimeError(
            f"Expected the full 1000-image test split, found {len(dataset)}"
        )
    if args.limit is not None:
        dataset.image_paths = dataset.image_paths[: args.limit]
    paths = dataset.image_paths
    image_records = [
        {
            "image": str(p.relative_to(ROOT)),
            "image_sha256": sha256(p),
            "mask": str((SPLIT / "masks" / p.name).relative_to(ROOT)),
            "mask_sha256": sha256(SPLIT / "masks" / p.name),
        }
        for p in paths
    ]
    protected = {
        str(p.relative_to(ROOT)): sha256(p)
        for p in [YOLO, UNET, REFERENCE, *SOURCE_FILES]
    }
    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "settings": {
            "seed": args.seed,
            "bootstrap_resamples": args.resamples,
            "bootstrap_method": "two-sided image percentile bootstrap",
            "yolo_confidence": 0.25,
            "unet_threshold": 0.5,
            "unet_batch_size": cfg["batch_size"],
            "unet_size": cfg["size"],
            "yolo_imgsz": 416,
            "fp16": False,
            "dropout_active": False,
            "training": False,
            "dataset_manager_used": False,
            "workers": args.workers,
            "smoke_test": args.limit is not None,
            "historical_comparison_absolute_tolerance": 1e-5,
        },
        "config": cfg,
        "protected_sha256": protected,
        "images": image_records,
        "limitations": [
            "Fixed trained model; no assessment of independent training-run variability.",
            "Mosaic-image resampling; unknown slide/patient clustering is not accounted for.",
            "Legacy AP scores are image-mean precision-based component scores, not standard confidence-ranked AP.",
            "Single-pass evaluation matches the saved score evaluator; MC Dropout is disabled.",
            "Code uses YOLO confidence 0.25, whereas the Word manuscript states 0.5.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    model = AttentionUNetMaskPredictor(
        in_channels=3,
        out_channels=1,
        dropout=cfg["dropout"],
        base_channels=64,
        depth=cfg["depth"],
        channel_factor=2,
        scale_factor=cfg["scale_factor"],
    )
    checkpoint = torch.load(UNET, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    manifest["unet_checkpoint_epoch"] = checkpoint.get("epoch")
    del checkpoint
    model = model.cuda().float().eval()
    loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
    )
    n = 0
    with torch.no_grad():
        for inputs, labels in loader:
            predictions = (
                (model(inputs.cuda().float()) > 0.5)
                .cpu()
                .numpy()[:, 0]
                .astype(np.uint8)
            )
            targets = labels.numpy().astype(np.uint8)
            for pred, target in zip(predictions, targets):
                np.savez_compressed(
                    cache / (paths[n].stem + ".npz"), prediction=pred, target=target
                )
                n += 1
            if n % 50 == 0 or n == len(paths):
                print(
                    f"Frozen inference: {n}/{len(paths)} images, {time.perf_counter()-started:.1f}s",
                    flush=True,
                )
    cache_paths = [str(cache / (p.stem + ".npz")) for p in paths]
    rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for row in pool.map(score_prediction, cache_paths, chunksize=5):
            rows.append(row)
            if len(rows) % 100 == 0 or len(rows) == len(paths):
                print(
                    f"Original component scoring: {len(rows)}/{len(paths)}, {time.perf_counter()-started:.1f}s",
                    flush=True,
                )
    write_csv(output / "per_image_metrics.csv", rows)
    point = aggregate(rows)
    write_json(output / "point_estimates.json", point)
    with REFERENCE.open(newline="") as stream:
        historical = next(csv.DictReader(stream))
    mapping = {
        "precision": "precision(M)",
        "recall": "recall(M)",
        "legacy_ap50": "mAP@50(M)",
        "legacy_ap50_95": "mAP@50:95(M)",
        "pixel_precision": "precision(pixel)",
        "pixel_recall": "recall(pixel)",
        "pixel_dice": "f1(pixel)",
        "pixel_iou": "iou(pixel)",
        "pixel_accuracy": "accuracy",
    }
    comparisons = [
        {
            "metric": name,
            "saved": float(historical[key]),
            "current": point[name],
            "difference": point[name] - float(historical[key]),
            "matches_within_1e-5": abs(point[name] - float(historical[key])) <= 1e-5,
        }
        for name, key in mapping.items()
    ]
    write_csv(output / "reproduction_check.csv", comparisons)
    changed = [s for s, h in protected.items() if sha256(ROOT / s) != h]
    for record in image_records:
        for kind in ("image", "mask"):
            if sha256(ROOT / record[kind]) != record[kind + "_sha256"]:
                changed.append(record[kind])
    manifest["input_integrity"] = {"unchanged": not changed, "changed": changed}
    if changed:
        raise RuntimeError(f"Inputs changed during evaluation: {changed}")
    matches = all(c["matches_within_1e-5"] for c in comparisons)
    manifest["historical_scores_reproduced"] = matches and args.limit is None
    manifest["prediction_sha256"] = {Path(p).name: sha256(Path(p)) for p in cache_paths}
    intervals = None
    if matches and args.limit is None:
        intervals = bootstrap(rows, output, args.seed, args.resamples)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["status"] = (
        "complete"
        if intervals
        else ("smoke_test" if args.limit else "reproduction_mismatch")
    )
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "point_estimates": point,
                "confidence_intervals": intervals,
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )
    if not matches and args.limit is None:
        raise SystemExit(
            "Historical scores did not reproduce; intervals intentionally withheld pending investigation."
        )


if __name__ == "__main__":
    main()
