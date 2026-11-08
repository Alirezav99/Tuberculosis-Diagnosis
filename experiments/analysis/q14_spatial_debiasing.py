"""Compare conventional patch augmentation with the manuscript's spatial debiasing.

This experiment creates isolated copies of the supplied DDS3 partitions. Both
conditions use the same patch transformations and add one transformed copy per
training image. The spatial-debiasing condition additionally shuffles the
40-by-40 tiles and applies six circular shifts to every training image and
mask. Validation and test data remain unchanged.

Run from the repository root with ``python -m experiments.analysis.q14_spatial_debiasing``.
The default output is under ``experiments/``; the configured source
dataset is read only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import platform
import random
import shutil
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import ultralytics
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

from afb_segmentation.data_pipeline.augmentation.dataset_augmentation import (  # noqa: E402
    augment_image,
)
from afb_segmentation.data_pipeline.augmentation.spatial_bias import (  # noqa: E402
    roll_image,
)
from afb_segmentation.data_pipeline.preparation.yolo_export import (  # noqa: E402
    create_yolo_dataset,
)
from experiments.analysis.spatial_entropy import (  # noqa: E402
    calculate_spatial_entropy,
    get_mean_patch_heatmap,
)

RAW_SPLITS = {
    "train": "TRAINING SET",
    "validation": "VALIDATION SET",
    "test": "TEST SET",
}
EXPECTED_PAIR_COUNTS = {"train": 6000, "validation": 1000, "test": 1000}
CONDITIONS = {
    "conventional_augmentation": False,
    "spatial_debiasing": True,
}
PRETRAINED_CHECKPOINTS = {
    "without_augmentation": REPO_ROOT
    / "artifacts/models/yolo11l-seg - False/train/weights/best.pt",
    "with_spatial_debiasing": REPO_ROOT
    / "artifacts/models/yolo11l-seg - True/train/weights/best.pt",
}
DEFAULT_OUTPUT = REPO_ROOT / "experiments" / "q14_spatial_debiasing_20261030"


def _seed_everything(seed: int) -> None:
    """Set Python, NumPy, and Torch seeds for one deterministic transform task."""
    seed = int(seed) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _task_seed(base_seed: int, stage: int, index: int) -> int:
    """Return a stable seed for a particular image and transformation stage."""
    return (base_seed + stage * 1_000_003 + index * 9_176) % (2**32)


def _augment_pair_worker(task: tuple[str, str, str, str, bool, int]) -> None:
    """Apply the package's paired tile augmentation and save one copy."""
    image_path, mask_path, output_image, output_mask, shuffle_tiles, seed = task
    torch.set_num_threads(1)
    _seed_everything(seed)

    with Image.open(image_path) as image_source:
        image = image_source.copy()
    with Image.open(mask_path) as mask_source:
        mask = mask_source.copy()

    augmented_image, augmented_mask = augment_image(
        image, mask, shuffle_tiles=shuffle_tiles
    )
    augmented_image.save(output_image)
    augmented_mask.save(output_mask)


def _roll_pair_worker(task: tuple[str, str, int, int]) -> None:
    """Apply one seeded circular roll to an image and its mask in place."""
    image_path, mask_path, shift_size, seed = task
    torch.set_num_threads(1)
    _seed_everything(seed)

    with Image.open(image_path) as image_source:
        image = image_source.copy()
    with Image.open(mask_path) as mask_source:
        mask = mask_source.copy()

    rolled_image, rolled_mask = roll_image(image, mask, shift_size=shift_size)
    rolled_image.save(image_path)
    rolled_mask.save(mask_path)


def _find_raw_pairs(split_root: Path) -> list[tuple[str, Path, Path]]:
    """Find exact image-mask filename pairs in one supplied DDS3 split."""
    pairs: list[tuple[str, Path, Path]] = []
    for category in sorted(path for path in split_root.iterdir() if path.is_dir()):
        subdirectories = {
            child.name.lower(): child for child in category.iterdir() if child.is_dir()
        }
        image_dir = subdirectories.get("image")
        mask_dir = subdirectories.get("mask")
        if image_dir is None and mask_dir is None:
            continue
        if image_dir is None or mask_dir is None:
            raise ValueError(f"Missing IMAGE or MASK directory under {category}")

        image_files = [path for path in image_dir.iterdir() if path.is_file()]
        mask_files = [path for path in mask_dir.iterdir() if path.is_file()]
        unsupported = [
            path
            for path in (*image_files, *mask_files)
            if path.suffix.lower() != ".png"
        ]
        if unsupported:
            raise ValueError(
                f"Expected PNG files only; found {unsupported[0].suffix} in {category}"
            )

        images = {path.stem: path for path in image_files}
        masks = {path.stem: path for path in mask_files}
        if images.keys() != masks.keys():
            missing_masks = sorted(images.keys() - masks.keys())[:5]
            missing_images = sorted(masks.keys() - images.keys())[:5]
            raise ValueError(
                f"Unmatched files under {category}: missing masks={missing_masks}, "
                f"missing images={missing_images}"
            )

        pairs.extend(
            (category.name, images[stem], masks[stem]) for stem in sorted(images)
        )
    return pairs


def _prepare_base_dataset(raw_root: Path, base_root: Path) -> list[dict[str, str]]:
    """Copy paired DDS3 files into a canonical split layout without modifying DDS3."""
    manifest_rows: list[dict[str, str]] = []
    for split, raw_split_name in RAW_SPLITS.items():
        raw_split = raw_root / raw_split_name
        if not raw_split.is_dir():
            raise FileNotFoundError(f"DDS3 split not found: {raw_split}")
        pairs = _find_raw_pairs(raw_split)
        if len(pairs) != EXPECTED_PAIR_COUNTS[split]:
            raise ValueError(
                f"Expected {EXPECTED_PAIR_COUNTS[split]} paired {split} images, "
                f"found {len(pairs)}"
            )

        image_dir = base_root / split / "images"
        mask_dir = base_root / split / "masks"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for index, (category, source_image, source_mask) in enumerate(pairs):
            with Image.open(source_image) as image, Image.open(source_mask) as mask:
                if image.size != (400, 400) or mask.size != (400, 400):
                    raise ValueError(
                        f"Expected paired 400x400 files, got {image.size} and "
                        f"{mask.size} for {source_image.name}"
                    )
            target_name = f"MSC{index:06d}.png"
            shutil.copy2(source_image, image_dir / target_name)
            shutil.copy2(source_mask, mask_dir / target_name)
            manifest_rows.append(
                {
                    "split": split,
                    "target_name": target_name,
                    "category": category,
                    "source_image": source_image.relative_to(raw_root).as_posix(),
                    "source_mask": source_mask.relative_to(raw_root).as_posix(),
                }
            )
    return manifest_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as a UTF-8 CSV file."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, content: dict[str, Any]) -> None:
    """Write indented JSON, converting paths and NumPy scalar values."""

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    path.write_text(
        json.dumps(content, indent=2, sort_keys=True, default=convert) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_digests(root: Path, split: str) -> dict[str, str]:
    """Hash all image and mask files belonging to a split."""
    split_root = root / split
    return {
        path.relative_to(split_root).as_posix(): _sha256(path)
        for path in sorted(split_root.rglob("*.png"))
    }


def _calculate_entropy(mask_dir: Path) -> float:
    """Calculate manuscript-defined normalized spatial entropy for PNG masks."""
    heatmap = get_mean_patch_heatmap(str(mask_dir))
    return float(calculate_spatial_entropy(heatmap, kernel_size=5, stride=5))


def _build_training_copy(
    condition: str,
    shuffle_tiles: bool,
    output_root: Path,
    base_root: Path,
    seed: int,
    workers: int,
) -> tuple[Path, dict[str, float | None]]:
    """Create one isolated condition and export it to YOLO format."""
    condition_root = output_root / "data" / condition
    restructured_root = condition_root / "dds3-restructured"
    shutil.copytree(base_root, restructured_root)

    train_images = restructured_root / "train" / "images"
    train_masks = restructured_root / "train" / "masks"
    original_names = sorted(path.name for path in train_images.glob("*.png"))
    if len(original_names) != EXPECTED_PAIR_COUNTS["train"]:
        raise ValueError(f"Unexpected original training count in {condition}")
    if original_names != sorted(path.name for path in train_masks.glob("*.png")):
        raise ValueError(f"Training image-mask names do not match in {condition}")

    augmentation_tasks = []
    for index, name in enumerate(original_names):
        output_name = f"MSC{len(original_names) + index:06d}.png"
        augmentation_tasks.append(
            (
                str(train_images / name),
                str(train_masks / name),
                str(train_images / output_name),
                str(train_masks / output_name),
                shuffle_tiles,
                _task_seed(seed, stage=1, index=index),
            )
        )

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("spawn")
    ) as executor:
        for _ in tqdm(
            executor.map(_augment_pair_worker, augmentation_tasks, chunksize=4),
            total=len(augmentation_tasks),
            desc=f"Patch augmentation: {condition}",
        ):
            pass

    if shuffle_tiles:
        training_names = sorted(path.name for path in train_images.glob("*.png"))
        entropy_after_patch_augmentation = _calculate_entropy(train_masks)
        shift_size = 20
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as executor:
            for iteration in range(6):
                shift_tasks = [
                    (
                        str(train_images / name),
                        str(train_masks / name),
                        shift_size,
                        _task_seed(seed, stage=iteration + 2, index=index),
                    )
                    for index, name in enumerate(training_names)
                ]
                for _ in tqdm(
                    executor.map(_roll_pair_worker, shift_tasks, chunksize=8),
                    total=len(shift_tasks),
                    desc=f"Circular shift {iteration + 1}/6: {condition}",
                ):
                    pass
        final_entropy = _calculate_entropy(train_masks)
        entropy_values = {
            "after_patch_augmentation": entropy_after_patch_augmentation,
            "after_six_shifts": final_entropy,
        }
    else:
        entropy_values = {
            "after_patch_augmentation": _calculate_entropy(train_masks),
            "after_six_shifts": None,
        }

    # The package exporter reads ROOT_DATA and writes only inside condition_root.
    previous_root_data = os.environ.get("ROOT_DATA")
    os.environ["ROOT_DATA"] = str(condition_root)
    try:
        create_yolo_dataset(
            target_folder_name="dds3-yolo",
            src_folder_name="dds3-restructured",
        )
    finally:
        if previous_root_data is None:
            os.environ.pop("ROOT_DATA", None)
        else:
            os.environ["ROOT_DATA"] = previous_root_data

    data_yaml = condition_root / "dds3-yolo" / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(f"YOLO data file was not created: {data_yaml}")
    return data_yaml, entropy_values


def _prepare_experiment(output_root: Path, seed: int, workers: int) -> None:
    """Create the two paired conditions, unchanged evaluation splits, and manifest."""
    root_data = os.environ.get("ROOT_DATA")
    if not root_data:
        raise EnvironmentError(
            "ROOT_DATA is not set. Configure it in the repository .env file."
        )
    raw_root = Path(root_data).expanduser().resolve() / "DDS3"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"DDS3 source directory not found: {raw_root}")
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError("Experiment output must not be inside the source DDS3 folder")
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists: {output_root}. Choose a new --output-dir."
        )
    output_root.mkdir(parents=True)

    base_root = output_root / "data" / "base" / "dds3-restructured"
    base_root.parent.mkdir(parents=True)
    source_manifest = _prepare_base_dataset(raw_root, base_root)
    _write_csv(output_root / "source_file_manifest.csv", source_manifest)

    original_entropy = _calculate_entropy(base_root / "train" / "masks")
    condition_data: dict[str, dict[str, Any]] = {}
    for condition, shuffle_tiles in CONDITIONS.items():
        data_yaml, entropy_values = _build_training_copy(
            condition=condition,
            shuffle_tiles=shuffle_tiles,
            output_root=output_root,
            base_root=base_root,
            seed=seed,
            workers=workers,
        )
        condition_data[condition] = {
            "data_yaml": data_yaml,
            "shuffle_tiles": shuffle_tiles,
            "n_train_images_after_augmentation": 12_000,
            **entropy_values,
        }

    for split in ("validation", "test"):
        expected = _split_digests(base_root, split)
        for condition in CONDITIONS:
            actual = _split_digests(
                output_root / "data" / condition / "dds3-restructured", split
            )
            if actual != expected:
                raise RuntimeError(
                    f"The {split} split changed in condition {condition}"
                )

    entropy_rows = [
        {
            "condition": "original_training_data",
            "stage": "original",
            "mask_count": EXPECTED_PAIR_COUNTS["train"],
            "normalized_entropy": original_entropy,
        }
    ]
    for condition, values in condition_data.items():
        entropy_rows.append(
            {
                "condition": condition,
                "stage": "after_patch_augmentation",
                "mask_count": 12_000,
                "normalized_entropy": values["after_patch_augmentation"],
            }
        )
        if condition == "spatial_debiasing":
            entropy_rows.append(
                {
                    "condition": condition,
                    "stage": "after_six_shifts",
                    "mask_count": 12_000,
                    "normalized_entropy": values["after_six_shifts"],
                }
            )
    _write_csv(output_root / "spatial_entropy.csv", entropy_rows)

    manifest = {
        "status": "prepared",
        "question": "Q14: conventional augmentation versus spatial debiasing",
        "source_split_counts": EXPECTED_PAIR_COUNTS,
        "predefined_split_assignments_preserved": True,
        "validation_and_test_byte_identical_to_source_copy": True,
        "conditions": condition_data,
        "augmentation": {
            "one_copy_per_original_training_pair": True,
            "tile_size_pixels": [40, 40],
            "conventional_condition_shuffle_tiles": False,
            "spatial_debiasing_condition_shuffle_tiles": True,
            "circular_shift_iterations": 6,
            "circular_shift_size": 20,
            "all_shifts_apply_same_transform_to_image_and_mask": True,
            "per_task_seeded": True,
            "seed": seed,
        },
        "inference": {
            "checkpoints": {
                name: {
                    "path": path,
                    "sha256": _sha256(path) if path.is_file() else None,
                }
                for name, path in PRETRAINED_CHECKPOINTS.items()
            },
            "batch_size": 42,
            "image_size": 416,
            "split": "test",
            "single_cls": True,
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    manifest["inference"]["transform_seed"] = seed
    _write_json(output_root / "manifest.json", manifest)


def _evaluate_checkpoint(
    condition: str,
    checkpoint: Path,
    data_yaml: Path,
    output_root: Path,
    batch_size: int,
    device: str,
    workers: int,
) -> dict[str, float]:
    """Evaluate an existing AFB checkpoint without training or fine-tuning."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint}")
    test_run_dir = output_root / "test" / condition
    if test_run_dir.exists():
        raise FileExistsError(
            f"A run already exists for {condition}. Use a fresh output directory."
        )

    # Checkpoints created with a newer pathlib layout may pickle PosixPath as
    # pathlib._local.PosixPath. Map that exact class path to Python 3.12's
    # equivalent before Ultralytics unpickles the trusted local checkpoint.
    if "pathlib._local" not in sys.modules:
        pathlib_local = types.ModuleType("pathlib._local")
        pathlib_local.PosixPath = pathlib.PosixPath
        sys.modules["pathlib._local"] = pathlib_local
        pathlib._local = pathlib_local

    test_model = YOLO(str(checkpoint))
    metrics = test_model.val(
        data=str(data_yaml),
        split="test",
        imgsz=416,
        batch=batch_size,
        device=device,
        workers=workers,
        single_cls=True,
        project=str(output_root / "test"),
        name=condition,
        exist_ok=False,
        plots=True,
        verbose=False,
    )
    results = {
        str(key): float(value)
        for key, value in metrics.results_dict.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    result_path = output_root / "test_metrics" / f"{condition}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        result_path,
        {
            "condition": condition,
            "checkpoint": checkpoint,
            "test_split": "predefined DDS3 test split",
            "results": results,
        },
    )
    return results


def _write_comparison(output_root: Path, results: dict[str, dict[str, float]]) -> None:
    """Write paired metric values and absolute/relative differences."""
    baseline = results["without_augmentation"]
    debiased = results["with_spatial_debiasing"]
    keys = sorted(set(baseline) & set(debiased))
    rows = []
    for key in keys:
        before = baseline[key]
        after = debiased[key]
        rows.append(
            {
                "metric": key,
                "without_augmentation_checkpoint": before,
                "with_spatial_debiasing_checkpoint": after,
                "absolute_change": after - before,
                "relative_change_percent": (
                    100.0 * (after - before) / abs(before) if before else float("nan")
                ),
            }
        )
    _write_csv(output_root / "test_comparison.csv", rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare isolated datasets and report entropy without model training.",
    )
    mode.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Evaluate existing AFB checkpoints on the dataset prepared at --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Dedicated experiment output directory (must not already exist for preparation).",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_root = args.output_dir.expanduser().resolve()
    if not args.evaluate_only:
        _prepare_experiment(output_root, seed=args.seed, workers=args.workers)
        print(f"Prepared isolated experiment at {output_root}")
    if args.prepare_only:
        return

    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared experiment manifest not found: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("training", None)
    manifest["inference"] = {
        "checkpoints": {
            name: {
                "path": path,
                "sha256": _sha256(path) if path.is_file() else None,
            }
            for name, path in PRETRAINED_CHECKPOINTS.items()
        },
        "batch_size": args.batch_size,
        "image_size": 416,
        "split": "test",
        "single_cls": True,
    }
    results: dict[str, dict[str, float]] = {}
    shared_data_yaml = (
        output_root / "data" / "conventional_augmentation" / "dds3-yolo" / "data.yaml"
    )
    for condition, checkpoint in PRETRAINED_CHECKPOINTS.items():
        if condition in manifest.get("completed_checkpoints", []):
            result_path = output_root / "test_metrics" / f"{condition}.json"
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            results[condition] = saved["results"]
            continue
        if not shared_data_yaml.is_file():
            raise FileNotFoundError(f"YOLO dataset missing: {shared_data_yaml}")
        results[condition] = _evaluate_checkpoint(
            condition=condition,
            checkpoint=checkpoint,
            data_yaml=shared_data_yaml,
            output_root=output_root,
            batch_size=args.batch_size,
            device=str(args.device),
            workers=args.workers,
        )
        manifest.setdefault("completed_checkpoints", []).append(condition)
        manifest["status"] = "evaluation"
        _write_json(manifest_path, manifest)

    _write_comparison(output_root, results)
    manifest["status"] = "complete"
    _write_json(manifest_path, manifest)
    print(f"Completed Q14 comparison. Results are in {output_root}")


if __name__ == "__main__":
    main()
