"""Run a leakage-controlled low-load and boundary-loss experiment for Q4."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.modules.setdefault("pathlib._local", pathlib)

import cv2
import numpy as np
import torch
import ultralytics.utils.nms as yolo_nms
import yaml
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "artifacts/dataset/dds3-restructured"
YOLO_LABELS = ROOT / "artifacts/dataset/dds3-yolo/labels"
SEED = 20261030
HOLDOUT_FRACTION = 0.20
YOLO_EPOCHS = 52
YOLO_BATCH = 8
UNET_EPOCHS = 4
UNET_EARLY_STOPPING_PATIENCE = 3
YOLO_CONFIDENCE = 0.25
BOUNDARY_TOLERANCE = 2.0


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def component_count(mask_path: Path) -> int:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    return int(ndimage.label(mask)[1])


def load_manifest() -> list[dict]:
    manifest = OUT / "split_manifest.csv"
    if manifest.exists():
        with manifest.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        return [
            {
                "filename": r["filename"],
                "components": int(r["components"]),
                "stratum": r["stratum"],
                "partition": r["partition"],
                "sha256": r["sha256"],
            }
            for r in rows
        ]

    train_images = DATA / "train/images"
    train_masks = DATA / "train/masks"
    names = sorted(p.name for p in train_images.glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No training images found in {train_images}")
    missing_masks = [n for n in names if not (train_masks / n).is_file()]
    if missing_masks:
        raise RuntimeError(
            f"Missing masks for training images, e.g. {missing_masks[:5]}"
        )

    bins: dict[str, list[str]] = {
        "negative_0": [],
        "low_1_2": [],
        "low_3_10": [],
        "higher_gt10": [],
    }
    counts: dict[str, int] = {}
    for name in names:
        n = component_count(train_masks / name)
        counts[name] = n
        group = (
            "negative_0"
            if n == 0
            else "low_1_2" if n <= 2 else "low_3_10" if n <= 10 else "higher_gt10"
        )
        bins[group].append(name)

    rng = np.random.default_rng(SEED)
    holdout: set[str] = set()
    for group in ("negative_0", "low_1_2", "low_3_10"):
        group_names = bins[group]
        selected_count = int(round(HOLDOUT_FRACTION * len(group_names)))
        if selected_count:
            holdout.update(
                rng.choice(group_names, size=selected_count, replace=False).tolist()
            )

    rows = []
    for name in names:
        n = counts[name]
        group = (
            "negative_0"
            if n == 0
            else "low_1_2" if n <= 2 else "low_3_10" if n <= 10 else "higher_gt10"
        )
        rows.append(
            {
                "filename": name,
                "components": n,
                "stratum": group,
                "partition": "test_low" if name in holdout else "train",
                "sha256": image_sha256(train_images / name),
            }
        )

    other_hashes = set()
    for split in ("validation", "test"):
        for image_path in sorted((DATA / split / "images").glob("*.png")):
            other_hashes.add(image_sha256(image_path))
    train_hashes = {r["sha256"] for r in rows if r["partition"] == "train"}
    holdout_hashes = {r["sha256"] for r in rows if r["partition"] == "test_low"}
    if train_hashes & holdout_hashes or (train_hashes | holdout_hashes) & other_hashes:
        raise RuntimeError(
            "Exact image duplicates were found across the experimental partitions"
        )

    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "components", "stratum", "partition", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)
    split_counts = Counter((r["partition"], r["stratum"]) for r in rows)
    summary = {
        "seed": SEED,
        "holdout_fraction_per_low_count_stratum": HOLDOUT_FRACTION,
        "selection": "Stratified random holdout from the pre-existing training split; component count is a proxy, not a clinical load label.",
        "counts_by_partition_and_stratum": {
            f"{a}/{b}": n for (a, b), n in sorted(split_counts.items())
        },
        "train_images": sum(r["partition"] == "train" for r in rows),
        "heldout_lowload_images": sum(r["partition"] == "test_low" for r in rows),
        "exact_image_hash_overlap_with_validation_or_test": 0,
        "exact_image_hash_overlap_between_train_and_lowload_holdout": 0,
    }
    (OUT / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return rows


def prepare_yolo_dataset(rows: list[dict]) -> None:
    base = OUT / "yolo_dataset"
    marker = base / "prepared.json"
    train_names = [r["filename"] for r in rows if r["partition"] == "train"]
    val_names = sorted(p.name for p in (DATA / "validation/images").glob("*.png"))
    if marker.exists():
        state = json.loads(marker.read_text())
        if state.get("train_images") == len(train_names) and state.get(
            "validation_images"
        ) == len(val_names):
            return
        shutil.rmtree(base)

    for split, names in (("train", train_names), ("val", val_names)):
        image_out = base / "images" / split
        label_out = base / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        source_split = "train" if split == "train" else "validation"
        for name in names:
            image_src = DATA / source_split / "images" / name
            label_src = YOLO_LABELS / source_split / f"{Path(name).stem}.txt"
            if not image_src.is_file() or not label_src.is_file():
                raise FileNotFoundError(
                    f"Missing YOLO training pair: {image_src} / {label_src}"
                )
            shutil.copy2(image_src, image_out / name)
            shutil.copy2(label_src, label_out / label_src.name)

    dataset_yaml = {
        "path": str(base),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "AFB"},
    }
    (base / "data.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False))
    marker.write_text(
        json.dumps(
            {"train_images": len(train_names), "validation_images": len(val_names)},
            indent=2,
        )
    )


def train_yolo() -> Path:
    best = OUT / "yolo_runs/lowload_holdout/weights/best.pt"
    if best.exists():
        print(f"Reusing completed YOLO checkpoint: {best}", flush=True)
        return best
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training experiment")

    dataset_yaml = OUT / "yolo_dataset/data.yaml"
    weights_dir = ROOT / "artifacts/models/low_load"
    model = YOLO(str(weights_dir / "yolo11l-seg.pt"))
    # Ultralytics resolves its AMP-check yolo11n.pt through weights_dir.
    previous_weights_dir = SETTINGS["weights_dir"]
    # Change only the in-memory lookup; the installed downloader expects a Path.
    dict.__setitem__(SETTINGS, "weights_dir", weights_dir)
    try:
        model.train(
            data=str(dataset_yaml),
            epochs=YOLO_EPOCHS,
            imgsz=416,
            batch=YOLO_BATCH,
            device=0,
            workers=4,
            project=str(OUT / "yolo_runs"),
            name="lowload_holdout",
            exist_ok=True,
            seed=1,
            deterministic=True,
            single_cls=True,
            cache=False,
            optimizer="Adam",
            lr0=0.00013,
            lrf=0.0001,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=0,
            dropout=0.441,
            close_mosaic=False,
            mosaic=1.0,
            auto_augment="randaugment",
            erasing=0.4,
            patience=YOLO_EPOCHS,
            amp=True,
            plots=False,
            verbose=True,
        )
    finally:
        dict.__setitem__(SETTINGS, "weights_dir", previous_weights_dir)
    if not best.exists():
        raise FileNotFoundError(
            f"YOLO training did not produce the expected checkpoint: {best}"
        )
    training = {
        "initial_weights": "Ultralytics yolo11l-seg.pt pretrained weights; not the DDS3-trained checkpoint",
        "epochs": YOLO_EPOCHS,
        "batch_size": YOLO_BATCH,
        "image_size": 416,
        "device": torch.cuda.get_device_name(0),
        "optimizer": "Adam",
        "learning_rate": 0.00013,
        "seed": 1,
        "deterministic": True,
        "validation": "Existing DDS3 validation split; used for YOLO validation/early stopping only",
        "checkpoint": str(best.relative_to(ROOT)),
    }
    (OUT / "yolo_training.json").write_text(json.dumps(training, indent=2))
    return best


class CudaYoloFilter:
    def __init__(self, model: YOLO, size: int = 400, conf: float = YOLO_CONFIDENCE):
        from afb_segmentation.utils.yolo import YoloModel

        self.wrapper = YoloModel(model, input_size=size, conf=conf)
        self.model = model
        self.output_size = self.wrapper.output_size
        self.conf = conf

    def predict(self, images: list[np.ndarray]):
        return self.model.predict(
            source=images,
            imgsz=self.output_size,
            conf=self.conf,
            agnostic_nms=True,
            verbose=False,
            device=0,
            batch=len(images),
        )

    def extract_masks(self, predictions):
        return self.wrapper.extract_masks(predictions)


def configure_nms_guard() -> None:
    original = yolo_nms.non_max_suppression

    def guarded(*args, **kwargs):
        kwargs["max_time_img"] = max(float(kwargs.get("max_time_img", 0.05)), 10.0)
        return original(*args, **kwargs)

    yolo_nms.non_max_suppression = guarded


def preprocess_split(
    split_name: str,
    image_paths: list[Path],
    source_mask_dir: Path,
    predictor: CudaYoloFilter,
) -> None:
    from afb_segmentation.utils.image_filtering import distance_based_filtering

    out = OUT / "unet_inputs" / split_name
    image_out, mask_out, yolo_out = out / "images", out / "masks", out / "yolo_masks"
    expected = len(image_paths)
    if (out / "complete.json").exists():
        state = json.loads((out / "complete.json").read_text())
        if state.get("count") == expected:
            return
        shutil.rmtree(out)
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    yolo_out.mkdir(parents=True, exist_ok=True)

    batch_size = 16
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as im:
                images.append(np.asarray(im.convert("RGB")))
        predictions = predictor.predict(images)
        yolo_masks = predictor.extract_masks(predictions)
        for path, image, yolo_mask in zip(batch_paths, images, yolo_masks):
            binary = yolo_mask > 0
            filtered = distance_based_filtering(image, binary.astype(np.uint8) * 255)
            filtered = cv2.resize(filtered, (400, 400))
            Image.fromarray(filtered).save(image_out / path.name)
            Image.fromarray(binary.astype(np.uint8) * 255).save(yolo_out / path.name)
            mask_src = source_mask_dir / path.name
            if not mask_src.is_file():
                raise FileNotFoundError(
                    f"Ground-truth mask not found for {path}: {mask_src}"
                )
            shutil.copy2(mask_src, mask_out / path.name)
        done = min(start + len(batch_paths), len(image_paths))
        print(f"Preprocessed {split_name}: {done}/{expected}", flush=True)

    (out / "complete.json").write_text(json.dumps({"count": expected}, indent=2))


def preprocess_all(rows: list[dict], checkpoint: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for preprocessing")
    configure_nms_guard()
    yolo = YOLO(str(checkpoint))
    yolo.model.to("cuda:0")
    predictor = CudaYoloFilter(yolo)
    train_names = [r["filename"] for r in rows if r["partition"] == "train"]
    holdout_names = [r["filename"] for r in rows if r["partition"] == "test_low"]
    splits = [
        (
            "train",
            [DATA / "train/images" / n for n in train_names],
            DATA / "train/masks",
        ),
        (
            "validation",
            sorted((DATA / "validation/images").glob("*.png")),
            DATA / "validation/masks",
        ),
        (
            "test_low",
            [DATA / "train/images" / n for n in holdout_names],
            DATA / "train/masks",
        ),
        (
            "test_high",
            sorted((DATA / "test/images").glob("*.png")),
            DATA / "test/masks",
        ),
    ]
    for name, images, masks in splits:
        preprocess_split(name, images, masks, predictor)
    (OUT / "preprocessing.json").write_text(
        json.dumps(
            {
                "yolo_checkpoint": str(checkpoint.relative_to(ROOT)),
                "confidence_threshold": YOLO_CONFIDENCE,
                "input_size": 400,
                "prediction_batch_size": 16,
                "nms_time_guard_seconds_per_image": 10.0,
            },
            indent=2,
        )
    )
    del predictor, yolo
    torch.cuda.empty_cache()


class CachedMaskDataset(Dataset):
    def __init__(self, root: Path):
        self.root = root
        self.images = sorted((root / "images").glob("*.png"))
        self.masks = root / "masks"
        if not self.images:
            raise FileNotFoundError(f"No preprocessed images found under {root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        path = self.images[index]
        image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask = np.asarray(Image.open(self.masks / path.name).convert("L")) > 127
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor


def make_loss(cfg, use_boundary: bool):
    from afb_segmentation.models.loss.segmentation import CompoundLoss

    return CompoundLoss(
        dice_weight=cfg.dice_weight,
        focal_weight=cfg.focal_weight,
        boundary_weight=cfg.boundary_weight,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        use_boundary=use_boundary,
    )


def run_unet_variant(
    cfg, initial_state: dict, use_boundary: bool, device: torch.device
) -> Path:
    from afb_segmentation.models.training.unet_components import (
        get_model,
        get_optimizer,
    )

    variant = "boundary_true" if use_boundary else "boundary_false"
    out = OUT / "unet_runs" / variant
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / "best.pt"
    history_path = out / "history.csv"
    if checkpoint_path.exists() and history_path.exists():
        print(f"Reusing completed U-Net checkpoint: {checkpoint_path}", flush=True)
        return checkpoint_path

    torch.manual_seed(SEED + 1)
    torch.cuda.manual_seed_all(SEED + 1)
    model = get_model(cfg).to(device)
    model.load_state_dict(initial_state)
    loss_fn = make_loss(cfg, use_boundary).to(device)
    optimizer = get_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg.scheduler_patience,
        factor=cfg.scheduler_factor,
        min_lr=cfg.scheduler_min_lr,
    )

    generator = torch.Generator().manual_seed(SEED + 2)
    train_loader = DataLoader(
        CachedMaskDataset(OUT / "unet_inputs/train"),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=generator,
    )
    val_loader = DataLoader(
        CachedMaskDataset(OUT / "unet_inputs/validation"),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, UNET_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_items = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).float()
            smoothed = labels * (1.0 - 0.08) + 0.5 * 0.08
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = loss_fn(predictions, smoothed)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(images)
            train_items += len(images)

        model.eval()
        val_loss_sum = 0.0
        val_items = 0
        tp = fp = fn = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True).float()
                labels = labels.to(device, non_blocking=True).float()
                predictions = model(images)
                loss = loss_fn(predictions, labels)
                val_loss_sum += float(loss) * len(images)
                val_items += len(images)
                predicted = predictions > 0.5
                expected = labels > 0.5
                tp += int((predicted & expected).sum())
                fp += int((predicted & ~expected).sum())
                fn += int((~predicted & expected).sum())

        train_loss = train_loss_sum / max(train_items, 1)
        val_loss = val_loss_sum / max(val_items, 1)
        val_dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
        scheduler.step(val_loss)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_pixel_dice": val_dice,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "use_boundary": use_boundary,
            }
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_pixel_dice": val_dice,
                    "use_boundary": use_boundary,
                    "seed": SEED + 1,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        print(
            f"U-Net {variant}: epoch {epoch}/{UNET_EPOCHS}, "
            f"train_loss={train_loss:.5f}, val_loss={val_loss:.5f}, "
            f"val_pixel_dice={val_dice:.5f}",
            flush=True,
        )
        if epochs_without_improvement >= UNET_EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping {variant}: validation loss did not improve for "
                f"{UNET_EARLY_STOPPING_PATIENCE} consecutive epochs; selecting best checkpoint.",
                flush=True,
            )
            break

    with history_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    del model, loss_fn, optimizer, scheduler
    torch.cuda.empty_cache()
    return checkpoint_path


def train_unets() -> tuple[Path, Path]:
    from afb_segmentation.models.training.unet_components import get_model
    from afb_segmentation.models.training.unet_config import load_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for U-Net training")
    cfg = load_config(str(ROOT))
    device = torch.device("cuda:0")
    cfg.device = device
    torch.manual_seed(SEED + 1)
    torch.cuda.manual_seed_all(SEED + 1)
    base_model = get_model(cfg)
    initial_state = {
        k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()
    }
    initial_hash = hashlib.sha256(
        b"".join(initial_state[k].numpy().tobytes() for k in sorted(initial_state))
    ).hexdigest()
    torch.save(initial_state, OUT / "unet_initial_state.pt")
    del base_model
    true_ckpt = run_unet_variant(cfg, initial_state, True, device)
    false_ckpt = run_unet_variant(cfg, initial_state, False, device)
    (OUT / "unet_training.json").write_text(
        json.dumps(
            {
                "architecture": "AttentionUNetMaskPredictor from project configuration",
                "training_images": len(
                    list((OUT / "unet_inputs/train/images").glob("*.png"))
                ),
                "validation_images": len(
                    list((OUT / "unet_inputs/validation/images").glob("*.png"))
                ),
                "epochs_per_variant": UNET_EPOCHS,
                "batch_size": cfg.batch_size,
                "optimizer": cfg.optimizer,
                "learning_rate": cfg.lr,
                "label_smoothing": 0.08,
                "dice_weight": cfg.dice_weight,
                "focal_weight": cfg.focal_weight,
                "boundary_weight": cfg.boundary_weight,
                "initial_state_sha256": initial_hash,
                "variants": {
                    "boundary_true": str(true_ckpt.relative_to(ROOT)),
                    "boundary_false": str(false_ckpt.relative_to(ROOT)),
                },
                "checkpoint_selection": f"Minimum validation compound loss with a {UNET_EPOCHS}-epoch cap and early-stopping patience of {UNET_EARLY_STOPPING_PATIENCE} consecutive non-improving epochs; validation split is the existing higher-load DDS3 validation set.",
                "training_order": "Both variants use the same initial state, seed, and shuffled loader generator seed.",
            },
            indent=2,
        )
    )
    return true_ckpt, false_ckpt


def component_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred_labels, n_pred = ndimage.label(pred.astype(np.uint8))
    gt_labels, n_gt = ndimage.label(gt.astype(np.uint8))
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=n_pred + 1)
    gt_sizes = np.bincount(gt_labels.ravel(), minlength=n_gt + 1)
    if n_pred and n_gt:
        pair_ids = pred_labels.ravel().astype(np.int64) * (n_gt + 1) + gt_labels.ravel()
        pairs = np.bincount(pair_ids, minlength=(n_pred + 1) * (n_gt + 1))
        intersections = pairs.reshape(n_pred + 1, n_gt + 1)[1:, 1:]
        unions = pred_sizes[1:, None] + gt_sizes[None, 1:] - intersections
        ious = intersections / (unions + 1e-6)
    else:
        ious = np.zeros((n_pred, n_gt), dtype=np.float64)
    aps = []
    values = {}
    for threshold in [0.5 + 0.05 * i for i in range(10)]:
        matched_gt: set[int] = set()
        tp = fp = 0
        if n_pred:
            order = np.argsort(-ious.max(axis=1)) if n_gt else np.arange(n_pred)
            for p_idx in order:
                best_gt, best_iou = -1, threshold
                for g_idx in range(n_gt):
                    if g_idx not in matched_gt and ious[p_idx, g_idx] >= best_iou:
                        best_gt, best_iou = g_idx, ious[p_idx, g_idx]
                if best_gt >= 0:
                    tp += 1
                    matched_gt.add(best_gt)
                else:
                    fp += 1
        fn = n_gt - len(matched_gt)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        ap = precision if recall > 0 else 0.0
        aps.append(ap)
        if abs(threshold - 0.5) < 1e-9:
            values = {
                "tp50": tp,
                "fp50": fp,
                "fn50": fn,
                "precision50": precision,
                "recall50": recall,
                "ap50": ap,
            }
    values.update(
        {
            "pred_components": int(n_pred),
            "gt_components": int(n_gt),
            "ap50_95": float(np.mean(aps)),
        }
    )
    return values


def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    return {
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
        "pixel_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
    }


def boundary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    structure = np.ones((3, 3), dtype=bool)
    pred_edge = pred & ~ndimage.binary_erosion(
        pred, structure=structure, border_value=0
    )
    gt_edge = gt & ~ndimage.binary_erosion(gt, structure=structure, border_value=0)
    pred_count, gt_count = int(pred_edge.sum()), int(gt_edge.sum())
    matched_pred = (
        int(
            np.logical_and(
                pred_edge,
                ndimage.distance_transform_edt(~gt_edge) <= BOUNDARY_TOLERANCE,
            ).sum()
        )
        if pred_count
        else 0
    )
    matched_gt = (
        int(
            np.logical_and(
                gt_edge,
                ndimage.distance_transform_edt(~pred_edge) <= BOUNDARY_TOLERANCE,
            ).sum()
        )
        if gt_count
        else 0
    )
    precision = (
        matched_pred / pred_count if pred_count else (1.0 if not gt_count else 0.0)
    )
    recall = matched_gt / gt_count if gt_count else (1.0 if not pred_count else 0.0)
    return {
        "boundary_pred_pixels": pred_count,
        "boundary_gt_pixels": gt_count,
        "boundary_matched_pred_pixels": matched_pred,
        "boundary_matched_gt_pixels": matched_gt,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
    }


def combined_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {
        **component_metrics(pred, gt),
        **pixel_metrics(pred, gt),
        **boundary_metrics(pred, gt),
    }


def aggregate(rows: list[dict], prefix: str) -> dict:
    if not rows:
        return {"images": 0}
    tp = sum(int(r[f"{prefix}_tp50"]) for r in rows)
    fp = sum(int(r[f"{prefix}_fp50"]) for r in rows)
    fn = sum(int(r[f"{prefix}_fn50"]) for r in rows)
    tp_px = sum(int(r[f"{prefix}_tp_pixels"]) for r in rows)
    fp_px = sum(int(r[f"{prefix}_fp_pixels"]) for r in rows)
    fn_px = sum(int(r[f"{prefix}_fn_pixels"]) for r in rows)
    pb = sum(int(r[f"{prefix}_boundary_pred_pixels"]) for r in rows)
    gb = sum(int(r[f"{prefix}_boundary_gt_pixels"]) for r in rows)
    mpb = sum(int(r[f"{prefix}_boundary_matched_pred_pixels"]) for r in rows)
    mgb = sum(int(r[f"{prefix}_boundary_matched_gt_pixels"]) for r in rows)
    bprec = mpb / pb if pb else 0.0
    brec = mgb / gb if gb else 0.0
    negative = [r for r in rows if int(r["gt_components"]) == 0]
    return {
        "images": len(rows),
        "gt_components": sum(int(r["gt_components"]) for r in rows),
        "pred_components": sum(int(r[f"{prefix}_pred_components"]) for r in rows),
        "component_precision_iou50": tp / (tp + fp) if tp + fp else 0.0,
        "component_recall_iou50": tp / (tp + fn) if tp + fn else 0.0,
        "mean_image_ap50": float(np.mean([float(r[f"{prefix}_ap50"]) for r in rows])),
        "mean_image_ap50_95": float(
            np.mean([float(r[f"{prefix}_ap50_95"]) for r in rows])
        ),
        "pixel_precision_pooled": tp_px / (tp_px + fp_px) if tp_px + fp_px else 0.0,
        "pixel_recall_pooled": tp_px / (tp_px + fn_px) if tp_px + fn_px else 0.0,
        "pixel_dice_pooled": (
            2 * tp_px / (2 * tp_px + fp_px + fn_px)
            if 2 * tp_px + fp_px + fn_px
            else 1.0
        ),
        "boundary_precision_pooled": bprec,
        "boundary_recall_pooled": brec,
        "boundary_f1_pooled": (
            2 * bprec * brec / (bprec + brec) if bprec + brec else 0.0
        ),
        "mean_image_boundary_f1": float(
            np.mean([float(r[f"{prefix}_boundary_f1"]) for r in rows])
        ),
        "negative_images": len(negative),
        "negative_images_with_any_predicted_component": sum(
            int(r[f"{prefix}_pred_components"]) > 0 for r in negative
        ),
        "negative_false_positive_image_rate": (
            sum(int(r[f"{prefix}_pred_components"]) > 0 for r in negative)
            / len(negative)
            if negative
            else None
        ),
        "mean_predicted_components_per_negative_image": (
            float(np.mean([int(r[f"{prefix}_pred_components"]) for r in negative]))
            if negative
            else None
        ),
    }


def bootstrap_ci(values: np.ndarray, seed: int = SEED, repeats: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        boot[i] = float(rng.choice(values, size=len(values), replace=True).mean())
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "images": int(len(values)),
        "bootstrap_repeats": repeats,
    }


def evaluate_all(rows_manifest: list[dict], checkpoints: tuple[Path, Path]) -> None:
    from afb_segmentation.models.training.unet_components import get_model
    from afb_segmentation.models.training.unet_config import load_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    device = torch.device("cuda:0")
    cfg = load_config(str(ROOT))
    cfg.device = device
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    strata = {r["filename"]: r["stratum"] for r in rows_manifest}
    records: dict[tuple[str, str], dict] = {}
    prediction_root = OUT / "predictions"

    for split in ("test_low", "test_high"):
        root = OUT / "unet_inputs" / split
        images = sorted((root / "images").glob("*.png"))
        for path in images:
            gt = np.asarray(Image.open(root / "masks" / path.name).convert("L")) > 127
            yolo_pred = (
                np.asarray(Image.open(root / "yolo_masks" / path.name).convert("L")) > 0
            )
            key = (split, path.name)
            record = {
                "split": split,
                "filename": path.name,
                "load_stratum": (
                    "higher_gt10"
                    if split == "test_high"
                    else strata.get(path.name, "higher_gt10")
                ),
                "gt_components": int(ndimage.label(gt.astype(np.uint8))[1]),
            }
            record.update(
                {f"yolo_{k}": v for k, v in combined_metrics(yolo_pred, gt).items()}
            )
            records[key] = record

    for variant, checkpoint in zip(("boundary_true", "boundary_false"), checkpoints):
        model = get_model(cfg).to(device)
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        for split in ("test_low", "test_high"):
            root = OUT / "unet_inputs" / split
            dataset = CachedMaskDataset(root)
            loader = DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            split_pred_dir = prediction_root / split / variant
            split_pred_dir.mkdir(parents=True, exist_ok=True)
            paths = dataset.images
            offset = 0
            with torch.no_grad():
                for images, masks in loader:
                    predictions = (
                        model(images.to(device, non_blocking=True).float())[:, 0]
                        .cpu()
                        .numpy()
                        > 0.5
                    )
                    mask_batch = masks[:, 0].numpy() > 0.5
                    for pred, gt, path in zip(
                        predictions,
                        mask_batch,
                        paths[offset : offset + len(predictions)],
                    ):
                        record = records[(split, path.name)]
                        record.update(
                            {
                                f"{variant}_{k}": v
                                for k, v in combined_metrics(pred, gt).items()
                            }
                        )
                        Image.fromarray(pred.astype(np.uint8) * 255).save(
                            split_pred_dir / path.name
                        )
                    offset += len(predictions)
            print(f"Evaluated {variant} on {split}: {len(dataset)} images", flush=True)
        del model
        torch.cuda.empty_cache()

    rows = list(records.values())
    fields = list(rows[0])
    with (OUT / "per_image_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    groups = {}
    for split in ("test_low", "test_high"):
        for group in ("all", "negative_0", "low_1_2", "low_3_10", "higher_gt10"):
            subset = [
                r
                for r in rows
                if r["split"] == split
                and (group == "all" or r["load_stratum"] == group)
            ]
            if not subset:
                continue
            groups[f"{split}/{group}"] = {
                prefix: aggregate(subset, prefix)
                for prefix in ("yolo", "boundary_true", "boundary_false")
            }

    low_rows = [r for r in rows if r["split"] == "test_low"]
    paired = {}
    for metric in ("boundary_f1", "pixel_dice", "recall50"):
        values = np.asarray(
            [
                float(r[f"boundary_true_{metric}"])
                - float(r[f"boundary_false_{metric}"])
                for r in low_rows
            ]
        )
        paired[metric] = bootstrap_ci(values)
    positive_low = [r for r in low_rows if int(r["gt_components"]) > 0]
    for metric in ("boundary_f1", "pixel_dice", "recall50"):
        values = np.asarray(
            [
                float(r[f"boundary_true_{metric}"])
                - float(r[f"boundary_false_{metric}"])
                for r in positive_low
            ]
        )
        paired[f"positive_low_only_{metric}"] = bootstrap_ci(values, seed=SEED + 1)

    summary = {
        "experiment": "Leakage-controlled internal low-count holdout and U-Net boundary-loss ablation",
        "device": torch.cuda.get_device_name(0),
        "test_low_is_internal": True,
        "test_low_definition": "20% stratified random holdout from the original train split, with mask component-count strata 0, 1-2, and 3-10; component count is only a proxy for bacterial load.",
        "models": {
            "yolo": "Fresh Ultralytics yolo11l-seg.pt pretrained weights, retrained without test_low images.",
            "unet_variants": f"Same Attention U-Net initialization, image inputs, masks, YOLO checkpoint, batch order, seed, and at most {UNET_EPOCHS} epochs with patience {UNET_EARLY_STOPPING_PATIENCE}; only use_boundary differs.",
        },
        "boundary_tolerance_pixels": BOUNDARY_TOLERANCE,
        "groups": groups,
        "paired_bootstrap_boundary_true_minus_false_on_test_low": paired,
        "limitations": [
            "The low-count holdout comes from the DDS3 training pool, not an external clinical cohort.",
            "Connected-component count is not a validated clinical bacterial-load category.",
            "Binary masks contain no staining-artifact, debris, or overlap subtype labels; these causes cannot be evaluated without expert annotations.",
            "Connected components merge touching foreground regions and cannot measure bacillus separation when annotations merge objects.",
            f"U-Net checkpoints were selected by minimum validation loss with a {UNET_EPOCHS}-epoch cap and patience {UNET_EARLY_STOPPING_PATIENCE}; this research-only loop does not modify the production trainer.",
            "YOLO training used batch size 8 to fit this 16-GB device; historical training used a larger batch, so absolute results may differ from the manuscript run.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    write_report(summary)


def write_report(summary: dict) -> None:
    lines = [
        "# Low-count held-out evaluation and boundary-loss ablation",
        "",
        "## Experimental design",
        "",
        "A fixed, stratified holdout was sampled from the original DDS3 training split before training. Twenty percent of images in each mask-component stratum (0, 1-2, and 3-10) were reserved for low-count evaluation; all remaining training images were used to train a fresh YOLOv11l-seg model from Ultralytics pretrained weights. Exact image hashes were checked to avoid overlap between the training subset, low-count holdout, and the pre-existing validation/test splits. The pre-existing validation set was used for training selection; the original high-load test set was evaluated separately.",
        "",
        f"Two U-Net variants were trained on the same YOLO-filtered inputs from the same training subset. They used identical initialization, shuffled batch order, seed, optimizer, a maximum of {UNET_EPOCHS} epochs, and early-stopping patience of {UNET_EARLY_STOPPING_PATIENCE} epochs without improvement; the only difference was whether the configured boundary-loss term was enabled. Each checkpoint was selected by minimum validation loss. This comparison measures the boundary-loss ablation under this experimental setup; it does not by itself classify artifact causes.",
        "",
        "The low-count designation is based on connected components in the binary mask and is not a clinical bacterial-load classification. The low-count holdout is internal to DDS3 rather than an external cohort.",
        "",
        "## Results",
        "",
    ]
    for group, results in summary["groups"].items():
        lines.extend(
            [
                f"### {group}",
                "",
                "| Model | Component precision | Component recall | Pixel Dice | Boundary F1 | Negative FP image rate |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for key, label in (
            ("yolo", "YOLO candidates"),
            ("boundary_true", "U-Net, boundary loss on"),
            ("boundary_false", "U-Net, boundary loss off"),
        ):
            m = results[key]
            fpr = m["negative_false_positive_image_rate"]
            fpr_text = "—" if fpr is None else f"{fpr:.4f}"
            lines.append(
                f"| {label} | {m['component_precision_iou50']:.4f} | {m['component_recall_iou50']:.4f} | {m['pixel_dice_pooled']:.4f} | {m['boundary_f1_pooled']:.4f} | {fpr_text} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Paired boundary-loss comparison on the low-count holdout",
            "",
            "The following intervals are image-level paired bootstrap 95% confidence intervals for boundary-loss-on minus boundary-loss-off. A positive difference favors the boundary-loss-on model for that metric.",
            "",
            "| Metric | Mean difference | 95% CI | Images |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in (
        "boundary_f1",
        "pixel_dice",
        "recall50",
        "positive_low_only_boundary_f1",
        "positive_low_only_pixel_dice",
        "positive_low_only_recall50",
    ):
        x = summary["paired_bootstrap_boundary_true_minus_false_on_test_low"][metric]
        lines.append(
            f"| {metric} | {x['mean']:.4f} | [{x['ci95_low']:.4f}, {x['ci95_high']:.4f}] | {x['images']} |"
        )
    lines.extend(
        [
            "",
            "## Limits on the referee's artifact question",
            "",
            "The masks are binary and do not identify staining artifacts, debris, scratches, or overlapping bacilli. False-positive regions therefore cannot be assigned to these causes from this experiment, and touching objects that are merged in the annotations cannot be scored separately. Expert review with subtype labels is still required to address those parts of the question.",
            "",
            "## Files",
            "",
            "- `split_manifest.csv` records each image, ground-truth component count, stratum, assigned partition, and image hash.",
            "- `summary.json` contains aggregate metrics and paired bootstrap intervals.",
            "- `per_image_metrics.csv` contains image-level YOLO and U-Net results.",
            "- `unet_runs/` contains both checkpoint histories and best weights.",
            "- `predictions/` contains predicted masks for the low-count and original test splits.",
            "- `yolo_runs/` contains the retrained detector run.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "yolo", "preprocess", "unet", "evaluate", "all"),
        default="all",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = load_manifest()
    if args.stage in ("prepare", "all"):
        prepare_yolo_dataset(rows)
        print(
            json.dumps(json.loads((OUT / "split_summary.json").read_text()), indent=2),
            flush=True,
        )
        if args.stage == "prepare":
            return
    if args.stage in ("yolo", "all"):
        prepare_yolo_dataset(rows)
        checkpoint = train_yolo()
        print(f"YOLO checkpoint ready: {checkpoint}", flush=True)
        if args.stage == "yolo":
            return
    else:
        checkpoint = OUT / "yolo_runs/lowload_holdout/weights/best.pt"
    if args.stage in ("preprocess", "all"):
        preprocess_all(rows, checkpoint)
        if args.stage == "preprocess":
            return
    if args.stage in ("unet", "all"):
        train_unets()
        if args.stage == "unet":
            return
    if args.stage in ("evaluate", "all"):
        checkpoints = (
            OUT / "unet_runs/boundary_true/best.pt",
            OUT / "unet_runs/boundary_false/best.pt",
        )
        evaluate_all(rows, checkpoints)
    print(f"Completed stage {args.stage} in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
