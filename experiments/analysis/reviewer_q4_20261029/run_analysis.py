"""Read-only, post-hoc Q4 analysis for the existing DDS3 test split."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import ultralytics
import ultralytics.utils.nms as yolo_nms
from PIL import Image
from scipy import ndimage
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from afb_segmentation.models.metrics.instance_segmentation import (
    compute_instance_segmentation_metrics,
)
from afb_segmentation.models.training.unet_components import get_model
from afb_segmentation.models.training.unet_config import load_config
from afb_segmentation.utils.image_filtering import distance_based_filtering
from afb_segmentation.utils.yolo import YoloModel

OUT = Path(__file__).resolve().parent
DATA = ROOT / "artifacts/dataset/dds3-restructured/test"
SNAPSHOT = ROOT / "artifacts/dataset/snapshots/dds3-restructured/test"
UNET_CKPT = ROOT / "artifacts/models/unet/with_augmentation/weights/best.pt"
SAVED_METRICS = (
    ROOT / "artifacts/models/unet/with_augmentation/results/test_results.csv"
)
THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]


def component_scores(pred: np.ndarray, gt: np.ndarray):
    """Match 4-connected components with the repository's greedy IoU rule."""
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

    by_threshold = {}
    matched_pred_50 = set()
    matched_gt_50 = set()
    for threshold in THRESHOLDS:
        matched_gt = set()
        matched_pred = set()
        tp = 0
        fp = 0
        if n_pred:
            order = np.argsort(-ious.max(axis=1)) if n_gt else np.arange(n_pred)
            for p_idx in order:
                best_gt = -1
                best_iou = threshold
                for g_idx in range(n_gt):
                    if g_idx not in matched_gt and ious[p_idx, g_idx] >= best_iou:
                        best_iou = ious[p_idx, g_idx]
                        best_gt = g_idx
                if best_gt >= 0:
                    tp += 1
                    matched_gt.add(best_gt)
                    matched_pred.add(int(p_idx))
                else:
                    fp += 1
        fn = n_gt - len(matched_gt)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        ap = precision if recall > 0 else 0.0
        by_threshold[threshold] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "ap": float(ap),
        }
        if threshold == 0.5:
            matched_pred_50 = matched_pred
            matched_gt_50 = matched_gt

    pred_slices = ndimage.find_objects(pred_labels)
    gt_slices = ndimage.find_objects(gt_labels)
    details = []
    for p_idx in range(n_pred):
        if p_idx in matched_pred_50:
            continue
        sl = pred_slices[p_idx]
        best_iou = float(ious[p_idx].max()) if n_gt else 0.0
        details.append(
            {
                "kind": "false_positive",
                "component": p_idx + 1,
                "area_pixels": int(pred_sizes[p_idx + 1]),
                "best_iou_to_gt": best_iou,
                "x_min": int(sl[1].start),
                "y_min": int(sl[0].start),
                "x_max_exclusive": int(sl[1].stop),
                "y_max_exclusive": int(sl[0].stop),
            }
        )
    for g_idx in range(n_gt):
        if g_idx in matched_gt_50:
            continue
        sl = gt_slices[g_idx]
        best_iou = float(ious[:, g_idx].max()) if n_pred else 0.0
        details.append(
            {
                "kind": "false_negative",
                "component": g_idx + 1,
                "area_pixels": int(gt_sizes[g_idx + 1]),
                "best_iou_to_gt": best_iou,
                "x_min": int(sl[1].start),
                "y_min": int(sl[0].start),
                "x_max_exclusive": int(sl[1].stop),
                "y_max_exclusive": int(sl[0].stop),
            }
        )

    s50 = by_threshold[0.5]
    return {
        "gt_components": int(n_gt),
        "pred_components": int(n_pred),
        "tp50": s50["tp"],
        "fp50": s50["fp"],
        "fn50": s50["fn"],
        "precision50": s50["precision"],
        "recall50": s50["recall"],
        "ap50": s50["ap"],
        "ap50_95": float(np.mean([v["ap"] for v in by_threshold.values()])),
        "details": details,
    }


def pixel_scores(pred: np.ndarray, gt: np.ndarray):
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    union = tp + fp + fn
    return {
        "tp_pixels": tp,
        "fp_pixels": fp,
        "fn_pixels": fn,
        "pixel_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pixel_recall": tp / (tp + fn) if tp + fn else 0.0,
        "pixel_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "pixel_iou": tp / union if union else 1.0,
        "image_iou": tp / (union + 1e-6),
        "pred_gt_area_ratio": (tp + fp) / (tp + fn) if tp + fn else 0.0,
    }


def boundary_scores(pred: np.ndarray, gt: np.ndarray, tolerance: float = 2.0):
    """Measure boundary agreement within a Euclidean pixel-distance tolerance."""
    structure = np.ones((3, 3), dtype=bool)
    pred_boundary = pred & ~ndimage.binary_erosion(
        pred, structure=structure, border_value=0
    )
    gt_boundary = gt & ~ndimage.binary_erosion(gt, structure=structure, border_value=0)
    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())
    matched_pred = (
        int(
            np.logical_and(
                pred_boundary, ndimage.distance_transform_edt(~gt_boundary) <= tolerance
            ).sum()
        )
        if pred_count
        else 0
    )
    matched_gt = (
        int(
            np.logical_and(
                gt_boundary, ndimage.distance_transform_edt(~pred_boundary) <= tolerance
            ).sum()
        )
        if gt_count
        else 0
    )
    precision = (
        matched_pred / pred_count if pred_count else (1.0 if not gt_count else 0.0)
    )
    recall = matched_gt / gt_count if gt_count else (1.0 if not pred_count else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "boundary_pred_pixels": pred_count,
        "boundary_gt_pixels": gt_count,
        "boundary_matched_pred_pixels": matched_pred,
        "boundary_matched_gt_pixels": matched_gt,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
    }


class CudaYoloFilter(YoloModel):
    """Use the repository wrapper settings while explicitly selecting CUDA."""

    def __call__(self, x):
        batch = len(x) if isinstance(x, list) else 1
        return self.model.predict(
            source=x,
            imgsz=self.output_size,
            conf=self.conf,
            agnostic_nms=True,
            verbose=False,
            device="cuda:0",
            batch=batch,
        )


def aggregate(rows, prefix):
    n = len(rows)
    tp = sum(r[f"{prefix}_tp50"] for r in rows)
    fp = sum(r[f"{prefix}_fp50"] for r in rows)
    fn = sum(r[f"{prefix}_fn50"] for r in rows)
    tp_px = sum(r[f"{prefix}_tp_pixels"] for r in rows)
    fp_px = sum(r[f"{prefix}_fp_pixels"] for r in rows)
    fn_px = sum(r[f"{prefix}_fn_pixels"] for r in rows)
    pred_boundary = sum(r[f"{prefix}_boundary_pred_pixels"] for r in rows)
    gt_boundary = sum(r[f"{prefix}_boundary_gt_pixels"] for r in rows)
    matched_pred_boundary = sum(
        r[f"{prefix}_boundary_matched_pred_pixels"] for r in rows
    )
    matched_gt_boundary = sum(r[f"{prefix}_boundary_matched_gt_pixels"] for r in rows)
    boundary_precision = matched_pred_boundary / pred_boundary if pred_boundary else 0.0
    boundary_recall = matched_gt_boundary / gt_boundary if gt_boundary else 0.0
    return {
        "images": n,
        "gt_components": sum(r["gt_components"] for r in rows),
        "pred_components": sum(r[f"{prefix}_pred_components"] for r in rows),
        "tp_components_iou50": tp,
        "fp_components_iou50": fp,
        "fn_components_iou50": fn,
        "instance_precision_iou50": tp / (tp + fp) if tp + fp else 0.0,
        "instance_recall_iou50": tp / (tp + fn) if tp + fn else 0.0,
        "mean_image_ap50": (
            float(np.mean([r[f"{prefix}_ap50"] for r in rows])) if n else 0.0
        ),
        "mean_image_ap50_95": (
            float(np.mean([r[f"{prefix}_ap50_95"] for r in rows])) if n else 0.0
        ),
        "pixel_tp": tp_px,
        "pixel_fp": fp_px,
        "pixel_fn": fn_px,
        "pixel_precision_pooled": tp_px / (tp_px + fp_px) if tp_px + fp_px else 0.0,
        "pixel_recall_pooled": tp_px / (tp_px + fn_px) if tp_px + fn_px else 0.0,
        "pixel_dice_pooled": (
            2 * tp_px / (2 * tp_px + fp_px + fn_px)
            if 2 * tp_px + fp_px + fn_px
            else 1.0
        ),
        "pixel_iou_pooled": (
            tp_px / (tp_px + fp_px + fn_px) if tp_px + fp_px + fn_px else 1.0
        ),
        "boundary_tolerance_pixels": 2.0,
        "boundary_precision_pooled": boundary_precision,
        "boundary_recall_pooled": boundary_recall,
        "boundary_f1_pooled": (
            2
            * boundary_precision
            * boundary_recall
            / (boundary_precision + boundary_recall)
            if boundary_precision + boundary_recall
            else 0.0
        ),
        "mean_image_boundary_f1": (
            float(np.mean([r[f"{prefix}_boundary_f1"] for r in rows])) if n else 0.0
        ),
        "mean_image_iou": (
            float(np.mean([r[f"{prefix}_image_iou"] for r in rows])) if n else 0.0
        ),
        "mean_pred_gt_area_ratio": (
            float(np.mean([r[f"{prefix}_pred_gt_area_ratio"] for r in rows]))
            if n
            else 0.0
        ),
    }


def make_overlay(image, mask, color, alpha=0.45):
    out = image.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.asarray(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_cases(
    case_rows, title, filename, yolo_filter, model, device, size, image_dir, mask_dir
):
    if not case_rows:
        return
    fig, axes = plt.subplots(
        len(case_rows), 4, figsize=(14, max(3.0, 2.9 * len(case_rows))), squeeze=False
    )
    for row_idx, stat in enumerate(case_rows):
        name = stat["filename"]
        with Image.open(image_dir / name) as im:
            image = np.asarray(im.convert("RGB"))
        with Image.open(mask_dir / name) as im:
            gt = np.asarray(im.convert("L")) > 127
        result = yolo_filter(image)
        yolo_mask = yolo_filter.extract_masks(result)[0] > 0
        filtered = distance_based_filtering(image, yolo_mask.astype(np.uint8) * 255)
        filtered = cv2.resize(filtered, (size, size))
        xt = torch.FloatTensor(filtered / 255).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(xt.float())[0, 0].detach().cpu().numpy() > 0.5
        yolo_score = component_scores(yolo_mask, gt)
        unet_score = component_scores(pred, gt)
        tp_mask = pred & gt
        fp_mask = pred & ~gt
        fn_mask = ~pred & gt
        overlay = image.astype(np.float32).copy()
        for m, color in (
            (tp_mask, (0, 220, 0)),
            (fp_mask, (255, 30, 30)),
            (fn_mask, (40, 100, 255)),
        ):
            overlay[m] = 0.50 * overlay[m] + 0.50 * np.asarray(color, dtype=np.float32)
        panels = [
            (image, "Original image"),
            (
                make_overlay(image, yolo_mask, (255, 210, 0)),
                f"YOLO mask: TP {yolo_score['tp50']} / FP {yolo_score['fp50']} / FN {yolo_score['fn50']}",
            ),
            (
                np.clip(overlay, 0, 255).astype(np.uint8),
                f"U-Net: TP {unet_score['tp50']} / FP {unet_score['fp50']} / FN {unet_score['fn50']}",
            ),
            ((gt.astype(np.uint8) * 255), "Ground truth mask"),
        ]
        for col, (panel, panel_title) in enumerate(panels):
            ax = axes[row_idx, col]
            ax.imshow(panel, cmap="gray" if panel.ndim == 2 else None)
            ax.set_title(panel_title, fontsize=8)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(
                    f"{name}\nGT components: {stat['gt_components']}", fontsize=8
                )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    started = time.time()
    if not DATA.exists() or not UNET_CKPT.exists():
        raise FileNotFoundError(
            "Expected existing test data/checkpoints were not found"
        )
    image_dir, mask_dir = DATA / "images", DATA / "masks"
    paths = list(image_dir.glob("*.png"))
    if len(paths) != 1000:
        raise RuntimeError(
            f"Expected the existing 1000-image test split, found {len(paths)}"
        )
    if not SNAPSHOT.exists():
        raise FileNotFoundError("The prepared test snapshot was not found")

    cfg = load_config(str(ROOT))
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this analysis, but PyTorch reports no available CUDA device"
        )
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    print(
        f"Using device: {device} ({gpu_name}); Ultralytics {ultralytics.__version__}; PyTorch {torch.__version__}",
        flush=True,
    )
    # Ultralytics' default time guard can stop batched NMS before later images
    # in a batch are processed. Extend only that guard; outputs were checked
    # against one-image inference before selecting this batched path.
    original_nms = yolo_nms.non_max_suppression

    def nms_with_batch_guard(*args, **kwargs):
        kwargs["max_time_img"] = max(float(kwargs.get("max_time_img", 0.05)), 10.0)
        return original_nms(*args, **kwargs)

    yolo_nms.non_max_suppression = nms_with_batch_guard
    yolo_path = ROOT / cfg.yolo_path_with_augmentation
    yolo = YOLO(str(yolo_path))
    yolo.model.to(device)
    yolo_filter = CudaYoloFilter(yolo, input_size=cfg.size)
    model = get_model(cfg)
    checkpoint = torch.load(UNET_CKPT, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Confirm the existing current test split matches the prepared snapshot.
    snapshot_check = {}
    for sub in ("images", "masks"):
        names_a = {p.name for p in (DATA / sub).glob("*.png")}
        names_b = {p.name for p in (SNAPSHOT / sub).glob("*.png")}
        snapshot_check[sub] = {
            "current_count": len(names_a),
            "snapshot_count": len(names_b),
            "same_names": names_a == names_b,
        }
    if not all(v["same_names"] for v in snapshot_check.values()):
        raise RuntimeError(
            f"Current/snapshot test split file lists differ: {snapshot_check}"
        )

    # Validate prediction and filtering on one test image before the full pass.
    with Image.open(paths[0]) as im:
        sample_image = np.asarray(im.convert("RGB"))
    with Image.open(mask_dir / paths[0].name) as im:
        sample_mask = np.asarray(im.convert("L"))
    result = yolo_filter(sample_image)
    yolo_device = getattr(yolo.predictor, "device", None)
    print(f"YOLO predictor device: {yolo_device}", flush=True)
    if not str(yolo_device).startswith("cuda"):
        raise RuntimeError(f"YOLO inference did not use CUDA: {yolo_device}")
    sample_yolo = yolo_filter.extract_masks(result)[0]
    sample_filtered = distance_based_filtering(sample_image, sample_yolo)
    sample_filtered = cv2.resize(sample_filtered, (cfg.size, cfg.size))
    sample_x = torch.FloatTensor(sample_filtered / 255).permute(2, 0, 1)
    sample_y = torch.IntTensor(sample_mask / 255)
    sample_pred = torch.zeros_like(sample_y).numpy().astype(bool)
    if "model_state_dict" not in checkpoint:
        raise RuntimeError("Unexpected checkpoint schema")
    with torch.no_grad():
        sample_pred = (
            model(sample_x.unsqueeze(0).to(device).float())[0, 0].cpu().numpy() > 0.5
        )
    sample_check = component_scores(sample_pred, sample_y.numpy() > 0)
    reference = compute_instance_segmentation_metrics(
        torch.from_numpy(sample_pred), sample_y
    )
    for local_key, reference_key in (
        ("gt_components", "num_gt_instances"),
        ("pred_components", "num_pred_instances"),
        ("tp50", "tp@50"),
        ("fp50", "fp@50"),
        ("fn50", "fn@50"),
    ):
        if sample_check[local_key] != reference[reference_key]:
            raise RuntimeError(f"Instance metric mismatch on pilot sample: {local_key}")

    pilot_paths = paths[:8]
    pilot_images = []
    for path in pilot_paths:
        with Image.open(path) as im:
            pilot_images.append(np.asarray(im.convert("RGB")))
    pilot_batch_masks = yolo_filter.extract_masks(yolo_filter(pilot_images))
    pilot_single_masks = [
        yolo_filter.extract_masks(yolo_filter(image))[0] for image in pilot_images
    ]
    pilot_differences = [
        float(np.mean((batch_mask > 0) != (single_mask > 0)))
        for batch_mask, single_mask in zip(pilot_batch_masks, pilot_single_masks)
    ]
    pilot_component_equal = [
        ndimage.label(batch_mask > 0)[1] == ndimage.label(single_mask > 0)[1]
        for batch_mask, single_mask in zip(pilot_batch_masks, pilot_single_masks)
    ]

    rows = []
    component_rows = []
    pending = []
    model.eval()

    def flush_pending():
        if not pending:
            return
        batch_paths = [item[0] for item in pending]
        xb = torch.stack([item[1] for item in pending]).to(device).float()
        yb = torch.stack([item[2] for item in pending])
        yolo_masks = [item[3] for item in pending]
        with torch.no_grad():
            outputs = model(xb)
        unet_masks = outputs[:, 0].detach().cpu().numpy() > 0.5

        for i, path in enumerate(batch_paths):
            gt = yb[i].numpy() > 0
            unet = unet_masks[i]
            yolo_bin = yolo_masks[i]
            gt_components = ndimage.label(gt.astype(np.uint8))[1]
            unet_inst = component_scores(unet, gt)
            yolo_inst = component_scores(yolo_bin, gt)
            unet_px = pixel_scores(unet, gt)
            yolo_px = pixel_scores(yolo_bin, gt)
            unet_boundary = boundary_scores(unet, gt)
            yolo_boundary = boundary_scores(yolo_bin, gt)
            row = {"filename": path.name, "gt_components": int(gt_components)}
            for prefix, inst, px, boundary in (
                ("unet", unet_inst, unet_px, unet_boundary),
                ("yolo", yolo_inst, yolo_px, yolo_boundary),
            ):
                row.update(
                    {
                        f"{prefix}_pred_components": inst["pred_components"],
                        f"{prefix}_tp50": inst["tp50"],
                        f"{prefix}_fp50": inst["fp50"],
                        f"{prefix}_fn50": inst["fn50"],
                        f"{prefix}_precision50": inst["precision50"],
                        f"{prefix}_recall50": inst["recall50"],
                        f"{prefix}_ap50": inst["ap50"],
                        f"{prefix}_ap50_95": inst["ap50_95"],
                        **{f"{prefix}_{k}": v for k, v in px.items()},
                        **{f"{prefix}_{k}": v for k, v in boundary.items()},
                    }
                )
                for detail in inst["details"]:
                    component_rows.append(
                        {"filename": path.name, "model": prefix, **detail}
                    )
            row["foreground_load_quartile"] = ""
            rows.append(row)
        pending.clear()

    yolo_batch_size = 16
    for batch_start in range(0, len(paths), yolo_batch_size):
        batch_paths = paths[batch_start : batch_start + yolo_batch_size]
        images, masks = [], []
        for path in batch_paths:
            with Image.open(path) as im:
                images.append(np.asarray(im.convert("RGB")))
            with Image.open(mask_dir / path.name) as im:
                masks.append(np.asarray(im.convert("L")))
        results = yolo_filter(images)
        batch_yolo_masks = yolo_filter.extract_masks(results)
        for path, image, mask, mask_yolo in zip(
            batch_paths, images, masks, batch_yolo_masks
        ):
            yolo_bin = mask_yolo > 0
            filtered = distance_based_filtering(image, yolo_bin.astype(np.uint8) * 255)
            filtered = cv2.resize(filtered, (cfg.size, cfg.size))
            x = torch.FloatTensor(filtered / 255).permute(2, 0, 1)
            y = torch.IntTensor(mask / 255)
            pending.append((path, x, y, yolo_bin))
            if len(pending) >= cfg.batch_size:
                flush_pending()
        flush_pending()
        processed = min(batch_start + len(batch_paths), len(paths))
        print(f"Processed {processed}/{len(paths)} test images", flush=True)

    # Assign equal-sized, explicitly relative test-set load quartiles (not clinical load classes).
    order = np.argsort([r["gt_components"] for r in rows], kind="stable")
    for q, indices in enumerate(np.array_split(order, 4), start=1):
        for idx in indices:
            rows[int(idx)]["foreground_load_quartile"] = f"Q{q}"

    fields = list(rows[0].keys())
    with (OUT / "per_image_metrics.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    detail_fields = [
        "filename",
        "model",
        "kind",
        "component",
        "area_pixels",
        "best_iou_to_gt",
        "x_min",
        "y_min",
        "x_max_exclusive",
        "y_max_exclusive",
    ]
    with (OUT / "unmatched_components.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(component_rows)

    yolo_all = aggregate(rows, "yolo")
    unet_all = aggregate(rows, "unet")
    quartiles = {}
    for q in ("Q1", "Q2", "Q3", "Q4"):
        subset = [r for r in rows if r["foreground_load_quartile"] == q]
        quartiles[q] = {
            "gt_component_min": min(r["gt_components"] for r in subset),
            "gt_component_max": max(r["gt_components"] for r in subset),
            "yolo": aggregate(subset, "yolo"),
            "unet": aggregate(subset, "unet"),
        }

    with SAVED_METRICS.open(newline="") as fh:
        saved = next(csv.DictReader(fh))
    metric_map = {
        "precision(M)": "instance_precision_iou50",
        "recall(M)": "instance_recall_iou50",
        "mAP@50(M)": "mean_image_ap50",
        "mAP@50:95(M)": "mean_image_ap50_95",
        "num_gt_instances": "gt_components",
        "num_pred_instances": "pred_components",
        "mean_image_iou": "mean_image_iou",
    }
    recomputed = unet_all
    comparison = {}
    for saved_key, calc_key in metric_map.items():
        comparison[saved_key] = {
            "saved": float(saved[saved_key]),
            "recomputed": float(recomputed[calc_key]),
            "difference": float(recomputed[calc_key] - float(saved[saved_key])),
        }

    counts = [r["gt_components"] for r in rows]
    summary = {
        "analysis": "Post-hoc Q4 error analysis on the existing fixed DDS3 test split",
        "device": str(device),
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "yolo_checkpoint": str(yolo_path.relative_to(ROOT)),
        "unet_checkpoint": str(UNET_CKPT.relative_to(ROOT)),
        "yolo_confidence_threshold": yolo_filter.conf,
        "unet_probability_threshold": 0.5,
        "yolo_batch_size": 16,
        "nms_time_guard_seconds_per_image": 10.0,
        "batch8_vs_single_image_pilot": {
            "images": len(pilot_paths),
            "max_fraction_of_pixels_different_per_image": max(pilot_differences),
            "component_counts_identical": bool(all(pilot_component_equal)),
            "component_counts_identical_images": int(sum(pilot_component_equal)),
        },
        "boundary_metric": "Inner binary-mask boundary; predicted and ground-truth boundary pixels match within 2 Euclidean pixels.",
        "unet_instance_matching": "4-connected foreground components, greedy IoU matching at 0.50; custom repository metric AP averaged over IoU 0.50:0.95",
        "split": {
            "images": len(rows),
            "current_and_snapshot_file_lists_match": snapshot_check,
            "gt_component_min": int(min(counts)),
            "gt_component_median": float(np.median(counts)),
            "gt_component_max": int(max(counts)),
            "negative_images": int(sum(c == 0 for c in counts)),
            "images_with_10_or_fewer_components": int(sum(c <= 10 for c in counts)),
        },
        "yolo_candidate_masks": yolo_all,
        "yolo_guided_unet_refinement": unet_all,
        "relative_foreground_load_quartiles": quartiles,
        "recomputed_unet_vs_saved_test_metrics": comparison,
        "gpu_name": gpu_name,
        "yolo_was_explicitly_routed_to_cuda": True,
        "pilot_component_metrics_match_repository_function": True,
        "limitations": [
            "The test split contains no negative or sparsely positive images by connected-component count; it cannot validate low-bacterial-load performance.",
            "The masks are binary and have no artifact/debris subtype labels; false-positive regions cannot be attributed to a specific artifact category without expert review.",
            "Connected-component matching merges touching foreground regions; overlapping bacilli cannot be separately assessed from these binary masks alone.",
            "This re-evaluation used the currently installed Ultralytics version and CUDA. Small differences from historical results are possible.",
            "The boundary metrics measure agreement with the binary annotations; they do not isolate the effect of boundary loss because no loss ablation was run.",
        ],
    }
    with (OUT / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    top_fp = sorted(
        rows,
        key=lambda r: (
            -r["unet_fp50"],
            -r["unet_fp_pixels"],
            -r["unet_fn50"],
            r["filename"],
        ),
    )[:6]
    top_fn = sorted(
        rows,
        key=lambda r: (
            -r["unet_fn50"],
            -r["unet_fn_pixels"],
            -r["unet_fp50"],
            r["filename"],
        ),
    )[:6]
    low_iou = sorted(
        rows, key=lambda r: (r["unet_image_iou"], -r["unet_fp50"], r["filename"])
    )[:6]
    render_cases(
        top_fp,
        "Six test images with most unmatched U-Net components",
        "highest_fp_cases.png",
        yolo_filter,
        model,
        device,
        cfg.size,
        image_dir,
        mask_dir,
    )
    render_cases(
        top_fn,
        "Six test images with most missed U-Net components",
        "highest_fn_cases.png",
        yolo_filter,
        model,
        device,
        cfg.size,
        image_dir,
        mask_dir,
    )
    render_cases(
        low_iou,
        "Six test images with lowest U-Net mask IoU",
        "lowest_iou_cases.png",
        yolo_filter,
        model,
        device,
        cfg.size,
        image_dir,
        mask_dir,
    )

    lines = [
        "# Post-hoc test-set error analysis for Reviewer 5 Q4",
        "",
        "## Scope and procedure",
        "",
        f"This analysis used the existing {len(rows):,}-image DDS3 test split and the saved augmented YOLO and U-Net checkpoints. It ran inference only; it did not train models, change split membership, or modify dataset files. The test images and masks matched the prepared snapshot by filename. The analysis used CUDA ({gpu_name}) explicitly for both models, Ultralytics {ultralytics.__version__}, YOLO confidence threshold {yolo_filter.conf}, and U-Net probability threshold 0.5. The repository's YOLO wrapper does not pass a device argument, so this analysis added a CUDA-only wrapper override without changing its confidence, image-size, NMS, or mask-extraction settings.",
        "",
        "For both YOLO candidate masks and refined U-Net masks, binary foregrounds were divided into 4-connected components and greedily matched to ground-truth components at IoU >= 0.50. The repository's simplified per-image AP calculation was also reproduced over IoU thresholds 0.50 to 0.95. Pixel-level false-positive and false-negative counts were computed from the binary masks. Boundary precision, recall, and F1 were also calculated using a 2-pixel Euclidean tolerance.",
        "",
        "## Test-set suitability for the review question",
        "",
        f"The ground-truth masks contain {min(counts)} to {max(counts)} connected components per image (median {np.median(counts):.0f}); {summary['split']['negative_images']} images are negative and {summary['split']['images_with_10_or_fewer_components']} have 10 or fewer components. Thus this test split does not contain negative or sparsely positive examples and cannot establish performance on low-bacterial-load samples.",
        "",
        "## Aggregate results",
        "",
        f"YOLO candidate masks at confidence {yolo_filter.conf}: component precision {yolo_all['instance_precision_iou50']:.4f}, recall {yolo_all['instance_recall_iou50']:.4f}, mean image AP@50 {yolo_all['mean_image_ap50']:.4f}, mean image AP@50:95 {yolo_all['mean_image_ap50_95']:.4f}, unmatched components FP/FN {yolo_all['fp_components_iou50']}/{yolo_all['fn_components_iou50']}, pooled pixel precision/recall {yolo_all['pixel_precision_pooled']:.4f}/{yolo_all['pixel_recall_pooled']:.4f}.",
        f"After U-Net refinement: component precision {unet_all['instance_precision_iou50']:.4f}, recall {unet_all['instance_recall_iou50']:.4f}, mean image AP@50 {unet_all['mean_image_ap50']:.4f}, mean image AP@50:95 {unet_all['mean_image_ap50_95']:.4f}, unmatched components FP/FN {unet_all['fp_components_iou50']}/{unet_all['fn_components_iou50']}, pooled pixel precision/recall {unet_all['pixel_precision_pooled']:.4f}/{unet_all['pixel_recall_pooled']:.4f}.",
        "",
        f"The candidate-mask comparison uses the same test masks and the project's connected-component matching. It is a post-hoc comparison at the refiner's YOLO confidence threshold, not the Ultralytics validation mAP reported in the paper. In an eight-image CUDA pilot, batched and one-image masks differed by at most {max(pilot_differences):.4%} of pixels in any image; connected-component counts matched in {sum(pilot_component_equal)}/{len(pilot_paths)} images. Ultralytics' default NMS time guard was extended for batches because it had otherwise returned empty masks for later images; this guard does not affect completed per-image NMS outputs.",
        "",
        f"At a 2-pixel boundary tolerance, pooled boundary precision/recall/F1 was {yolo_all['boundary_precision_pooled']:.4f}/{yolo_all['boundary_recall_pooled']:.4f}/{yolo_all['boundary_f1_pooled']:.4f} for YOLO and {unet_all['boundary_precision_pooled']:.4f}/{unet_all['boundary_recall_pooled']:.4f}/{unet_all['boundary_f1_pooled']:.4f} after U-Net refinement.",
        "",
        "## Relative load analysis",
        "",
        "The CSV and JSON files report four equal-sized quartiles based only on test-set ground-truth component count. These are relative groups within this unusually narrow positive-load split, not clinical bacterial-load categories.",
        "",
        "## What this can and cannot establish",
        "",
        "The boundary scores assess geometric agreement with the binary annotations but do not show that the boundary loss caused an improvement; no loss ablation was run. The per-image and unmatched-component files quantify where the predicted masks add foreground, miss annotated foreground, or produce poor mask overlap. The image sheets show the largest error cases for visual review. The binary annotations do not label false-positive causes (staining precipitate, debris, scratches, or other structures), so this analysis alone cannot assign those causes. Likewise, connected-component metrics cannot separate bacilli that touch in a binary mask. Expert review with artifact and overlap labels, plus negative and low-load held-out cases, is needed for those claims.",
        "",
        "## Files",
        "",
        "- `per_image_metrics.csv`: per-image pixel and connected-component statistics, including YOLO and U-Net outputs.",
        "- `unmatched_components.csv`: bounding boxes and sizes of unmatched predicted/ground-truth components.",
        "- `summary.json`: run settings, aggregate metrics, load quartiles, and comparison with saved aggregate metrics.",
        "- `highest_fp_cases.png`, `highest_fn_cases.png`, `lowest_iou_cases.png`: visual review sheets.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "output": str(OUT),
                "split": summary["split"],
                "yolo": yolo_all,
                "unet": unet_all,
                "saved_metric_comparison": comparison,
                "elapsed_seconds": round(time.time() - started, 1),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
