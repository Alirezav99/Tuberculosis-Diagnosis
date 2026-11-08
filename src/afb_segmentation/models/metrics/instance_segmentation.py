"""Connected-component instance metrics for binary segmentation masks.

Predicted and target components are greedily matched by intersection over
union. The reported AP is a precision-based approximation because the masks
do not provide confidence-ranked instances.
"""

import numpy as np
import torch
from scipy import ndimage


def compute_instance_segmentation_metrics(
    pred_mask: torch.Tensor,
    gt_mask: torch.Tensor,
    iou_thresholds: list[float] = None,
) -> dict:
    """Compute instance-level segmentation metrics similar to YOLO's mAP(M).

    Notes
    -----
    1. Extracts connected components (instances) from both pred and GT masks
    2. Matches predicted instances to GT instances using IoU
    3. Computes TP/FP/FN at each IoU threshold
    4. Returns precision, recall, and AP at each threshold

    Parameters
    ----------
    pred_mask : torch.Tensor
        Binary prediction mask (H, W) or (1, H, W)
    gt_mask : torch.Tensor
        Binary ground truth mask (H, W) or (1, H, W)
    iou_thresholds : list[float]
        List of IoU thresholds (default: [0.5, 0.55, ..., 0.95])

    Returns
    -------
    result : dict
        Dictionary with instance-level metrics
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.squeeze().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.squeeze().cpu().numpy()

    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > 0.5).astype(np.uint8)

    pred_labeled, num_pred = ndimage.label(pred_mask)
    gt_labeled, num_gt = ndimage.label(gt_mask)

    if num_gt == 0 and num_pred == 0:

        return {
            "num_gt_instances": 0,
            "num_pred_instances": 0,
            "precision@50": 1.0,
            "recall@50": 1.0,
            "ap@50": 1.0,
            "ap@50:95": 1.0,
        }
    elif num_gt == 0:

        return {
            "num_gt_instances": 0,
            "num_pred_instances": num_pred,
            "precision@50": 0.0,
            "recall@50": 1.0,
            "ap@50": 0.0,
            "ap@50:95": 0.0,
        }
    elif num_pred == 0:

        return {
            "num_gt_instances": num_gt,
            "num_pred_instances": 0,
            "precision@50": 1.0,
            "recall@50": 0.0,
            "ap@50": 0.0,
            "ap@50:95": 0.0,
        }

    iou_matrix = np.zeros((num_pred, num_gt))
    for p_idx in range(1, num_pred + 1):
        pred_instance = pred_labeled == p_idx
        for g_idx in range(1, num_gt + 1):
            gt_instance = gt_labeled == g_idx
            intersection = np.logical_and(pred_instance, gt_instance).sum()
            union = np.logical_or(pred_instance, gt_instance).sum()
            iou_matrix[p_idx - 1, g_idx - 1] = intersection / (union + 1e-6)

    aps = []
    metrics = {
        "num_gt_instances": num_gt,
        "num_pred_instances": num_pred,
    }

    for thresh in iou_thresholds:

        matched_gt = set()
        tp = 0
        fp = 0

        pred_order = np.argsort(-iou_matrix.max(axis=1))

        for p_idx in pred_order:
            best_gt = -1
            best_iou = thresh
            for g_idx in range(num_gt):
                if g_idx not in matched_gt and iou_matrix[p_idx, g_idx] >= best_iou:
                    best_iou = iou_matrix[p_idx, g_idx]
                    best_gt = g_idx

            if best_gt >= 0:
                tp += 1
                matched_gt.add(best_gt)
            else:
                fp += 1

        fn = num_gt - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        ap = precision * recall

        ap = precision if recall > 0 else 0.0

        thresh_key = int(thresh * 100)
        metrics[f"tp@{thresh_key}"] = tp
        metrics[f"fp@{thresh_key}"] = fp
        metrics[f"fn@{thresh_key}"] = fn
        metrics[f"precision@{thresh_key}"] = precision
        metrics[f"recall@{thresh_key}"] = recall
        metrics[f"ap@{thresh_key}"] = ap
        aps.append(ap)

    metrics["ap@50"] = metrics["ap@50"]
    metrics["ap@50:95"] = np.mean(aps)

    return metrics


def compute_batch_instance_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute instance-level metrics for a batch of predictions.

    Parameters
    ----------
    preds : torch.Tensor
        Batch of binary predictions (B, 1, H, W)
    labels : torch.Tensor
        Batch of binary GT masks (B, 1, H, W)
    device : torch.device
        Device for metric tensors

    Returns
    -------
    result : dict
        Aggregated metrics across the batch
    """
    batch_size = preds.shape[0]

    total_tp_50 = 0
    total_fp_50 = 0
    total_fn_50 = 0
    all_aps_50 = []
    all_aps_50_95 = []
    total_gt = 0
    total_pred = 0

    for i in range(batch_size):
        metrics = compute_instance_segmentation_metrics(preds[i], labels[i])
        total_tp_50 += metrics.get("tp@50", 0)
        total_fp_50 += metrics.get("fp@50", 0)
        total_fn_50 += metrics.get("fn@50", 0)
        all_aps_50.append(metrics["ap@50"])
        all_aps_50_95.append(metrics["ap@50:95"])
        total_gt += metrics["num_gt_instances"]
        total_pred += metrics["num_pred_instances"]

    precision_50 = (
        total_tp_50 / (total_tp_50 + total_fp_50)
        if (total_tp_50 + total_fp_50) > 0
        else 0.0
    )
    recall_50 = (
        total_tp_50 / (total_tp_50 + total_fn_50)
        if (total_tp_50 + total_fn_50) > 0
        else 0.0
    )

    return {
        "instance_precision@50": precision_50,
        "instance_recall@50": recall_50,
        "instance_mAP@50": np.mean(all_aps_50) if all_aps_50 else 0.0,
        "instance_mAP@50:95": np.mean(all_aps_50_95) if all_aps_50_95 else 0.0,
        "total_gt_instances": total_gt,
        "total_pred_instances": total_pred,
    }


__all__ = ["compute_batch_instance_metrics", "compute_instance_segmentation_metrics"]
