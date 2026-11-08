"""PyTorch train and evaluation epoch loops for segmentation models.

The loops calculate pixel-level metrics and connected-component
instance-level metrics while preserving the training and evaluation behavior
used by the existing workflows.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, MeanMetric, Precision, Recall
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm

from ..loss.segmentation import CompoundLoss
from ..metrics.instance_segmentation import (
    compute_batch_instance_metrics,
    compute_instance_segmentation_metrics,
)


def train_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device | str,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    label_smoothing: float = 0.0,
) -> tuple[float, dict]:
    """Train for one epoch and collect pixel- and instance-level metrics.

    Parameters
    ----------
    model : nn.Module
        Segmentation model to train.
    loss_fn : nn.Module
        Loss function applied to model predictions and smoothed labels.
    device : torch.device | str
        Device used for inputs, labels, model, and metric accumulators.
    train_loader : DataLoader
        Training batches in the order provided by the loader.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    label_smoothing : float
        Smoothing amount applied to labels for loss calculation. Metrics use
        the original labels.

    Returns
    -------
    metrics : dict
        Mean loss, pixel-level scores, instance-level scores, and legacy score
        keys used by the existing training workflow.
    """

    model.train()
    model.to(device)

    loss_metric = MeanMetric().to(device)
    accuracy = Accuracy(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)
    f1 = F1Score(task="binary").to(device)
    iou = BinaryJaccardIndex().to(device)

    loss_metric.reset()
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    iou.reset()

    total_tp_50 = 0
    total_fp_50 = 0
    total_fn_50 = 0
    all_instance_aps_50 = []
    all_instance_aps_50_95 = []
    total_gt_instances = 0
    total_pred_instances = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device).float()
        labels = labels.to(device).float().unsqueeze(1)

        if label_smoothing > 0:
            labels_smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
        else:
            labels_smooth = labels

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels_smooth)
        loss.backward()
        optimizer.step()

        preds = (outputs > 0.5).float()

        loss_metric.update(loss)
        accuracy.update(preds, labels)
        precision.update(preds, labels)
        recall.update(preds, labels)
        f1.update(preds, labels)
        iou.update(preds.int(), labels.int())

        preds_int = preds.int()
        labels_int = labels.int()
        for i in range(preds_int.shape[0]):
            instance_metrics = compute_instance_segmentation_metrics(
                preds_int[i], labels_int[i]
            )
            total_tp_50 += instance_metrics.get("tp@50", 0)
            total_fp_50 += instance_metrics.get("fp@50", 0)
            total_fn_50 += instance_metrics.get("fn@50", 0)
            all_instance_aps_50.append(instance_metrics["ap@50"])
            all_instance_aps_50_95.append(instance_metrics["ap@50:95"])
            total_gt_instances += instance_metrics["num_gt_instances"]
            total_pred_instances += instance_metrics["num_pred_instances"]

        pbar.set_postfix(
            {
                "loss": loss_metric.compute().item(),
                "acc": accuracy.compute().item(),
                "iou": iou.compute().item(),
                "f1": f1.compute().item(),
            }
        )

    instance_precision_50 = (
        total_tp_50 / (total_tp_50 + total_fp_50)
        if (total_tp_50 + total_fp_50) > 0
        else 0.0
    )
    instance_recall_50 = (
        total_tp_50 / (total_tp_50 + total_fn_50)
        if (total_tp_50 + total_fn_50) > 0
        else 0.0
    )
    instance_map_50 = np.mean(all_instance_aps_50) if all_instance_aps_50 else 0.0
    instance_map_50_95 = (
        np.mean(all_instance_aps_50_95) if all_instance_aps_50_95 else 0.0
    )

    metrics = {
        "loss": loss_metric.compute().item(),
        "accuracy": accuracy.compute().item(),
        "precision(pixel)": precision.compute().item(),
        "recall(pixel)": recall.compute().item(),
        "f1(pixel)": f1.compute().item(),
        "iou(pixel)": iou.compute().item(),
        "precision(M)": instance_precision_50,
        "recall(M)": instance_recall_50,
        "mAP@50(M)": instance_map_50,
        "mAP@50:95(M)": instance_map_50_95,
        "precision": precision.compute().item(),
        "recall": recall.compute().item(),
        "f1": f1.compute().item(),
        "iou": iou.compute().item(),
        "mAP@50": instance_map_50,
        "mAP@50:95": instance_map_50_95,
    }

    return metrics


def eval_one_epoch(
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device | str,
    eval_loader: DataLoader,
) -> tuple[float, dict]:
    """Evaluate for one epoch and return loss and metrics.

    Parameters
    ----------
    model : nn.Module
        Segmentation model to evaluate.
    loss_fn : nn.Module
        Loss function applied to model predictions and labels.
    device : torch.device | str
        Device used for inputs, labels, model, and metric accumulators.
    eval_loader : DataLoader
        Evaluation batches in the order provided by the loader.

    Returns
    -------
    metrics : dict
        Mean loss, pixel-level scores, instance-level scores, and legacy score
        keys used by the existing evaluation workflow.

    Notes
    -----
    Instance metrics use connected-component analysis to identify individual
    instances and match predictions to targets by intersection over union.
    """

    model.eval()
    model.to(device)

    loss_metric = MeanMetric().to(device)
    accuracy = Accuracy(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)
    f1 = F1Score(task="binary").to(device)
    iou = BinaryJaccardIndex().to(device)

    loss_metric.reset()
    accuracy.reset()
    precision.reset()
    recall.reset()
    f1.reset()
    iou.reset()

    total_tp_50 = 0
    total_fp_50 = 0
    total_fn_50 = 0
    all_instance_aps_50 = []
    all_instance_aps_50_95 = []
    total_gt_instances = 0
    total_pred_instances = 0

    image_level_ious = []

    with torch.no_grad():
        pbar = tqdm(eval_loader, desc="Evaluation")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device).float()
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)

            preds = (outputs > 0.5).float()

            loss_metric.update(loss)
            accuracy.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)
            iou.update(preds.int(), labels.int())

            preds_int = preds.int()
            labels_int = labels.int()

            for i in range(preds_int.shape[0]):

                instance_metrics = compute_instance_segmentation_metrics(
                    preds_int[i], labels_int[i]
                )

                total_tp_50 += instance_metrics.get("tp@50", 0)
                total_fp_50 += instance_metrics.get("fp@50", 0)
                total_fn_50 += instance_metrics.get("fn@50", 0)
                all_instance_aps_50.append(instance_metrics["ap@50"])
                all_instance_aps_50_95.append(instance_metrics["ap@50:95"])
                total_gt_instances += instance_metrics["num_gt_instances"]
                total_pred_instances += instance_metrics["num_pred_instances"]

                pred_mask = preds_int[i].flatten()
                gt_mask = labels_int[i].flatten()
                intersection = (pred_mask & gt_mask).sum().float()
                union = (pred_mask | gt_mask).sum().float()
                sample_iou = (intersection / (union + 1e-6)).item()
                image_level_ious.append(sample_iou)

            pbar.set_postfix(
                {
                    "loss": loss_metric.compute().item(),
                    "acc": accuracy.compute().item(),
                    "iou": iou.compute().item(),
                    "f1": f1.compute().item(),
                }
            )

    loss_value = loss_metric.compute().item()

    instance_precision_50 = (
        total_tp_50 / (total_tp_50 + total_fp_50)
        if (total_tp_50 + total_fp_50) > 0
        else 0.0
    )
    instance_recall_50 = (
        total_tp_50 / (total_tp_50 + total_fn_50)
        if (total_tp_50 + total_fn_50) > 0
        else 0.0
    )
    instance_map_50 = np.mean(all_instance_aps_50) if all_instance_aps_50 else 0.0
    instance_map_50_95 = (
        np.mean(all_instance_aps_50_95) if all_instance_aps_50_95 else 0.0
    )

    mean_image_iou = np.mean(image_level_ious) if image_level_ious else 0.0

    frac_images_above_50 = (
        np.mean([1.0 if iou >= 0.5 else 0.0 for iou in image_level_ious])
        if image_level_ious
        else 0.0
    )

    metrics = {
        "loss": loss_value,
        "accuracy": accuracy.compute().item(),
        "precision(pixel)": precision.compute().item(),
        "recall(pixel)": recall.compute().item(),
        "f1(pixel)": f1.compute().item(),
        "iou(pixel)": iou.compute().item(),
        "precision(M)": instance_precision_50,
        "recall(M)": instance_recall_50,
        "mAP@50(M)": instance_map_50,
        "mAP@50:95(M)": instance_map_50_95,
        "num_gt_instances": total_gt_instances,
        "num_pred_instances": total_pred_instances,
        "mean_image_iou": mean_image_iou,
        "frac_images_iou>=0.5": frac_images_above_50,
        "mAP@50": instance_map_50,
        "mAP@50:95": instance_map_50_95,
    }
    return metrics


__all__ = ["eval_one_epoch", "train_one_epoch"]
