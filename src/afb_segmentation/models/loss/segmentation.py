"""Loss functions for binary microscopy segmentation.

The module provides Dice, focal, boundary-weighted, compound, and Dice-BCE
losses. The configured U-Net training path constructs its compound loss with
the boundary term enabled.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


class DiceLoss(nn.Module):
    """Dice Loss for handling severe class imbalance in segmentation.

    Directly optimizes the overlap (IoU-like) between prediction and ground truth,
    treating the entire mask as a single object rather than individual pixels.

    Formula
    -------
        L_dice = 1 - (2 * |P ∩ G| + ε) / (|P| + |G| + ε)

    Benefits for TB detection
    -------------------------
        - Tiny bacilli (~125 pixels) contribute equally to large background
        - Directly related to IoU evaluation metric
        - Prevents model from predicting all-background

    Parameters
    ----------
    smooth : float
        Smoothing factor to avoid division by zero (default: 1.0)
    reduction : str
        Stored compatibility option; ``forward`` currently returns
        a scalar after flattening the full input, regardless of this value.
    """

    def __init__(self, smooth: float = 1.0, reduction: str = "mean"):
        """Store the smoothing value and reduction option."""
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        pred : torch.Tensor
            Predicted probabilities, shape (B, 1, H, W) or (B, H, W)
        target : torch.Tensor
            Ground truth binary mask, same shape as pred

        Returns
        -------
        result : torch.Tensor
            Dice loss value
        """

        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )

        return 1.0 - dice_coeff


class FocalLoss(nn.Module):
    """Focal Loss for focusing on hard-to-classify pixels.

    Down-weights easy pixels (clear background, obvious bacilli) and focuses
    training on hard pixels (staining artifacts, boundaries).

    Formula
    -------
        L_focal = -α * (1 - p_t)^γ * log(p_t)

    where p_t = p if y=1 else (1-p)

    Benefits for TB detection
    -------------------------
        - Forces model to learn artifact vs bacilli distinction
        - Hard boundary pixels get more gradient signal
        - Reduces false positives from staining debris

    Parameters
    ----------
    alpha : float
        Class weight for positive class (default: 0.75 for rare bacilli)
    gamma : float
        Focusing parameter, higher = more focus on hard examples (default: 2.0)
    reduction : str
        'mean', 'sum', or 'none'
    """

    def __init__(
        self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"
    ):
        """Configure positive-class weighting, focusing, and output reduction."""
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        pred : torch.Tensor
            Predicted probabilities, shape (B, 1, H, W) or (B, H, W)
        target : torch.Tensor
            Ground truth binary mask, same shape as pred

        Returns
        -------
        result : torch.Tensor
            Focal loss value
        """

        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)

        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_weight = self.alpha * target + (1 - self.alpha) * (1 - target)

        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)

        focal_loss = alpha_weight * focal_weight * bce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class BoundaryLoss(nn.Module):
    """Boundary Loss for improving mask edge precision.

    Penalizes errors near object boundaries more heavily than interior/exterior
    pixels, improving morphological accuracy for bacillus counting.

    Uses distance transform to weight boundary regions.

    Benefits for TB detection
    -------------------------
        - Accurate bacillus shape for WHO morphology assessment
        - Better separation of touching/overlapping bacilli
        - Prevents mask "bleeding" into background

    Parameters
    ----------
    theta0 : float
        Distance threshold for boundary region (default: 3 pixels)
    theta : float
        Decay factor for distance weighting (default: 5)
    """

    def __init__(self, theta0: float = 3.0, theta: float = 5.0):
        """Set the distance threshold and decay scale for boundary weights."""
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta

    def _compute_boundary_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary weights using distance transform.

        Pixels near boundaries get weight ~1, far pixels get weight ~0.
        """

        target_np = target.detach().cpu().numpy()
        batch_size = target_np.shape[0]

        weights = np.zeros_like(target_np, dtype=np.float32)

        for b in range(batch_size):

            if target_np.ndim == 4:
                mask = target_np[b, 0]
            elif target_np.ndim == 3:
                mask = target_np[b]
            else:
                mask = target_np

            if mask.sum() > 0 and (1 - mask).sum() > 0:
                dist_in = ndimage.distance_transform_edt(mask)
                dist_out = ndimage.distance_transform_edt(1 - mask)
                dist = dist_in + dist_out

                weight = 1.0 / (1.0 + np.exp((dist - self.theta0) / self.theta))
            else:

                weight = np.ones_like(mask) * 0.5

            if target_np.ndim == 4:
                weights[b, 0] = weight
            elif target_np.ndim == 3:
                weights[b] = weight
            else:
                weights = weight

        return torch.from_numpy(weights).to(target.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        pred : torch.Tensor
            Predicted probabilities, shape (B, 1, H, W)
        target : torch.Tensor
            Ground truth binary mask, same shape as pred

        Returns
        -------
        result : torch.Tensor
            Boundary-weighted loss value
        """

        boundary_weights = self._compute_boundary_weights(target)

        loss = torch.abs(pred - target) * boundary_weights

        return loss.mean()


class CompoundLoss(nn.Module):
    """Compound Loss combining Dice, Focal, and Boundary losses.

    The component combination targets the following segmentation challenges.
        - Dice: Handles class imbalance (tiny bacilli vs large background)
        - Focal: Focuses on hard pixels (staining artifacts)
        - Boundary: Improves edge precision (morphology accuracy)

    Formula
    -------
        L_total = λ_dice * L_dice + λ_focal * L_focal + λ_boundary * L_boundary

    Recommended weights for TB microscopy
    -------------------------------------
        - dice_weight: 0.5 (primary driver)
        - focal_weight: 0.3 (artifact suppression)
        - boundary_weight: 0.2 (edge refinement)

    Parameters
    ----------
    dice_weight : float
        Weight for Dice loss component
    focal_weight : float
        Weight for Focal loss component
    boundary_weight : float
        Weight for Boundary loss component
    dice_smooth : float
        Smoothing for Dice loss
    focal_alpha : float
        Class weight for Focal loss
    focal_gamma : float
        Focusing parameter for Focal loss
    use_boundary : bool
        Whether to include boundary loss (slower due to CPU ops)

    Examples
    --------
        >>> loss_fn = CompoundLoss(
        ...     dice_weight=0.5,
        ...     focal_weight=0.3,
        ...     boundary_weight=0.2,
        ... )
        >>> pred = model(images)
        >>> loss = loss_fn(pred, masks)
        >>> loss.backward()
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.3,
        boundary_weight: float = 0.2,
        dice_smooth: float = 1.0,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        use_boundary: bool = True,
    ):
        """Initialize the weighted loss components.

        Parameters
        ----------
        dice_weight : float
            Multiplier for Dice loss.
        focal_weight : float
            Multiplier for focal loss.
        boundary_weight : float
            Multiplier for boundary loss when enabled.
        dice_smooth : float
            Smoothing value passed to ``DiceLoss``.
        focal_alpha : float
            Positive-class weight passed to ``FocalLoss``.
        focal_gamma : float
            Focusing parameter passed to ``FocalLoss``.
        use_boundary : bool
            Whether to construct and include boundary loss.
        """
        super(CompoundLoss, self).__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.use_boundary = use_boundary

        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

        if use_boundary:
            self.boundary_loss = BoundaryLoss()

        self.last_dice = 0.0
        self.last_focal = 0.0
        self.last_boundary = 0.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute compound loss.

        Parameters
        ----------
        pred : torch.Tensor
            Predicted probabilities, shape (B, 1, H, W)
        target : torch.Tensor
            Ground truth binary mask, same shape as pred

        Returns
        -------
        result : torch.Tensor
            Weighted sum of all loss components
        """

        if pred.shape != target.shape:
            target = target.view_as(pred)

        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        self.last_dice = dice.item()
        self.last_focal = focal.item()

        total_loss = self.dice_weight * dice + self.focal_weight * focal

        if self.use_boundary and self.boundary_weight > 0:
            boundary = self.boundary_loss(pred, target)
            self.last_boundary = boundary.item()
            total_loss += self.boundary_weight * boundary

        return total_loss

    def get_component_losses(self) -> dict:
        """Return the most recently computed component losses.

        Returns
        -------
        losses : dict
            Dice, focal, and boundary loss values for logging.
        """
        return {
            "dice": self.last_dice,
            "focal": self.last_focal,
            "boundary": self.last_boundary if self.use_boundary else 0.0,
        }


class DiceBCELoss(nn.Module):
    """Simple Dice + BCE combination (faster alternative to full CompoundLoss).

    A starting point when ``BoundaryLoss`` is too slow or ``FocalLoss`` adds
    unnecessary complexity.

    Parameters
    ----------
    dice_weight : float
        Weight for Dice component (default: 0.5)
    bce_weight : float
        Weight for BCE component (default: 0.5)
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        """Configure the Dice and binary cross-entropy weights."""
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the weighted sum of Dice and binary cross-entropy losses."""
        if pred.shape != target.shape:
            target = target.view_as(pred)

        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)

        return self.dice_weight * dice + self.bce_weight * bce
