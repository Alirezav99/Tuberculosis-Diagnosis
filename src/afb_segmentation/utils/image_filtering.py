"""Filter RGB images with YOLO masks and distance-based weighting."""

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from . import yolo as model


def create_distance_matrix(mask: np.ndarray, max_distance: int):
    """Create a squared foreground-centered brightness weight map.

    Parameters
    ----------
    mask : numpy.ndarray
        Array thresholded at zero to identify foreground pixels.
    max_distance : int
        Distance scale used to clip the background falloff.

    Returns
    -------
    numpy.ndarray
        Weight array with foreground values of one and values that decrease
        with distance from foreground.
    """
    binary_mask = (mask > 0).astype(np.uint8)
    distance_map = distance_transform_edt(1 - binary_mask)
    normalized_distance = 1 - np.clip((distance_map * 2) / max_distance, 0, 1)
    return normalized_distance**2


def distance_based_filtering(img: np.ndarray, msk: np.ndarray):
    """Darken RGB image pixels according to their distance from a mask.

    Parameters
    ----------
    img : numpy.ndarray
        RGB image array.
    msk : numpy.ndarray
        Foreground mask used to compute distance weights.

    Returns
    -------
    numpy.ndarray
        Filtered RGB image. The fixed distance scale of 100 is applied to the
        HSV value channel before converting the result back to RGB.
    """
    distance_matrix = create_distance_matrix(msk, max_distance=100)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_img[..., 2] = hsv_img[..., 2] * (distance_matrix)
    filtered_img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return filtered_img


def filter_image_background(
    image: np.ndarray, mask: np.ndarray, yolo_model, size
) -> torch.Tensor:
    """Filter an RGB image using a YOLO-predicted foreground mask.

    Parameters
    ----------
    image : numpy.ndarray
        Input RGB image.
    mask : numpy.ndarray
        Ground-truth mask argument. The current implementation uses the mask
        predicted by ``yolo_model`` for filtering instead.
    yolo_model
        Ultralytics model passed to ``YoloModel``.
    size
        Optional square output size. When truthy, the filtered image and local
        mask are resized to this size.

    Returns
    -------
    torch.Tensor
        Filtered image in channel-first ``(C, H, W)`` layout. The mask is not
        returned.
    """
    yolo_model = model.YoloModel(yolo_model, input_size=size, conf=0.25)
    result = yolo_model(image)
    pred_mask = yolo_model.extract_masks(result)
    filtered_img = distance_based_filtering(img=image, msk=pred_mask[0])

    if size:
        filtered_img = cv2.resize(filtered_img, (size, size))
        mask = cv2.resize(mask, (size, size))

    filtered_img = torch.FloatTensor(filtered_img / 255).permute(2, 0, 1)

    return filtered_img
