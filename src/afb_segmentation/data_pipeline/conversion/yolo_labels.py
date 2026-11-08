"""Convert segmentation masks into YOLO polygon annotations."""

import cv2


def convert_mask_to_polygons(mask_path):
    """Convert a mask to polygon format for YOLO annotations.

    Parameters
    ----------
    mask_path : str
        Path to the mask image file.

    Returns
    -------
    str
        Text representation of the polygons in YOLO format.
    """
    mask = cv2.imread(mask_path, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [cnt.reshape(-1).tolist() for cnt in contours if len(cnt) >= 6]

    h, w = mask.shape
    normalized = [
        [x / w if i % 2 == 0 else x / h for i, x in enumerate(poly)]
        for poly in polygons
    ]

    text = ""
    for object in normalized:
        text += "0 " + " ".join([str(item) for item in object]) + "\n"

    return text
