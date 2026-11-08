"""Wrap Ultralytics YOLO prediction and result extraction."""

import cv2
import numpy as np
import ultralytics


class YoloModel:
    """Wrap a YOLO model with project-sized prediction and mask helpers."""

    def __init__(self, model: ultralytics.YOLO, input_size: int, conf=0.25):
        """Store the model, confidence threshold, and mask output size.

        Parameters
        ----------
        model : ultralytics.YOLO
            Loaded Ultralytics YOLO model.
        input_size : int
            Square size used when resizing extracted masks.
        conf : float, default=0.25
            Confidence threshold passed to ``model.predict``.
        """
        self.model = model
        self.conf = conf
        self.input_size = input_size
        self.output_size = self.input_size + (self.input_size % 32)

    def __call__(self, x: list):
        """Run YOLO prediction on an input image or image collection.

        Parameters
        ----------
        x : list
            Prediction source passed to ``model.predict``.

        Returns
        -------
        list
            Ultralytics prediction results.
        """
        y = self.model.predict(
            source=x,
            imgsz=self.output_size,
            conf=self.conf,
            agnostic_nms=True,
            verbose=False,
        )
        return y

    def extract_boxes(self, predictions):
        """Return each result's XYXY boxes as NumPy arrays.

        Parameters
        ----------
        predictions : iterable
            Ultralytics prediction results.

        Returns
        -------
        list[numpy.ndarray]
            One array of XYXY coordinates per prediction result. Coordinates
            are multiplied by the current width and height scale factors, both
            of which are 1.0.
        """
        results = []
        scale_w = self.input_size / self.input_size
        scale_h = self.input_size / self.input_size
        for result in predictions:
            boxes = result.boxes.xyxy.cpu().numpy()
            boxes[:, [0, 2]] *= scale_w
            boxes[:, [1, 3]] *= scale_h
            results.append(boxes)
        return results

    def extract_masks(self, predictions):
        """Return one binary uint8 mask per result, resized to ``input_size``.

        Parameters
        ----------
        predictions : iterable
            Ultralytics prediction results.

        Returns
        -------
        list[numpy.ndarray]
            One uint8 mask per result. Instance masks are summed and
            thresholded at 0.1; results without mask data produce an all-zero
            mask.
        """
        results = []
        for result in predictions:
            try:
                masks = result.masks.data.cpu().numpy()
                masks = masks.sum(axis=0)

                masks = cv2.resize(
                    masks, [self.input_size] * 2, interpolation=cv2.INTER_NEAREST
                )
                results.append((masks > 0.1).astype(np.uint8) * 255)
            except:
                results.append(
                    np.zeros((self.input_size, self.input_size), dtype=np.uint8)
                )

        return results
