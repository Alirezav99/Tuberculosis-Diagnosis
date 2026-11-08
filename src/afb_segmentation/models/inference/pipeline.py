"""Compose YOLO segmentation with a U-Net mask-refinement stage.

``Pipeline`` provides single-image prediction and test-set evaluation for both
components. Paths are resolved from ``ROOT_DATA`` and ``ROOT_PROJECT`` when the
module is imported. The implementation imports ``MaskRefiner`` from the U-Net
trainer, but that symbol is not defined in this package.
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from afb_segmentation.data_pipeline.load.dataset import CustomDatasetTBDD3
from afb_segmentation.models.unet_trainer import MaskRefiner
from afb_segmentation.utils.image_filtering import filter_image_background
from afb_segmentation.utils.yolo import YoloModel

ROOT_PROJECT = os.getenv("ROOT_PROJECT", ".")
MODEL_PATH = os.path.join(
    ROOT_PROJECT, "artifacts", "experimental_models", "models", "results"
)
RUNS_PATH = os.path.join(ROOT_PROJECT, "artifacts", "runs", "segment")
CONFIG_TRAIN_PATH = os.getenv("CONFIG_TRAIN")
DATA_PATH = os.getenv("ROOT_DATA")


class Pipeline:
    """Compose YOLO segmentation and a UNet mask-refinement model."""

    def __init__(
        self,
        device: str,
        yolo_model: YoloModel,
        mask_refiner: torch.nn.Module,
        threshold_refinement: float = 0.5,
        size: int = 400,
        batch_size: int = 5,
    ):
        """Initialize inference and evaluation settings.

        Parameters
        ----------
        device : str
            Device identifier used by both models.
        yolo_model : YoloModel
            Wrapper providing prediction, mask, and box extraction.
        mask_refiner : torch.nn.Module
            Trained module that refines the filtered image.
        threshold_refinement : float
            Probability threshold used to binarize the
            refiner output.
        size : int
            Square image size passed to background filtering.
        batch_size : int
            Batch size used for test-set evaluation.
        """

        self.size = size
        self.device = device
        self.yolo = yolo_model
        self.batch_size = batch_size
        self.mask_refiner = mask_refiner.to(device)
        self.mask_refiner.eval()
        self.threshold_refinement = threshold_refinement

        self.result_refinement = None
        self.result_segmentation = None

        self.testset_config_path = f"{DATA_PATH}/dds3-yolo/data.yaml"

    def __call__(self, img: np.ndarray | torch.Tensor | Image.Image):
        """Predict and refine a mask for one image.

        Parameters
        ----------
        img : np.ndarray | torch.Tensor | Image.Image
            Three-dimensional image as a NumPy array, CPU tensor, or PIL
            image. Tensor inputs are converted with ``.numpy()``.

        Returns
        -------
        refinement_mask : np.ndarray
            Thresholded output from the mask-refinement model.
        yolo_mask : np.ndarray
            Binary mask returned by YOLO.
        yolo_boxes : np.ndarray
            YOLO boxes cast to ``int16``.
        Raises
        ------
        ValueError
            If the converted image does not have three dimensions.
        """
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        elif isinstance(img, Image.Image):
            img = np.array(img)
        if img.ndim != 3:
            raise ValueError("Image should be 3D array")

        result = self.yolo(img)
        mask = self.yolo.extract_masks(result)[0]
        boxes = self.yolo.extract_boxes(result)[0]
        boxes = boxes.astype(np.int16)

        filtered_img = filter_image_background(
            image=img, mask=mask, yolo_model=self.yolo, size=(self.size, self.size)
        )
        filtered_img = filtered_img.unsqueeze(0).to(self.device)

        refined_mask = self.mask_refiner(filtered_img)
        refined_mask = (refined_mask > self.threshold_refinement).int().cpu()

        return refined_mask[0, 0, ...], mask, boxes

    def _evaluate_yolo(self):
        """Evaluate YOLO on the ``test`` split declared by the dataset YAML.

        Returns
        -------
        results : dict
            Ultralytics' results dictionary.
        """
        results = self.yolo.model.val(
            data=self.testset_config_path,
            batch=self.batch_size,
            single_cls=True,
            split="test",
            device=self.device,
            project=RUNS_PATH,
            verbose=False,
        )

        return results.results_dict

    def _evaluate_mask_refiner(
        self,
    ):
        """Evaluate the mask refiner on the restructured DDS3 test split.

        Returns
        -------
        metrics : dict
            Metrics returned by ``MaskRefiner.eval_one_epoch``.
        """
        dataset = CustomDatasetTBDD3(
            resource_folder=f"{DATA_PATH}/dds3-restructured/test",
            yolo=self.yolo,
            size=(self.size, self.size),
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        refiner = MaskRefiner(refiner_model=self.mask_refiner, device=self.device)
        results = refiner.eval_one_epoch(dataloader)[1]

        return results

    def evaluate_pipeline(self):
        """Evaluate both components and collect their metrics in one mapping.

        Returns
        -------
        metrics : dict
            Metrics from both components and their selected summary values.
        """
        segmentation_results = self._evaluate_yolo()
        refinement_results = self._evaluate_mask_refiner()

        pipeline_results = {
            "segmentation_model": segmentation_results,
            "mask_refiner_model": refinement_results,
            "pipeline_summary": {
                "segmentation_mAP50_masks": segmentation_results.get(
                    "metrics/mAP50(M)", 0
                ),
                "segmentation_precision_masks": segmentation_results.get(
                    "metrics/precision(M)", 0
                ),
                "segmentation_recall_masks": segmentation_results.get(
                    "metrics/recall(M)", 0
                ),
                "refiner_accuracy": refinement_results.get("accuracy", 0),
                "refiner_precision": refinement_results.get("precision", 0),
                "refiner_recall": refinement_results.get("recall", 0),
                "refiner_f1": refinement_results.get("f1", 0),
            },
        }

        return pipeline_results
