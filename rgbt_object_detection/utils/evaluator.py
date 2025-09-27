import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torchmetrics.detection import MeanAveragePrecision
from hydra import compose, initialize

from data.data_interface import ImageLabels
from models.model_interface import ImageModelPrediction


with initialize(config_path=os.path.join("..", "conf"), version_base="1.3"):
    cfg_data = compose(config_name="config").data


class Evaluator(nn.Module):
    """
    Class for evaluating object detection results using the Mean Average Precision metric and following COCO protocoll
    This class supports:
    - Filtering predictions based on specified classes.
    - Grouping similar labels under a common class.
    - Computing per-class or overall evaluation metrics.
    """

    def __init__(
        self,
        considered_classes: Optional[List[int]] = None,
        group_similar_labels_mapping: Optional[Dict[str, str]] = None,
        class_metrics: bool = False,
        iou_thresholds: Optional[List[float]] = None,
        rec_thresholds: Optional[List[float]] = None,
        extended_summary: bool = False,
    ):
        """Initialize the Evaluator object

        Args:
            considered_classes (Optional[List[int]]): List of classes as int (COCO class ids) to consider for evaluation.
                If None, all classes will be used.
            group_similar_labels_mapping (Optional[Dict[str, str]]): A dictionary to map similar labels to a common label
                (e.g., {'bus': 'car', 'truck': 'car'}). If None, no mapping is applied.
            class_metrics (bool): Whether to calculate metrics for each class separately.
            iou_thresholds (Optional[List[float]]): List of IoU thresholds to use for evaluation. If None, default thresholds will be used.
            rec_thresholds (Optional[List[float]]): List of recall thresholds to use for evaluation. If None, default thresholds will be used.
            extended_summary (bool): Whether to generate an extended summary of the evaluation metrics.
        """
        super().__init__()
        self.eval()
        self.metric = MeanAveragePrecision(
            iou_type="bbox",
            class_metrics=class_metrics,
            iou_thresholds=iou_thresholds,
            rec_thresholds=rec_thresholds,
            extended_summary=extended_summary,
        )
        self.considered_classes = considered_classes
        self.group_similar_labels_mapping = group_similar_labels_mapping

        with open(cfg_data.labels_path, "r") as file:
            inv_dict = eval(file.read())
            self.coco_labels = {v: k for k, v in inv_dict.items()}

    def to(self, device: torch.device) -> None:
        """Move evaluator to device
        Args:
            device (torch.device): The device where you want to move the model.
        """
        self.metric.to(device)

    def update(
        self,
        predictions: List[ImageModelPrediction],
        ground_truths: Tuple[ImageLabels],
    ) -> None:
        """
        Updates the evaluation with new predictions and ground truths.

        Args:
            predictions (List[ImageModelPrediction]): A list of model predictions in standardized format.
            ground_truths (Tuple[ImageLabels]): A tuple of ground truth annotations in standardized format.
        """

        predictions_eval_format = self._pred_to_eval_format(predictions)
        ground_truths_eval_format = self._gt_to_eval_format(ground_truths)

        predictions_eval_format = self._filter_and_group_predictions(
            predictions_eval_format
        )

        self.metric.update(predictions_eval_format, ground_truths_eval_format)

    def _pred_to_eval_format(self, predictions: List[ImageModelPrediction]) -> List[Dict[str, torch.tensor]]:
        """
        Args:
            predictions (List[ImageModelPrediction]): list of the model's predictions in the standardized format
        Returns:
            predictions (List[Dict[str, torch.tensor]]): List of dictionary containing the predictions in the format:
                        [{"labels": torch.tensor, "boxes": torch.tensor, "scores": torch.tensor}, ...]
        """

        return [
            {
                "labels": p.class_ids,
                "boxes": p.bboxes,
                "scores": p.scores,
            }
            for p in predictions
        ]

    def _gt_to_eval_format(self, ground_truths: Tuple[ImageLabels]) -> List[Dict[str, torch.tensor]]:
        """
        Args:
            ground_truths (Tuple[ImageLabels]): tupel of ground truths in the standardized format
        Returns:
            ground_truths: List of dictionary containing the ground truths in the format:
                        [{"labels": torch.tensor, "boxes": torch.tensor}, ...]
        """
        return [{"labels": gt.class_ids, "boxes": gt.bboxes} for gt in ground_truths]

    def _filter_and_group_predictions(self, predictions: List[Dict[str, torch.tensor]]) -> List[Dict[str, torch.tensor]]:
        """
        Filter and group predictions based on the configuration.
        This method performs two main tasks:
        1. Group similar labels based on the provided mapping.
        2. Filter out predictions that are not in the considered classes list.

        Args:
            predictions (List[Dict[str, torch.tensor]]): List of dictionary containing the predictions
        Returns:
            predictions (List[Dict[str, torch.tensor]]): List of dictionary containing the predictions, with the predictions grouped and filtered.
        """

        if (
            self.group_similar_labels_mapping is not None
            and len(self.group_similar_labels_mapping) > 0
        ):
            for prediction in predictions:
                labels = prediction["labels"]
                for (
                    original_label,
                    mapped_label,
                ) in self.group_similar_labels_mapping.items():
                    original_label_id = self.coco_labels.get(original_label)
                    mapped_label_id = self.coco_labels.get(mapped_label)
                    labels = torch.where(
                        labels == original_label_id, mapped_label_id, labels
                    )
                prediction["labels"] = labels

        if self.considered_classes is not None:
            considered_classes_set = set(self.considered_classes)
            filtered_predictions = []
            for prediction in predictions:
                mask = torch.tensor(
                    [
                        label.item() in considered_classes_set
                        for label in prediction["labels"]
                    ],
                    dtype=torch.bool,
                ).to(prediction["labels"].device)
                filtered_prediction = {
                    "labels": prediction["labels"][mask],
                    "boxes": prediction["boxes"][mask],
                    "scores": prediction["scores"][mask],
                }
                filtered_predictions.append(filtered_prediction)
            predictions = filtered_predictions

        return predictions

    def compute(self) -> Dict[str, Any]:
        """
        Computes and returns the evaluation metrics.

        Returns:
            Dict[str, Any]: A dictionary containing computed evaluation metrics, where values are either floats or lists.
        """

        computing_results = self.metric.compute()
        computing_results = {
            k: v.item() if v.numel() == 1 else v.tolist()
            for k, v in computing_results.items()
        }

        return computing_results
