import os
import sys
from typing import Callable, Dict, List, Optional, Tuple, Type

import torch
from torchvision.ops import batched_nms

import models.model_interface as model_interface
from data.data_interface import box_cxcywh_to_xyxy, ir_transformation_copy
from models.model_interface import ImageModelPrediction
from ultralytics import RTDETR as RTDETRUL
from utils.finetuning_utils.finetuning_rtdetr import finetune_rtdetr

sys.stdout = sys.__stdout__

class RTDETR(model_interface.Model):
    """
    This model uses the RT-DETR architecture from ultralytics
    """

    def __init__(
        self,
        model_size: str = "L",
        modality: str = "rgb",
        modality_rgbir_use_individual_models: bool = False,
        pre_trained_weights: Optional[str] = None,
        class_mapping_to_coco: Optional[Dict[int, int]] = None,
        ir_transformation: Callable[[torch.Tensor], torch.Tensor] = ir_transformation_copy,
    ):
        """
        Initializes the RT-DETR model, according to selected modality and model size

        Args:
            model_size (str): The size of the RT-DETR model. It can be 'L' or 'X'.
            modality (str): The type of modality to be used by the model. It can be 'rgb', 'ir', or 'rgbir'.
            modality_rgbir_use_individual_models (bool): If True, uses separate models for RGB and IR.
            pre_trained_weights (Optional[str]): Path to pretrained weights file. Defaults to COCO pre-trained weights.
            class_mapping_to_coco (Optional[Dict[int, int]]): Mapping from model class IDs to COCO classes.
            ir_transformation (Callable[[torch.Tensor], torch.Tensor]): Transformation applied to IR images for RT-DETR.
        """

        super().__init__(
            modality=modality,
            ir_transformation=ir_transformation,
            pre_trained_weights=pre_trained_weights,
            modality_rgbir_use_individual_models=modality_rgbir_use_individual_models,
            class_mapping_to_coco=class_mapping_to_coco,
        )

        if self.pre_trained_weights:
            selected_model_checkpoint = self.pre_trained_weights
        elif model_size == "L":
            selected_model_checkpoint = "rtdetr-l.pt"
        elif model_size == "X":
            selected_model_checkpoint = "rtdetr-x.pt"
        else:
            raise ValueError("Invalid model size. Choose 'L', or 'X'")
        
        if self.modality == "rgbir" and modality_rgbir_use_individual_models:
            self.model_rgb = RTDETRUL(selected_model_checkpoint[0] if self.pre_trained_weights else selected_model_checkpoint).model
            self.model_ir = RTDETRUL(selected_model_checkpoint[1] if self.pre_trained_weights else selected_model_checkpoint).model
        else:
            self.model = RTDETRUL(selected_model_checkpoint).model

        self.model_size = model_size
        self.min_conf = 0.25 # Hard coded minimum confidence, introduce parameter for this if wanted.
        self.nms_iou_threshold = 0.8 # Hard coded NMS threshold, introduce parameter for this if wanted.


    def eval(self):
        """
        Sets the model to evaluation mode
        """
        if self.modality == "rgbir" and self.modality_rgbir_use_individual_models:
            self.model_rgb.eval()
            self.model_ir.eval()
        else:
            self.model.eval()

    def to(self, device: torch.device):
        """Move model to device
        Args:
            device (torch.device): The device where you want to move the model.
        """
        if self.modality == "rgbir" and self.modality_rgbir_use_individual_models:
            self.model_rgb.to(device)
            self.model_ir.to(device)
        else:
            self.model.to(device)


    def pre_process(self, standardized_input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Converts the standardized input data to the format required for the RT-DETR model.

        Args:
            standardized_input (Tuple[torch.Tensor, torch.Tensor]): Input data in the standardized format.

        Returns:
            model_input (torch.Tensor): Takes only the rgb part of the standardized input
        """

        rgb_batch, ir_batch = standardized_input

        if self.modality == "rgb":
            return rgb_batch
        elif self.modality == "ir":
            return self.ir_transformation(ir_batch)
        elif self.modality == "rgbir":
            return (rgb_batch, self.ir_transformation(ir_batch))
        else:
            raise ValueError(
                "Fatal error: Modality must be either 'rgb', 'ir' or 'rgbir'"
            )

    def post_process(self, model_output: Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]) -> List[ImageModelPrediction]:
        """
        Converts the output data of the RT-DETR model to the standardized format.

        Args:
            model_output (Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]):
                Output data in the raw format of the RT-DETR model.

        Returns:
            standardized_output (List[ImageModelPrediction]): Processed output data in the standardized format.
        """

        batch_scores, batch_bboxes, batch_class_ids = model_output

        standardized_output = [
            ImageModelPrediction(
                class_ids=batch_class_ids[i].int() + 1
                if not self.pre_trained_weights
                else torch.Tensor([self.class_mapping_to_coco[i.item()+1] for i in batch_class_ids[i].int()]),
                bboxes=box_cxcywh_to_xyxy(batch_bboxes[i]),
                scores=batch_scores[i],
            )
            for i in range(len(batch_scores))
        ]

        return standardized_output

    def predict(self, model_input: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Predicts the output data for the given input data using the RT-DETR model.

        Args:
            model_input (torch.Tensor): with the size (bs, 3, img_height, img_width)

        Returns:
            model_output (Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]):
                Output data in the raw format of the RT-DETR model.
        """
        # Note: For rgbir, proper inference time evaluation is not reasonable with current implementation, since NMS is done multiple times!
        if self.modality == "rgbir":
            rgb_batch, ir_batch = model_input

            if self.modality_rgbir_use_individual_models:
                # For a correct evaluation of the inference time, the following two lines would have to be parallelized.
                model_output_rgb_unfiltered = self.model_rgb(rgb_batch)
                model_output_ir_unfiltered = self.model_ir(ir_batch)
                model_output_unfiltered = torch.cat((model_output_rgb_unfiltered[0], model_output_ir_unfiltered[0]), dim=-2)

            else:
                concatenated_model_input = torch.cat([rgb_batch, ir_batch], dim=0)
                concatenated_model_output = self.model(concatenated_model_input)
                bs = rgb_batch.size(0)
                rgb_outputs, ir_outputs = torch.split(
                    concatenated_model_output[0], split_size_or_sections=bs, dim=0
                )
                model_output_unfiltered = (torch.cat([rgb_outputs, ir_outputs], dim=-2))

            model_output = model_output_unfiltered
        else:
            model_output = self.model(model_input)[0]


        img_height, img_width = model_input.shape[-2:] if not self.modality == "rgbir" else model_input[0].shape[-2:]
        batch_size = model_input.shape[0] if not self.modality == "rgbir" else model_input[0].shape[0]
        nd = model_output.shape[-1]
        bboxes, logits = model_output.split((4, nd - 4), dim=-1)

        max_logits, class_ids = torch.max(logits, dim=2)

        conf_mask = max_logits >= self.min_conf

        batch_scores = []
        batch_class_ids = []
        batch_bboxes = []

        for i in range(batch_size):
            batch_mask = conf_mask[i]
            scores_i = max_logits[i][batch_mask]
            class_ids_i = class_ids[i][batch_mask]
            bboxes_i = bboxes[i][batch_mask]

            bboxes_i[..., [0, 2]] *= img_width
            bboxes_i[..., [1, 3]] *= img_height

            if self.modality == "rgbir":
                keep = batched_nms(box_cxcywh_to_xyxy(bboxes_i), scores_i, class_ids_i, iou_threshold=self.nms_iou_threshold)

                batch_scores.append(scores_i[keep])
                batch_class_ids.append(class_ids_i[keep])
                batch_bboxes.append(bboxes_i[keep])
            else:
                batch_scores.append(scores_i)
                batch_class_ids.append(class_ids_i)
                batch_bboxes.append(bboxes_i)

        return (batch_scores, batch_bboxes, batch_class_ids)
    
    def finetune(
        self,
        dataset_name: str,
        dataset_class: Type[torch.utils.data.Dataset],
        splits: Tuple[str, str],
        seed: int,
        run_name: str
    ) -> None:
        """
        Finetunes the RT-DETR model on the provided dataset.

        Args:
            dataset_name (str): The name of the dataset.
            dataset_class (Type[torch.utils.data.Dataset]): The dataset class to be used for finetuning.
            splits (Tuple[str, str]): The dataset splits for training and evaluation.
            seed (int): Random seed for reproducibility.
            run_name (str): Experiment name for logging and saving.
        """
        datasets = {}
        datasets["train"] = dataset_class(split=splits[0], original_class_ids=True, resize_for_yolo=True)
        datasets["val"] = dataset_class(split=splits[1], original_class_ids=True, resize_for_yolo=True)
        # So far no extra Validation set is used as there is no hyperparameter optimization or early stopping done. If one wants to do this,
        # an extra validation set needs to be defined and passed instead of the current test dataset!

        finetune_rtdetr(self.model, datasets, dataset_name, self.modality, self.model_size, seed, run_name)
