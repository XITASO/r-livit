import sys
from typing import Callable, Dict, Optional

import torch
from data.data_interface import ir_transformation_copy
from models.yolo.yolo_base_class import YOLO

sys.stdout = sys.__stdout__


class YOLOv3(YOLO):
    """
    This model uses the YOLOv3 architecture from ultralytics
    """

    def __init__(
        self,
        modality: str = "rgb",
        model_size: str = "M",
        modality_rgbir_use_individual_models: bool = False,
        pre_trained_weights: Optional[str] = None,
        class_mapping_to_coco: Optional[Dict[int, int]] = None,
        ir_transformation: Callable[[torch.Tensor], torch.Tensor] = ir_transformation_copy,
    ):
        """
        Initializes the YOLOv3 model, according to selected modality and model size

        Args:
            modality (str): The type of modality to be used by the model. It can be 'rgb' 'ir', or 'rgbir'.
                If 'rgbir' is selected, the model(s) will be ran for both modalities and a simple nms late fusion will be applied.
            model_size (str): The size of the YOLOv3 model. It can be 'T', 'S', or 'M'.
            modality_rgbir_use_individual_models (bool): Only relevant when modality == 'rgbir': use the identical model instance for rgb and
                ir detection or use individual model instances for each modality.
            pre_trained_weights (Optional[str]): Path to pretrained weights file. If None, by default the pre-trained COCO weights are loaded.
            class_mapping_to_coco (Dict[int, int]): A dictionary that specifies the mapping from the model's output classes to the COCO classes.
                This is only necessary if an own pre-trained model is used. Must be provided if pre-trained weights are used.
            ir_transformation (Callable[[torch.Tensor], torch.Tensor]): A transformation applied to the ir image to be suitable for the 3 channel YOLO model.
                Default is just copying the 1 ir dimension to all 3 channels.
        """

        if pre_trained_weights:
            selected_model_checkpoint = pre_trained_weights
        else:
            if model_size == "T":
                selected_model_checkpoint = "yolov3-tinyu.pt"
            elif model_size == "S":
                selected_model_checkpoint = "yolov3-sppu.pt"
            elif model_size == "M":
                selected_model_checkpoint = "yolov3.pt"
            else:
                raise ValueError("Invalid model size. Choose 'T', 'S', or 'M'")

        super().__init__(
            selected_model_checkpoint=selected_model_checkpoint,
            model_size=model_size,
            modality=modality,
            modality_rgbir_use_individual_models=modality_rgbir_use_individual_models,
            pre_trained_weights=pre_trained_weights,
            class_mapping_to_coco=class_mapping_to_coco,
            ir_transformation=ir_transformation,
        )
