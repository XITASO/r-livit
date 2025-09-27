import sys
from typing import Callable, Dict, Optional

import torch
from data.data_interface import ir_transformation_copy
from models.yolo.yolo_base_class import YOLO

sys.stdout = sys.__stdout__


class YOLOv5(YOLO):
    """
    This model uses the YOLOv5 architecture from ultralytics
    """

    def __init__(
        self,
        modality: str = "rgb",
        model_size: str = "M",
        modality_rgbir_use_individual_models: bool = False,
        pre_trained_weights: Optional[str] = None,
        class_mapping_to_coco: Optional[Dict[int, int]] = None,
        ir_transformation: Callable[[torch.tensor], torch.tensor] = ir_transformation_copy,
    ):
        """
        Initializes the YOLOv5 model, according to selected modality and model size

        Args:
            modality (str): The type of modality to be used by the model. It can be 'rgb' 'ir', or 'rgbir'.
                If 'rgbir' is selected, the model(s) will be ran for both modalities and a simple nms late fusion will be applied.
            model_size (str): The size of the YOLOv5 model. It can be 'N', 'S', 'M', 'L', or 'X'.
            modality_rgbir_use_individual_models (bool): Only relevant when modality == 'rgbir': use the identical model instance for rgb and
                ir detection or use individual model instances for each modality.
            pre_trained_weights (Optional[str]): Path to pretrained weights file. If None, by default the pre-trained COCO weights are loaded.
            class_mapping_to_coco (Dict[int, int]): A dictionary that specifies the mapping from the model's output classes to the COCO classes.
                This is only necessary if an own pre-trained model is used. Must be provided if pre-trained weights are used.
            ir_transformation (Callable[[torch.tensor], torch.tensor]): A transformation applied to the ir image to be suitable for the 3 channel YOLO model.
                Default is just copying the 1 ir dimension to all 3 channels.
        """

        if pre_trained_weights:
            selected_model_checkpoint = pre_trained_weights
        else:
            if model_size == "N":
                selected_model_checkpoint = "yolov5n.pt"
            elif model_size == "S":
                selected_model_checkpoint = "yolov5s.pt"
            elif model_size == "M":
                selected_model_checkpoint = "yolov5m.pt"
            elif model_size == "L":
                selected_model_checkpoint = "yolov5l.pt"
            elif model_size == "X":
                selected_model_checkpoint = "yolov5x.pt"
            else:
                raise ValueError("Invalid model size. Choose 'N', 'S', 'M', 'L', or 'X'")

        super().__init__(
            selected_model_checkpoint=selected_model_checkpoint,
            model_size=model_size,
            modality=modality,
            modality_rgbir_use_individual_models=modality_rgbir_use_individual_models,
            pre_trained_weights=pre_trained_weights,
            class_mapping_to_coco=class_mapping_to_coco,
            ir_transformation=ir_transformation,
        )
