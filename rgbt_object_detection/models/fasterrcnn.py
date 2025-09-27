import os
import sys
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from hydra import compose, initialize
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms

import models.model_interface as model_interface
from models.model_interface import ImageModelPrediction
from data.data_interface import (
    convert_coco_labels_from_old_to_new,
    get_default_train_augmentation,
    ir_transformation_copy,
)
from utils.finetuning_utils.finetuning_faster_rcnn import finetune_faster_rcnn

sys.stdout = sys.__stdout__

with initialize(config_path=os.path.join("..", "conf"), version_base="1.3"):
    cfg = compose(config_name="config")
    setup_cfg = cfg.setup


class FasterRCNN(model_interface.Model):
    """
    This model uses the Faster R-CNN architecture from torchvision.
    """

    def __init__(
        self,
        model_size: str = "-",
        modality: str = "rgb",
        modality_rgbir_use_individual_models: bool = False,
        pre_trained_weights: Optional[str] = None,
        class_mapping_to_coco: Optional[Dict[int, int]] = None,
        ir_transformation: Callable[[torch.Tensor], torch.Tensor] = ir_transformation_copy,
    ):
        """
        Initializes the Faster R-CNN model, according to selected modality and model size

        Args:
            model_size (str): Ignored here, as Faster R-CNN  is one-sized in our implementation.
            modality (str): The type of modality to be used by the model. It can be 'rgb' 'ir', or 'rgbir'.
                If 'rgbir' is selected, the model(s) will be ran for both modalities and a simple nms late fusion will be applied.
            modality_rgbir_use_individual_models (bool): Only relevant when modality == 'rgbir': use the identical model instance for rgb and ir
                detection or use individual model instances for each modality.
            pre_trained_weights (Optional[str]): Path to pretrained weights file. If None, by default the pre-trained COCO weights are loaded.
            class_mapping_to_coco (Dict[int, int]): A dictionary that specifies the mapping from the model's output classes to the COCO classes.
                This is only necessary if an own pre-trained model is used. Must be provided if pre-trained weights are used.
            ir_transformation (Callable[[torch.Tensor], torch.Tensor]): A transformation applied to the ir image to be suitable for the 3 channel
                Faster R-CNN model. Default is just copying the 1 ir dimension to all 3 channels.
        """

        super().__init__(
            modality=modality,
            ir_transformation=ir_transformation,
            pre_trained_weights=pre_trained_weights,
            modality_rgbir_use_individual_models=modality_rgbir_use_individual_models,
            class_mapping_to_coco=class_mapping_to_coco,
        )

        if self.modality == "rgbir" and self.modality_rgbir_use_individual_models:
            self.model_rgb = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
            self.model_ir = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

        if self.class_mapping_to_coco:
            self.class_mapping_to_coco[0] = 0  # The background class is added by default for FasterRCNN

        if self.pre_trained_weights:
            self._load_pretrained_weights_from_finetuning()
        
        self.nms_iou_threshold = 0.8 # Hard coded NMS threshold, introduce parameter for this if wanted.

    def _load_pretrained_weights_from_finetuning(self) -> None:
        """
        Loads the pre-trained weights into the model. In doing so, a new prediction head is added with the correct number of classes.

        """
        num_classes = (
            len(self.class_mapping_to_coco)
            if self.class_mapping_to_coco
            else self.model.roi_heads.box_predictor.cls_score.out_features
        )

        if self.modality == "rgbir" and self.modality_rgbir_use_individual_models:
            in_features = self.model_rgb.roi_heads.box_predictor.cls_score.in_features
            self.model_rgb.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )
            state_dict_rgb = torch.load(
                self.pre_trained_weights[0], map_location=torch.device(setup_cfg.device)
            )
            self.model_rgb.load_state_dict(state_dict_rgb)

            self.model_ir.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )
            state_dict_ir = torch.load(
                self.pre_trained_weights[1], map_location=torch.device(setup_cfg.device)
            )
            self.model_ir.load_state_dict(state_dict_ir)

        else:
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, num_classes
            )
            state_dict = torch.load(
                self.pre_trained_weights, map_location=torch.device(setup_cfg.device)
            )
            self.model.load_state_dict(state_dict)

    def eval(self):
        """
        Sets the model to evaluation mode
        """
        if self.modality == "rgbir" and self.modality_rgbir_use_individual_models:
            self.model_rgb.eval()
            self.model_ir.eval()
        else:
            self.model.eval()

    def to(self, device: torch.device) -> None:
        """Move model to device
        Args:
            device (torch.device): The device where you want to move the model.
        """
        if self.modality == "rgbir" and self.modality_rgbir_use_individual_models:
            self.model_rgb.to(device)
            self.model_ir.to(device)
        else:
            self.model.to(device)

    def pre_process(self, standardized_input: Tuple[torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the standardized input data to the format required for the Faster R-CNN specific model.

        Args:
            standardized_input (Tuple[torch.Tensor, torch.Tensor]): Input data in the standardized format.

        Returns:
            model_input (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): Either the rgb image, the ir image,
            or both depending on the selected modality.
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

    def post_process(self, model_output: List[Dict[str, torch.Tensor]]) -> List[ImageModelPrediction]:
        """
        Converts the output data of the Faster R-CNN specific model to the standardized format.

        Args:
            model_output (List[Dict[str, torch.Tensor]]): Output data in the raw format of the Faster R-CNN specific model.
                It's a list of length batch_size of dictionaries with the keys 'labels', 'boxes', and 'scores'.

        Returns:
            standardized_output (List[ImageModelPrediction]): Processed output data in the standardized format.
        """

        standardized_output = [
            ImageModelPrediction(
                class_ids=convert_coco_labels_from_old_to_new(pred["labels"])
                if self.pre_trained_weights is None
                else pred["labels"],
                bboxes=pred["boxes"],
                scores=pred["scores"],
            )
            for pred in model_output
        ]

        return standardized_output

    def predict(self, model_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts the output data for the given input data using the Faster R-CNN specific model.

        Args:
            model_input (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): Either the rgb image, the ir image,
                or both depending on the selected modality.

        Returns:
            List[Dict[str, torch.Tensor]]: A list (batch_size) of dictionaries containing:
                - "labels": Class labels for detected objects.
                - "boxes": Bounding box coordinates in (x_min, y_min, x_max, y_max) format.
                - "scores": Confidence scores for each detection.
        """
        # Note: For rgbir, proper inference time evaluation is not reasonable with current implementation, since NMS is done multiple times!
        if self.modality == "rgbir":
            rgb_batch, ir_batch = model_input
            bs = rgb_batch.size(0)

            if self.modality_rgbir_use_individual_models:
                # For a correct evaluation of the inference time, the following two lines would have to be parallelized.
                model_output_rgb = self.model_rgb(rgb_batch)
                model_output_ir = self.model_ir(ir_batch)

                model_output_combined = [
                    {
                        "labels": torch.cat(
                            [
                                model_output_rgb[i]["labels"],
                                model_output_ir[i]["labels"],
                            ],
                            dim=0,
                        ),
                        "boxes": torch.cat(
                            [model_output_rgb[i]["boxes"], model_output_ir[i]["boxes"]],
                            dim=0,
                        ),
                        "scores": torch.cat(
                            [
                                model_output_rgb[i]["scores"],
                                model_output_ir[i]["scores"],
                            ],
                            dim=0,
                        ),
                    }
                    for i in range(bs)
                ]

            else:
                concatenated_model_input = torch.cat([rgb_batch, ir_batch], dim=0)
                concatenated_model_output = self.model(concatenated_model_input)

                model_output_combined = [
                    {
                        "labels": torch.cat(
                            [
                                concatenated_model_output[i]["labels"],
                                concatenated_model_output[i + bs]["labels"],
                            ],
                            dim=0,
                        ),
                        "boxes": torch.cat(
                            [
                                concatenated_model_output[i]["boxes"],
                                concatenated_model_output[i + bs]["boxes"],
                            ],
                            dim=0,
                        ),
                        "scores": torch.cat(
                            [
                                concatenated_model_output[i]["scores"],
                                concatenated_model_output[i + bs]["scores"],
                            ],
                            dim=0,
                        ),
                    }
                    for i in range(bs)
                ]

            nms_selection = [
                nms(
                    boxes=model_output_combined[i]["boxes"],
                    scores=model_output_combined[i]["scores"],
                    iou_threshold=self.nms_iou_threshold,
                )
                for i in range(bs)
            ]
            model_output = [
                {
                    "labels": model_output_combined[i]["labels"][nms_selection[i]],
                    "boxes": model_output_combined[i]["boxes"][nms_selection[i]],
                    "scores": model_output_combined[i]["scores"][nms_selection[i]],
                }
                for i in range(bs)
            ]

        else:
            model_output = self.model(model_input)

        if self.class_mapping_to_coco:
            for i in range(len(model_output)):
                original_labels = model_output[i]["labels"].clone()
                for original_id, coco_id in self.class_mapping_to_coco.items():
                    model_output[i]["labels"][original_labels == original_id] = coco_id

        return model_output

    def finetune(
        self,
        dataset_name: str,
        dataset_class: Type[torch.utils.data.Dataset],
        splits: Tuple[str, str],
        seed: int,
        run_name: str
    ) -> None:
        """
        Finetunes the Faster R-CNN model on the provided dataset.

        Args:
            dataset_name (str): The name of the dataset
            dataset_class (Type[torch.utils.data.Dataset]): The dataset class to be used for finetuning.
            splits (Tuple[str, str]): The split names, the first is the training split and the second for evaluation.
            seed (int): Seed for the finetuning, only needed for save model name here
            run_name (str): Name identifier for saving the finetuned model.
        """
        dataset_train = dataset_class(split=splits[0], original_class_ids=True, transform=get_default_train_augmentation())
        dataset_test = dataset_class(split=splits[1], original_class_ids=True)
        # So far no extra Validation set is used as there is no hyperparameter optimization or early stopping done. If one wants to do this,
        # an extra validation set needs to be defined and passed instead of the current test dataset!

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        num_classes = len(dataset_train.classes()) + 1  # +1 for Background class

        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        finetune_faster_rcnn(self.model, dataset_name, dataset_train, dataset_test, self.modality, seed, run_name)
