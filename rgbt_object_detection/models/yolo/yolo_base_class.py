import sys
from abc import ABC
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import models.model_interface as model_interface
import torch
from models.model_interface import ImageModelPrediction
from ultralytics import YOLO as YOLOultralytics
from ultralytics.utils.ops import non_max_suppression
from utils.finetuning_utils.finetuning_yolo import finetune_yolo

sys.stdout = sys.__stdout__


class YOLO(model_interface.Model, ABC):
    """
    This is the abstract YOLO class, that enables the integration of YOLO versions from ultralytics.
    """

    def __init__(
        self,
        selected_model_checkpoint: str,
        model_size: str,
        modality: str,
        modality_rgbir_use_individual_models: bool,
        pre_trained_weights: Optional[Union[str, List[str]]],
        class_mapping_to_coco: Optional[Dict[int, int]],
        ir_transformation: Callable[[torch.Tensor], torch.Tensor],
    ):
        """
        Initializes the YOLO model, according to selected modality and model size

        Args:
            selected_model_checkpoint (str): The checkpoint (.pt file) that specifies the weights to be loaded
            model_size (str): The size of the YOLO model.
            modality (str): The type of modality to be used by the model. It can be 'rgb' 'ir', or 'rgbir'. If 'rgbir' is selected,
                the model(s) will be ran for both modalities and a simple nms late fusion will be applied.
            modality_rgbir_use_individual_models (bool): Only relevant when modality == 'rgbir': use the identical model instance for rgb
                and ir detection or use individual model instances for each modality.
            pre_trained_weights (Optional[str]): Path to pretrained weights file. If None, by default the pre-trained COCO weights are loaded.
            class_mapping_to_coco (Dict[int, int]): A dictionary that specifies the mapping from the model's output classes to the COCO classes.
                This is only necessary if an own pre-trained model is used. Must be provided if pre-trained weights are used.
            ir_transformation (Callable[[torch.Tensor], torch.Tensor]): A transformation applied to the ir image to be suitable for the
                3 channel YOLO model. Default is just copying the 1 ir dimension to all 3 channels.
        """
        super().__init__(
            modality=modality,
            ir_transformation=ir_transformation,
            pre_trained_weights=pre_trained_weights,
            modality_rgbir_use_individual_models=modality_rgbir_use_individual_models,
            class_mapping_to_coco=class_mapping_to_coco,
        )

        if self.modality == "rgbir" and modality_rgbir_use_individual_models:
            self.model_rgb = YOLOultralytics(selected_model_checkpoint).model
            self.model_ir = YOLOultralytics(selected_model_checkpoint).model
        else:
            self.model = YOLOultralytics(selected_model_checkpoint).model

        self.model_size = model_size
        self.nms_conf = 0.25 # Hard coded minimum confidence, introduce parameter for this if wanted.
        self.nms_iou = 0.75 # Hard coded minimum confidence, introduce parameter for this if wanted.

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

    def pre_process(self, standardized_input: Tuple[torch.Tensor, torch.Tensor]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Converts the standardized input data to the format required for the YOLO specific model.

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
            raise ValueError("Fatal error: Modality must be either 'rgb', 'ir' or 'rgbir'")

    def post_process(
        self, model_output: List[torch.Tensor]
    ) -> List[ImageModelPrediction]:
        """
        Converts the output data of the YOLO specific model to the standardized format.

        Args:
            model_output (List[torch.Tensor]): Output data in the raw format of the YOLO model.
                It's a list of length batch_size with the predictions per image as torch.Tensor of size (n_predictions, 6).

        Returns:
            standardized_output (List[ImageModelPrediction]): Processed output data in the standardized format.
        """

        standardized_output = [
            ImageModelPrediction(
                class_ids=pred[:, 5].int() + 1
                if not self.pre_trained_weights
                else pred[:, 5].int(),
                bboxes=pred[:, :4],
                scores=pred[:, 4],
            )
            for pred in model_output
        ]

        return standardized_output

    def predict(
        self, model_input: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Predicts the output data for the given input data using the YOLO specific model.

        Args:
            model_input (Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]): Either tuple of two torch.Tensor
                or one torch.Tensor each with the size (bs, 3, img_height, img_width)

        Returns:
            model_output (List[torch.Tensor]): Output data in the raw format of the YOLO model.
                It's a list of length batch_size with the predictions per image as torch.Tensor of size (n_predictions, 6).
        """

        # Note: For rgbir, proper inference time evaluation is not reasonable with current implementation, since NMS is done multiple times!
        if self.modality == "rgbir":
            rgb_batch, ir_batch = model_input

            # If a late fusion needs to be done, the output is brought to the required raw YOLO output format from ultralytics and then afterwards a NMS is done.
            # The data format looks a little confusing, please refer to the ultralytics repository for clarification.

            if self.modality_rgbir_use_individual_models:
                # For a correct evaluation of the inference time, the following two lines would have to be parallelized.
                model_output_rgb_unfiltered = self.model_rgb(rgb_batch)
                model_output_ir_unfiltered = self.model_ir(ir_batch)
                model_output_unfiltered = (
                    torch.cat(
                        (model_output_rgb_unfiltered[0], model_output_ir_unfiltered[0]),
                        dim=-1,
                    ),
                    None,
                )  # The second dimension is ignored here in general, since not relevant for inference evaluation.

            else:
                concatenated_model_input = torch.cat([rgb_batch, ir_batch], dim=0)
                concatenated_model_output = self.model(concatenated_model_input)
                bs = rgb_batch.size(0)
                rgb_outputs, ir_outputs = torch.split(concatenated_model_output[0], split_size_or_sections=bs, dim=0)
                model_output_unfiltered = (
                    torch.cat([rgb_outputs, ir_outputs], dim=-1),
                    None,
                )  # The second dimension is ignored here in general, since not relevant for inference evaluation.

        else:
            model_output_unfiltered = self.model(model_input)

        model_output = non_max_suppression(
            model_output_unfiltered,
            conf_thres=self.nms_conf,
            iou_thres=self.nms_iou,
        )

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
            seed (int): Random seed for reproducibility.
            run_name (str): Experiment name for logging and saving.
        """
        datasets = {}
        datasets["train"] = dataset_class(split=splits[0], original_class_ids=True)
        datasets["val"] = dataset_class(split=splits[1], original_class_ids=True)

        finetune_yolo(self.model, datasets, dataset_name, self.modality, self.__class__.__name__, self.model_size, seed, run_name)
