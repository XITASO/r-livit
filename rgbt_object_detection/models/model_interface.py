import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from hydra import compose, initialize

with initialize(config_path=os.path.join("..", "conf"), version_base="1.3"):
    cfg = compose(config_name="config").data


@dataclass
class ImageModelPrediction:
    """A class representing the standardized prediction of a model for one image.

    Attributes:
        class_ids (torch.Tensor): A torch tensor of size (batch_size, n_predictions).
            It contains integer values of class ids corresponding to COCO labels.
        bboxes (torch.Tensor): A torch tensor of size (batch_size, n_predictions, 4).
            It contains the predicted bounding boxes in the format [x_min, y_min, x_max, y_max].
            The values are not normalized and x lies within range(img_width)
            and y within range(img_height).
        scores (torch.Tensor): A torch tensor of size (batch_size, n_predictions),
            that contains the scores for each prediction as float values.
    """

    class_ids: torch.Tensor
    bboxes: torch.Tensor
    scores: torch.Tensor

    def __post_init__(self):
        """Performs assertions after initializing the instance to ensure data validity."""
        assert (
            self.class_ids.size(0) == self.bboxes.size(0) == self.scores.size(0)
        ), "Mismatch in number of predictions."
        assert (
            self.bboxes.size(1) == 4
        ), f"BBoxes must be of length 4. Got {self.bboxes.size(1)}"
        assert torch.all(
            self.bboxes[..., 0] < self.bboxes[..., 2]
        ), "x_min must be less than x_max in bbox."
        assert torch.all(
            self.bboxes[..., 1] < self.bboxes[..., 3]
        ), "y_min must be less than y_max in bbox."
        assert torch.all(
            (0.0 <= self.scores) & (self.scores <= 1.0)
        ), "All scores values must be within [0, 1] interval."

    def to(self, device: torch.device) -> None:
        """Move ImageModelPrediction to device
        Args:
            device (torch.device): The device where you want to move the model.
        """
        self.class_ids = self.class_ids.to(device)
        self.bboxes = self.bboxes.to(device)
        self.scores = self.scores.to(device)


class Model(metaclass=ABCMeta):
    """
    Abstract base class for implementing different model wrappers.
    Each model should inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        modality: str,
        ir_transformation: Callable[[torch.tensor], torch.tensor],
        pre_trained_weights: Optional[Union[str, List[str]]] = None,
        modality_rgbir_use_individual_models: bool = False,
        class_mapping_to_coco: Optional[Dict[int, int]] = None,
    ):
        """Initialize the model with its modality type.

        Args:
            modality (str): The type of modality to be used by the model. It can be 'rgb', 'ir', or 'rgbir'.
            ir_transformation (Optional[Callable[[torch.tensor], torch.tensor]])): that transforms the ir image to a 3 channel image
                only needs to be provided if the model requires a 3 channel input
            pre_trained_weights (Optional[Union[str, List[str]]]): Path to pretrained weights file. If None, by default the pre-trained
                COCO weights are loaded. If a rgbir late fusion model is used with seperate rgb and ir models, a list of 2 strings must
                be provided, whereas the first represents the pre-trained weights for the rgb model and the second for the ir model
            modality_rgbir_use_individual_models (bool): Only relevant when modality == 'rgbir' and a single modality model is used: use
                the identical model instance for rgb and ir detection or use individual model instances for each modality.
            class_mapping_to_coco (Dict[int, int]): A dictionary that specifies the mapping from the model's output classes to the COCO
                classes. This is only necessary if an own pre-trained model is used. Must be provided if pre-trained weights are used.
        """
        if modality not in ["rgb", "ir", "rgbir"]:
            raise ValueError("Modality must be either 'rgb', 'ir' or 'rgbir'")
        self.modality = modality

        assert (
            not pre_trained_weights or class_mapping_to_coco
        ), "If pre_trained_weights are used, class_mapping_to_coco must be provided."

        assert (
            not pre_trained_weights
            or modality != "rgbir"
            or not modality_rgbir_use_individual_models
            or (len(pre_trained_weights) == 2)
        ), "If pre_trained_weights are used with an rgbir model where individual models are required, a list of two weight-paths must be provided."

        self.ir_transformation = ir_transformation
        self.pre_trained_weights = pre_trained_weights
        self.modality_rgbir_use_individual_models = modality_rgbir_use_individual_models
        self.class_mapping_to_coco = class_mapping_to_coco

    @abstractmethod
    def eval(self):
        """
        Sets the model to evaluation mode
        """
        raise NotImplementedError

    @abstractmethod
    def to(self, device: torch.device):
        """ 
        Moves model to device.

        Args:
            device (torch.device): The device where you want to move the model.
        """
        raise NotImplementedError

    @abstractmethod
    def pre_process(self, standardized_input: Tuple[torch.tensor, torch.tensor]) -> Any:
        """
        Converts the standardized input data to the format required for the specific model.

        Args:
            standardized_input: Input data in the standardized format where
                Tuple[0]: rgb_batch (torch.tensor) of the size (bs, 3, img_height, img_width)
                Tupe[1]: ir_batch (torch.tensor) of the size (bs, 1, img_height, img_width)

        Returns:
            model_input: Processed input data in the required format for the specific model
        """
        raise NotImplementedError

    @abstractmethod
    def post_process(self, model_output: Any) -> List[ImageModelPrediction]:
        """
        Converts the output data of the specific model to the standardized format.

        Args:
            model_output: Output data in the raw format of the specific model

        Returns:
            standardized_output: Processed output data in the standardized format where
                Tuple is of size bs with bs ImageModelPrediction
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, model_input: Any) -> Any:
        """
        Predicts the output data for the given input data.

        Args:
            model_input: Input data in the required format for the specific model

        Returns:
            model_output: Output data in the raw format of the specific model
        """
        raise NotImplementedError


def get_pretrained_weights_path(
    model: str,
    dataset: str,
    modality: str,
    seed: int,
    model_size: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    """
    Helper function to retrieve the path to the pretrained weights file (.pth or .pt) for a specific model, dataset, and modality.

    Args:
        model (str): The name of the model (e.g., 'FasterRCNN').
        dataset (str): The name of the dataset (e.g., 'M3FD').
        modality (str): The modality (e.g., 'ir').
        seed (int): The finetuning seed of the saved model.
        model_size (Optional[str]): The size of the model.
                              If None, files without model_size in the name are considered.
        timestamp (Optional[str]): The specific timestamp of the file in the format 'YYYYMMDD_HHMMSS'.
                                   If None, the newest available file is returned.
    Returns:
        str: The file path to the pretrained weights file.
    """
    model_path = os.path.join(cfg.results_path, "finetuning", model.lower())

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model directory '{model_path}' for model {model} does not exist."
        )

    files = os.listdir(model_path)

    filtered_files = []
    for file in files:
        if file.startswith("state_dict") and (
            file.endswith(".pth") or file.endswith(".pt")
        ):
            parts = file.split("-")
            if parts[1] == dataset.lower() and parts[2] == modality.lower() and int(parts[3]) == seed:
                if model_size is None:
                    if len(parts) == 5:
                        filtered_files.append(file)
                else:
                    if len(parts) == 6 and parts[4] == f"size_{model_size.lower()}":
                        filtered_files.append(file)

    if not filtered_files:
        raise FileNotFoundError(
            f"No files found for model: {model}, dataset: {dataset}, modality: {modality}, seed: {seed}, model_size: {model_size}"
        )

    if timestamp:
        model_size_substring = (
            "" if model_size is None else f"-size_{model_size.lower()}"
        )
        timestamp_file_pth = f"state_dict-{dataset.lower()}-{modality.lower()}-{seed}{model_size_substring}-{timestamp}.pth"
        timestamp_file_pt = f"state_dict-{dataset.lower()}-{modality.lower()}-{seed}{model_size_substring}-{timestamp}.pt"
        if timestamp_file_pth in filtered_files:
            return os.path.join(model_path, timestamp_file_pth)
        elif timestamp_file_pt in filtered_files:
            return os.path.join(model_path, timestamp_file_pt)
        else:
            raise FileNotFoundError(f"No files found for timestamp: {timestamp}")

    filtered_files.sort(
        key=lambda x: datetime.strptime(
            x.split("-")[-1].rstrip(".pth").rstrip(".pt"), "%Y%m%d_%H%M%S"
        ),
        reverse=True,
    )

    newest_file = filtered_files[0]
    return os.path.join(model_path, newest_file)
