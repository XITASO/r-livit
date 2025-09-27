import json
import os
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from hydra import compose, initialize


with initialize(config_path=os.path.join("..", "conf"), version_base="1.3"):
    cfg = compose(config_name="config")
    cfg_data = cfg.data
    cfg_setup = cfg.setup


@dataclass
class ImageLabels:
    """
    A class representing the standardized label for one image, which is several objects
    It contains a tensor of class identifiers and a tensor of bounding boxes of the same size.
    Attributes:
        class_ids (torch.Tensor): Tensor of size (n_objects) with integer values.
                Represents class ids as per COCO labels.
        bboxes (torch.Tensor): Tensor of size (n_objects, 4).
                Contains bounding boxes format [x_min, y_min, x_max, y_max]
                whereas the values are NOT normalized and x in range(img_width), y in range(img_height)
    """

    class_ids: torch.Tensor
    bboxes: torch.Tensor

    def __post_init__(self):
        assert self.class_ids.size(0) == self.bboxes.size(
            0
        ), "Mismatch in number of objects."
        assert self.class_ids.dtype in {
            torch.int16,
            torch.int32,
            torch.int64,
        }, "class_ids must be of type torch.int."
        assert (
            self.bboxes.size(1) == 4
        ), f"BBoxes must be of length 4. Got {self.class_ids.size(1)}"
        assert torch.all(
            self.bboxes[..., 0] <= self.bboxes[..., 2]
        ), "x_min must be less than x_max in bbox."
        assert torch.all(
            self.bboxes[..., 1] <= self.bboxes[..., 3]
        ), "y_min must be less than y_max in bbox."

    def to(self, device: torch.device) -> None:
        """Move tensors to device
        Args:
            device (torch.device): The device where you want to move your tensors.
        """
        self.class_ids = self.class_ids.to(device)
        self.bboxes = self.bboxes.to(device)


class LabelsEncoderDecoder(json.JSONEncoder):
    """
    This class offers functions to (de)serialize the ImageLabels class for json
    """

    def default(self, obj: Any) -> Dict[str, Any]:
        """
        Overrides the `default` method of `json.JSONEncoder` for encoding `ImageLabels` objects.

        Args:
            obj (Any): The python object to be encoded.

        Returns:
            A dictionary representation of the `ImageLabels` object if the `obj` is an instance of `ImageLabels`.
            Otherwise, calls the parent method to handle the `obj`.
        """
        if isinstance(obj, ImageLabels):
            return {"class_ids": obj.class_ids.tolist(), "bboxes": obj.bboxes.tolist()}
        return super().default(obj)

    @staticmethod
    def from_json(json_label: Dict[str, Any]) -> Union[ImageLabels, Dict[str, Any]]:
        """
        Converts a JSON representation of a `Label` object back to `Label`.

        Args:
        json_label (Dict[str, Any]): A dictionary containing 'class_ids' and 'bboxes' keys.

        Returns:
        Union[ImageLabels, Dict[str, Any]]: An `ImageLabels` object if the input contains the required keys.
            Otherwise, the input dictionary is returned unchanged.
        """
        if "class_ids" in json_label and "bboxes" in json_label:
            if len(json_label["class_ids"]) > 0:
                class_ids = torch.tensor(json_label["class_ids"], dtype=torch.int)
                bboxes = torch.tensor(json_label["bboxes"])
            else:
                class_ids = torch.empty(0, dtype=torch.int)
                bboxes = torch.empty(0, 4)
            return ImageLabels(
                class_ids,
                bboxes,
            )
        return json_label


class Dataset(torch.utils.data.Dataset, metaclass=ABCMeta):
    """
    Abstract dataset class, that sepcifies the required standardized format for our pipeline
    """

    with open(cfg_data.labels_path, "r") as file:
        coco_labels = {v: k for k, v in eval(file.read()).items()}

    def __init__(
        self,
        split: str,
        data_dir: str,
        transform: Optional[A.Compose],
        original_class_ids: bool = False,
    ):
        """
        Args:
            split (str): The dataset split to use. Must be one of 'all', 'train', 'test', or 'val'.
            data_dir (str): path to the root folder of the dataset
            transform (A.Compose): A torchvision.transforms.Compose object that contains a list of
                                           transforms to apply to the input data. Default is None.
            original_class_ids (bool): Whether to use the datasets class IDs or the default label IDs.

        Raises:
            ValueError: If the value of the 'split' parameter is not one of 'train', 'test', or 'val'.
        """
        self.data_dir = data_dir
        if split not in ["all", "train", "test", "val"]:
            raise ValueError(
                f"Invalid value for parameter 'split': {split}. Allowed values are 'all', 'train', 'test', and 'val'."
            )
        self.split = split
        if transform is None:
            transform = A.Compose([ToTensorV2()],bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_ids"]))
        self.transform = transform
        self.original_class_ids = original_class_ids

    def transform_images_and_annotations(
        self, rgb_img: np.ndarray, ir_img: np.ndarray, labels: ImageLabels
    ) -> Tuple[torch.Tensor, torch.Tensor, ImageLabels]:
        """
        Applies transformations to the input RGB and IR images and their corresponding labels.

        Args:
            rgb_image (np.ndarray): Input RGB image.
            ir_img (np.ndarray): Input IR image.
            labels (ImageLabels): The targets of the image.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, ImageLabels]: The transformed RGB image, IR image, and labels.
        """

        ir_img = ir_img.reshape((ir_img.shape[0], ir_img.shape[1], 1))
        combined_img = np.concatenate([rgb_img, ir_img], axis=-1)
        bboxes = labels.bboxes.numpy()
        class_ids = labels.class_ids.numpy().tolist()

        height, width = combined_img.shape[:2]

        # Check for valid bounding boxes
        corrected_bboxes = []
        corrected_class_ids = []
        for bbox, class_id in zip(bboxes, class_ids):
            x_min, y_min, x_max, y_max = bbox
            # Clamp values to ensure they are within image bounds
            x_min = max(0, min(width, x_min))
            y_min = max(0, min(height, y_min))
            x_max = max(0, min(width, x_max))
            y_max = max(0, min(height, y_max))

            if x_min < x_max and y_min < y_max:
                corrected_bboxes.append((x_min, y_min, x_max, y_max))
                corrected_class_ids.append(class_id)

        augmented = self.transform(
            image=combined_img, bboxes=corrected_bboxes, class_ids=corrected_class_ids
        )

        transformed_rgb_img = augmented["image"][:3, :, :]
        transformed_ir_img = augmented["image"][3, :, :]
        transformed_bboxes = augmented['bboxes']
        transformed_class_ids = augmented['class_ids']

        filtered_boxes_and_classes = [
            (bbox, class_id) for bbox, class_id in zip(transformed_bboxes, transformed_class_ids)
            if (bbox[2] - bbox[0]) > 1 and (bbox[3] - bbox[1]) > 1
        ]
        transformed_bboxes, transformed_class_ids = zip(*filtered_boxes_and_classes) if filtered_boxes_and_classes else (torch.empty(0, 4, dtype=torch.int64), torch.empty(0, dtype=torch.int64))

        if transformed_rgb_img.dtype in [torch.uint8, torch.int]:
            transformed_rgb_img = transformed_rgb_img.float() / 255.0
        if transformed_ir_img.dtype in [torch.uint8, torch.int]:
            transformed_ir_img = transformed_ir_img.float() / 255.0

        transformed_rgb_img = torch.clamp(transformed_rgb_img, 0.0, 1.0)
        transformed_ir_img = torch.clamp(transformed_ir_img, 0.0, 1.0)
        transformed_ir_img = transformed_ir_img.unsqueeze(0)

        transformed_labels = ImageLabels(
            class_ids=torch.tensor(transformed_class_ids, dtype=labels.class_ids.dtype),
            bboxes=torch.tensor(transformed_bboxes, dtype=labels.bboxes.dtype),
        )

        return transformed_rgb_img, transformed_ir_img, transformed_labels

    def group_similar_labels_mapping(self) -> Dict[str, str]:
        """
        Provides a mapping of dataset-specific class names to COCO class names.

        Returns:
            classes (Dict[str, str]): Returns a grouping/aggregation from the datasets classes to the coco classes.
        """
        return {}

    @abstractmethod
    def classes(self) -> List[int]:
        """
        Returns:
            classes (List[int]): list of the supported classes of the dataset as COCO IDs.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def standard_dataset_split_names() -> Tuple[str, str]:
        """
        Returns a tuple with the default train (mostly "train") and test (often "test" or "val") split name of the dataset, and hence indirectly maps it to a standard.
        This is necessary, since different datasets use different split names and we keep the original split names in the datasets.

        Returns:
            Tuple[str, str]: A tuple containing the train and validation split names.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def class_mapping_to_coco() -> Dict[int, int]:
        """
        Returns the mapping of the datasets specific class IDs to the corresponding COCO class IDs.
        This is necessary if models were fine-tuned on the dataset and used afterwards for inference to align with the default labels.

        Returns:
            class_mapping_to_coco (Dict[int, int]): The mapping of the datasets class IDs to the COCO class IDs
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns:
            len (int): length of the dataset
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            tuple: (rgb_img, ir_img, labels) where
                - rgb_img (torch.Tensor)
                - ir_img (torch.Tensor)
                - target (Dict[str, Any]) of which one entry is "imageLabels": ImageLabels
        """
        raise NotImplementedError


def collate_fn_benchmarking(
    batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]],
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[ImageLabels]]:
    """
    Collates a batch of samples into a batch of tensors.

    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]): This is the standard output of the datasets

    Returns:
        tuple: (rgb_img, ir_img, labels) in the standard batch format where
            - rgb_img (torch.Tensor)
            - ir_img (torch.Tensor)
            - labels (Tuple[ImageLabels])
    """

    rgb_batch, ir_batch, targets_batch = zip(*batch)

    labels_batch = tuple([target["imageLabels"] for target in targets_batch])

    rgb_batch = torch.stack(rgb_batch)
    ir_batch = torch.stack(ir_batch)

    return rgb_batch, ir_batch, labels_batch


def get_default_train_augmentation() -> A.Compose:
    def apply_brightness_to_rgb_ir(image: np.array, **kwargs) -> np.array:
        rgb_img = image[:, :, :3]
        ir_img = image[:, :, 3:]
        if np.random.rand() < 0.5:
            rgb_img = A.ColorJitter(
                brightness=0.2, contrast=0.0, saturation=0.7, hue=0.015, p=1.0
            )(image=rgb_img)["image"]
            ir_img = A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.0, p=1.0
            )(image=ir_img)["image"]
        return np.dstack((rgb_img, ir_img))

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Lambda(image=apply_brightness_to_rgb_ir),
            A.GaussianBlur(blur_limit=(1, 3), p=0.2),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_ids"]),
    )


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Transforms bounding box from cxcywh to xyxy format

    Args:
        x (torch.Tensor): Tensor of bounding boxes in center format [cx,cy,w,h]

    Returns:
        x (torch.Tensor): Transformed bounding boxes in corner format [xmin, ymin, xmax, ymax]
    """
    cxcy, wh = x[:, :2], x[:, 2:]
    x = torch.zeros_like(x)
    x[:, :2] = cxcy - wh / 2
    x[:, 2:] = cxcy + wh / 2
    return x


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    """
    Transforms bounding box from xyxy to cxcywh format

    Args:
        x (torch.Tensor): Tensor of bounding boxes in corners format [xmin, ymin, xmax, ymax]

    Returns:
        x (torch.Tensor): Transformed bounding boxes in center format [cx,cy,w,h]
    """
    xy_min, xy_max = x[:, :2], x[:, 2:]
    x = torch.zeros_like(x)
    x[:, :2] = (xy_min + xy_max) / 2
    x[:, 2:] = xy_max - xy_min
    return x


def ir_transformation_copy(ir_batch: torch.Tensor) -> torch.Tensor:
    """
    Copies the ir_batch from 1 to 3 channels

    Args:
        ir_batch (torch.Tensor): the ir tensor with 1 input channel

    Returns:
        ir_batch (torch.Tensor): the ir tensor with 3 input channels
    """
    return ir_batch.repeat(1, 3, 1, 1)


def convert_coco_labels_from_old_to_new(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts old COCO label indices to new COCO label indices based on mapping files.

    Args:
        input_tensor (torch.Tensor): A tensor containing integer values of old COCO label indices.

    Returns:
        torch.Tensor: A tensor of the same shape as input_tensor, with old COCO label indices replaced by new COCO label indices.
                      Labels that no longer exist are replaced with -1.
    """

    input_tensor = input_tensor.cpu()

    with open(cfg_data.labels_path, "r") as file:
        coco_inv_dict = eval(file.read())

    with open(cfg_data.labels_coco90_path, "r") as file:
        coco_old_inv_dict = eval(file.read())

    coco_dict = {v: k for k, v in coco_inv_dict.items()}
    label_mapping = {
        index_old: coco_dict.get(label_old, -1)
        for index_old, label_old in coco_old_inv_dict.items()
    }

    output_tensor = input_tensor.apply_(lambda x: label_mapping[x])

    output_tensor.to(cfg_setup.device)

    return output_tensor
