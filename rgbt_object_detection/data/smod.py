import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import torch
from tqdm import tqdm
import albumentations as A
from hydra import compose, initialize

import data.data_interface as data_interface
from data.data_interface import ImageLabels, LabelsEncoderDecoder


logger = logging.getLogger(__name__)

with initialize(config_path=os.path.join("..", "conf"), version_base="1.3"):
    cfg_data = compose(config_name="config").data


class SMOD(data_interface.Dataset):
    supported_classes = {
        "person": 1,
        "bicycle": 2,
        "car": 3,
        "motorcycle": 4,
    }

    def __init__(
        self,
        split: str,
        data_dir: str = cfg_data.dataset_paths.smod,
        transform: Optional[A.Compose] = None,
        original_class_ids: bool = False,
    ):
        """
        Initializes the SMOD dataset class with options for data loading and processing.

        Args:
            split (str): The dataset split to use, typically "train" or "test".
            data_dir (str): Directory of the dataset.
            transform (Optional[A.Compose]): Transformations to apply to the images.
            original_class_ids (bool): Whether to use the original class IDs defined in supported_classes.
        """
        super().__init__(split, data_dir, transform, original_class_ids)
        self.annotations_path = os.path.join(data_dir, "anno")
        self.img_label_data = self._get_imgs_labels_dict()

    def _extract_annotations(self, data: Dict[str, Any], img_id: int) -> ImageLabels:
        """
        Extracts and transforms all annotations of one image into the standardized ImageLabels format.

        Args:
            data (Dict[str, Any]): The SMOD annotation dictionary containing all annotations.
            img_id (int): The ID of the image.

        Returns:
            ImageLabels: The standardized label object with all labels for the image.
        """

        class_ids = []
        bboxes = []

        for annotation in data["annotations"]:
            if annotation["image_id"] == img_id:
                class_id = annotation["category_id"]
                bbox = annotation["bbox"]
                bbox[2] = bbox[0] + bbox[2]
                bbox[3] = bbox[1] + bbox[3]

                class_ids.append(class_id)
                bboxes.append(bbox)

        class_ids = torch.tensor(class_ids, dtype=torch.int64)
        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes, dtype=torch.int64)
        else:
            bboxes = torch.empty(0, 4, dtype=torch.int64)

        annotations = ImageLabels(class_ids, bboxes)

        return annotations

    def _get_imgs_labels_dict(self) -> Dict[int, Dict[str, Any]]:
        """
        Stores image paths and their corresponding annotations in a dictionary.
        The dictionary is also saved as a JSON file for future use.

        Returns:
            Dict[int, Dict[str, Any]]: A dictionary where:
                - Keys are indices (int).
                - Values are dictionaries containing:
                    - "rgb_path" (str): Path to the RGB image.
                    - "thermal_path" (str): Path to the thermal image.
                    - "annotations" (ImageLabels): An object storing the extracted annotations.
        """

        data_dict_path = os.path.join(self.annotations_path, f"annotations_{self.split}{'_originalid' if self.original_class_ids else ''}.json")

        if os.path.exists(data_dict_path):
            with open(data_dict_path, "r") as json_file:
                data_dict = json.load(json_file, object_hook=LabelsEncoderDecoder.from_json)

            data_dict = {int(key): value for key, value in data_dict.items()}
            logger.info(f"Loaded pre-setuped dataset SMOD {self.split} from {data_dict_path}")

        else:
            data_dict = {}

            if self.split == "all":
                splits = ["train", "test"]
            elif self.split == "train" or self.split == "test":
                splits = [self.split]
            elif self.split == "val":
                raise ValueError("No explicit val set for this dataset available.")

            for split in splits:
                mannotations_json = os.path.join(self.annotations_path, f"new_{split}_annotations_rgb.json")
                with open(mannotations_json, "r") as file:
                    data = json.load(file)

                file_names = [(img["id"], img["file_name"]) for img in data["images"]]

                for idx, (img_id, filename) in enumerate(tqdm(file_names, desc=f"Setup dataset SMOD {self.split}")):
                    data_dict[idx] = {
                        "rgb_path": None,
                        "thermal_path": None,
                        "annotations": None,
                    }
                    data_dict[idx]["rgb_path"] = os.path.join(self.data_dir, filename)
                    data_dict[idx]["thermal_path"] = os.path.join(self.data_dir, filename.replace("rgb", "tir"))
                    data_dict[idx]["annotations"] = self._extract_annotations(data, img_id)

            with open(data_dict_path, "w") as json_file:
                json.dump(data_dict, json_file, cls=LabelsEncoderDecoder)

            logger.info(f"Saved dataset setup of SMOD {self.split} to {data_dict_path}")

        return data_dict

    def classes(self) -> List[int]:
        """
        Returns:
            classes (List[int]): list of the supported classes of the dataset
        """
        return [self.coco_labels[cls_str] for cls_str in self.supported_classes.keys()]

    @staticmethod
    def standard_dataset_split_names() -> Tuple[str, str]:
        """
        Returns:
            dataset_split_names (Tuple[str, str]): A tuple with the train and test split name
        """
        return ("train", "test")

    @staticmethod
    def class_mapping_to_coco() -> Dict[int, int]:
        """
        Returns the mapping of the datasets specific class IDs to the corresponding COCO class IDs.
        This is necessary if models were fine-tuned on the dataset and used afterwards for inference to align with the standard COCO labels.

        Returns:
            class_mapping_to_coco (Dict[int, int]): The mapping of the datasets class IDs to the COCO class IDs

        """
        return {cls_id: SMOD.coco_labels[cls_str] for cls_str, cls_id in SMOD.supported_classes.items()}

    def __len__(self) -> int:
        """
        Returns:
            len (int): The number of images in the dataset.
        """
        return len(self.img_label_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Retrieves an image pair (RGB and IR) along with its annotations.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: A tuple containing:
                - RGB image (torch.Tensor) of size (3, H, W).
                - IR image (torch.Tensor) of size (1, H, W).
                - Target dictionary (Dict[str, Any]) containing "imageLabels" (ImageLabels).
        """

        img_rgb = cv2.cvtColor(cv2.imread(self.img_label_data[idx]["rgb_path"]), cv2.COLOR_BGR2RGB)
        img_ir = cv2.imread(self.img_label_data[idx]["thermal_path"], cv2.IMREAD_GRAYSCALE)
        labels = self.img_label_data[idx]["annotations"]
        img_rgb, img_ir, labels = self.transform_images_and_annotations(img_rgb, img_ir, labels)

        target = {"image_id": idx,
                  "imageLabels": labels}

        return img_rgb, img_ir, target
