import json
import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import albumentations as A
import torch
from hydra import compose, initialize
from tqdm import tqdm

import data.data_interface as data_interface
from data.data_interface import ImageLabels, LabelsEncoderDecoder


logger = logging.getLogger(__name__)

with initialize(config_path=os.path.join("..", "conf"), version_base="1.3"):
    cfg_data = compose(config_name="config").data


class FLIRaligned(data_interface.Dataset):
    supported_classes = {"person": 1, "bicycle": 2, "car": 3}

    def __init__(
        self,
        split: str,
        data_dir: str = cfg_data.dataset_paths.fliraligned,
        transform: Optional[A.Compose] = None,
        original_class_ids: bool = False,
    ):
        """
        Initializes the FLIR-aligned dataset class with options for data loading and processing.

        Args:
            split (str): The dataset split to use, typically "train" or "val".
            data_dir (str): Directory of the dataset.
            transform (Optional[A.Compose]): Transformations to apply to the images.
            original_class_ids (bool): Whether to use the original class IDs defined in supported_classes.
        """

        super().__init__(split, data_dir, transform, original_class_ids)
        self.images_path = os.path.join(data_dir, "JPEGImages")
        self.annotations_path = os.path.join(data_dir, "Annotations")

        self.filenames = self._get_filenames()
        self.img_label_data = self._get_imgs_labels_dict()

    def _get_filenames(self) -> List[str]:
        """
        Retrieves a list of filenames for the selected dataset split.

        Returns:
            List[str]: A list of image filenames corresponding to the dataset split.
        """
        filenames_dict = {
            "train": "align_train.txt",
            "val": "align_validation.txt",
        }

        if self.split == "all":
            file_keys = list(filenames_dict.keys())
        elif self.split == "train" or self.split == "val":
            file_keys = [self.split]
        elif self.split == "test":
            raise ValueError("No explicit test set for this dataset available.")

        filenames = []
        for key in file_keys:
            filename = filenames_dict[key]
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, "r") as file:
                filenames += [line.strip() for line in file.readlines()]

        return filenames

    def _extract_annotations(self, xml_annotation_path: str) -> ImageLabels:
        """
        Extracts and transforms annotations from an XML file into the standardized ImageLabels format.

        Args:
            xml_annotation_path (str): Path to the XML annotation file.

        Returns:
            ImageLabels: The standardized label object with all annotations for one image.
        """

        tree = ET.parse(xml_annotation_path)
        root = tree.getroot()

        class_ids = []
        bboxes = []

        tmp_coords = {"xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1}
        for obj in root.findall("object"):
            class_name = cast(ET.Element, obj.find("name")).text

            if class_name not in self.supported_classes:
                continue

            if self.original_class_ids:
                class_id = self.supported_classes[class_name]
            else:
                class_id = self.coco_labels[class_name]

            class_ids.append(class_id)

            bbox = cast(ET.Element, obj.find("bndbox"))
            for coord in tmp_coords.keys():
                element = bbox.find(coord)
                assert (element is not None and element.text is not None), f"Expected a value for {coord}, found None"
                tmp_coords[coord] = int(element.text)

            bboxes.append([value for value in tmp_coords.values()])

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

        data_dict_path = os.path.join(
            self.annotations_path,
            f"annotations_{self.split}{'_originalid' if self.original_class_ids else ''}.json",
        )

        if os.path.exists(data_dict_path):
            with open(data_dict_path, "r") as json_file:
                data_dict = json.load(json_file, object_hook=LabelsEncoderDecoder.from_json)

            data_dict = {int(key): value for key, value in data_dict.items()}
            logger.info(f"Loaded pre-setuped dataset FLIR aligned {self.split} from {data_dict_path}")

        else:
            data_dict = {}

            for idx, filename in enumerate(
                tqdm(self.filenames, desc=f"Setup dataset FLIR aligned {self.split}")
            ):
                data_dict[idx] = {
                    "rgb_path": None,
                    "thermal_path": None,
                    "annotations": None,
                }
                data_dict[idx]["rgb_path"] = os.path.join(self.images_path, f"{filename.replace('PreviewData', 'RGB')}.jpg")
                data_dict[idx]["thermal_path"] = os.path.join(self.images_path, f"{filename}.jpeg")
                data_dict[idx]["annotations"] = self._extract_annotations(os.path.join(self.annotations_path, f"{filename}.xml"))

            with open(data_dict_path, "w") as json_file:
                json.dump(data_dict, json_file, cls=LabelsEncoderDecoder)

            logger.info(f"Saved dataset setup of FLIR aligned {self.split} to {data_dict_path}")

        return data_dict

    def group_similar_labels_mapping(self) -> Dict[str, str]:
        """
        Returns a mapping from dataset-specific class labels to COCO class labels.

        Returns:
            Dict[str, str]: Mapping from dataset-specific class names to COCO class names.
        """
        return dict(cfg_data.dataset_label_mapping.flir_aligned)

    def classes(self) -> List[int]:
        """
        Returns:
            classes (List[int]): list of the supported classes of the dataset as COCO IDs
        """
        return [self.coco_labels[cls_str] for cls_str in self.supported_classes.keys()]

    @staticmethod
    def standard_dataset_split_names() -> Tuple[str, str]:
        """
        Returns:
            Tuple[str, str]: A tuple containing the train and validation split names.
        """
        return ("train", "val")

    @staticmethod
    def class_mapping_to_coco() -> Dict[int, int]:
        """
        Returns the mapping of the datasets specific class IDs to the corresponding COCO class IDs.
        This is necessary if models were fine-tuned on the dataset and used afterwards for inference to align with the standard COCO labels.

        Returns:
            class_mapping_to_coco (Dict[int, int]): The mapping of the datasets class IDs to the COCO class IDs

        """
        return {
            cls_id: FLIRaligned.coco_labels[cls_str]
            for cls_str, cls_id in FLIRaligned.supported_classes.items()
        }

    def __len__(self) -> int:
        """
        Returns:
            len (int): The number of images in the dataset.
        """
        return len(self.img_label_data.keys())

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor, Dict[str, Any]]:
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
