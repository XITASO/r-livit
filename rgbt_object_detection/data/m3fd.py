import json
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple, cast

import cv2
import torch
import albumentations as A
from tqdm import tqdm
from hydra import compose, initialize

import data.data_interface as data_interface
from data.data_interface import ImageLabels, LabelsEncoderDecoder


logger = logging.getLogger(__name__)

with initialize(config_path=os.path.join("..", "conf"), version_base="1.3"):
    cfg_data = compose(config_name="config").data


class M3FD(data_interface.Dataset):
    supported_classes = {
        "person": 1,
        "car": 2,
        "bus": 3,
        "truck": 4,
        "motorcycle": 5,
    }

    def __init__(
        self,
        split: str,
        data_dir: str = cfg_data.dataset_paths.m3fd,
        transform: Optional[A.Compose] = None,
        original_class_ids: bool = False,
    ):
        """
        Initializes the M3FD dataset class with options for data loading and processing.

        Args:
            split (str): The dataset split to use, typically "train" or "val".
            data_dir (str): Directory of the dataset.
            transform (Optional[A.Compose]): Transformations to apply to the images.
            original_class_ids (bool): Whether to use the original class IDs defined in supported_classes.
        """

        super().__init__(split, data_dir, transform, original_class_ids)
        self.rgb_images_path = os.path.join(data_dir, "Vis")
        self.ir_images_path = os.path.join(data_dir, "Ir")
        self.annotations_path = os.path.join(data_dir, "Annotation")
        label_name_map = {
            "People": "person",
            "Car": "car",
            "Bus": "bus",
            "Truck": "truck",
            "Motorcycle": "motorcycle",
            "Lamp": "lamp",
        }
        self.filenames = self._get_filenames()
        self.img_label_data = self._get_imgs_labels_dict(label_name_map)

    def _get_filenames(self) -> List[str]:
        """
        Retrieves a list of filenames for the selected dataset split.

        Returns:
            List[str]: A list of image filenames corresponding to the dataset split.
        """

        if self.split == "all":
            splits = ["train", "val"]
        elif self.split == "train" or self.split == "val":
            splits = [self.split]
        elif self.split == "test":
            raise ValueError("No explicit test set for this dataset available.")

        considered_image_numbers = []
        for split in splits:
            metadata_json = os.path.join(
                self.annotations_path, f"instances_{split}2014.json"
            )
            with open(metadata_json, "r") as file:
                data = json.load(file)
            considered_image_numbers.extend([int(re.search(r"\d+", image["file_name"]).group()) for image in data["images"]])

        filenames = []
        for file in os.listdir(self.annotations_path):
            if file.endswith(".xml"):
                root = ET.parse(os.path.join(self.annotations_path, file)).getroot()
                width = cast(ET.Element, root.find("size").find("width").text)
                # The images of other sizes then 1024 will be ignored, since they're very few and not relevant for autonomous driving.
                # Adjust if needed or if full representable benchmarking should be run.
                if int(width) != 1024:         
                    continue

                img_number = file[:-4]
                if int(img_number) not in considered_image_numbers:
                    continue

                filenames.append(img_number)

        return filenames

    def _extract_annotations(self, xml_annotation_path: str, label_name_map: Dict[str, str]) -> ImageLabels:
        """
        Extracts and transforms annotations from an XML file into the standardized ImageLabels format.

        Args:
            xml_annotation_path (str): Path to the XML annotation file.
            label_name_map (Dict[str, str]): Mapping from dataset-specific class names to COCO class names.

        Returns:
            ImageLabels: The standardized label object with all annotations for one image.
        """

        tree = ET.parse(xml_annotation_path)
        root = tree.getroot()

        class_ids = []
        bboxes = []

        tmp_coords = {"xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1}
        for obj in root.findall("object"):
            class_name_m3fd = cast(ET.Element, obj.find("name")).text
            assert class_name_m3fd is not None
            class_name = label_name_map[class_name_m3fd]

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
                assert (
                    element is not None and element.text is not None
                ), f"Expected a value for {coord}, found None"
                tmp_coords[coord] = int(element.text)

            bboxes.append([value for value in tmp_coords.values()])

        class_ids = torch.tensor(class_ids, dtype=torch.int64)
        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes)
        else:
            bboxes = torch.empty(0, 4)

        annotations = ImageLabels(class_ids, bboxes)

        return annotations

    def _get_imgs_labels_dict(self, label_name_map: Dict[str, str]) -> Dict[int, Dict[str, Any]]:
        """
        Stores image paths and their corresponding annotations in a dictionary.
        The dictionary is also saved as a JSON file for future use.

        Args:
            label_name_map (Dict[str, str]): Mapping from dataset-specific class names to COCO class names.

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
            logger.info(f"Loaded pre-setuped dataset M3FD {self.split} from {data_dict_path}")

        else:
            data_dict = {}

            for idx, filename in enumerate(tqdm(self.filenames, desc=f"Setup dataset M3FD {self.split}")):
                data_dict[idx] = {
                    "rgb_path": None,
                    "thermal_path": None,
                    "annotations": None,
                }
                data_dict[idx]["rgb_path"] = os.path.join(self.rgb_images_path, f"{filename}.png")
                data_dict[idx]["thermal_path"] = os.path.join(self.ir_images_path, f"{filename}.png")
                data_dict[idx]["annotations"] = self._extract_annotations(os.path.join(self.annotations_path, f"{filename}.xml"), label_name_map)

            with open(data_dict_path, "w") as json_file:
                json.dump(data_dict, json_file, cls=LabelsEncoderDecoder)

            logger.info(f"Saved dataset setup of M3FD {self.split} to {data_dict_path}")

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
        return ("train", "val")

    @staticmethod
    def class_mapping_to_coco() -> Dict[int, int]:
        """
        Returns the mapping of the datasets specific class IDs to the corresponding COCO class IDs.
        This is necessary if models were fine-tuned on the dataset and used afterwards for inference to align with the standard COCO labels.

        Returns:
            class_mapping_to_coco (Dict[int, int]): The mapping of the datasets class IDs to the COCO class IDs

        """
        return {cls_id: M3FD.coco_labels[cls_str] for cls_str, cls_id in M3FD.supported_classes.items()}

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
