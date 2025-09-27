import json
import logging
import os
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


class KAIST(data_interface.Dataset):
    supported_classes = {"person": 1}

    def __init__(
        self,
        split: str,
        data_dir: str = cfg_data.dataset_paths.kaist,
        transform: Optional[A.Compose] = None,
        original_class_ids: bool = False,
    ):
        """
        Initializes the KAIST dataset class with options for data loading and processing.

        Args:
            split (str): The dataset split to use, typically "train" or "test".
            data_dir (str): Directory of the dataset.
            transform (Optional[A.Compose]): Transformations to apply to the images.
            original_class_ids (bool): Whether to use the original class IDs defined in supported_classes.
        """

        super().__init__(split, data_dir, transform)
        self.data_dir = data_dir
        self.images_path = os.path.join(data_dir, "images")
        self.annotations_path = os.path.join(data_dir, "annotations-xml-new-sanitized")
        self.filenames = self._get_filenames()
        self.img_label_data = self._get_imgs_labels_dict()
        self.original_class_ids = original_class_ids

    def _get_filenames(self) -> List[str]:
        """
        Retrieves a list of filenames for the selected dataset split.

        Returns:
            List[str]: A list of image filenames corresponding to the dataset split.
        """

        packages_dict = {
            "train": ["train-all-01", "train-all-02", "train-all-04", "train-all-20"],
            "test": ["test-all-01", "test-all-20"],
        }

        if self.split == "all":
            package_keys = list(packages_dict.keys())
        elif self.split == "train" or self.split == "test":
            package_keys = [self.split]
        elif self.split == "val":
            raise ValueError("No explicit val set for this dataset available.")

        packages = [package for key in package_keys for package in packages_dict[key]]

        filenames = []
        for package in packages:
            with open(
                os.path.join(self.data_dir, "imageSets", f"{package}.txt"), "r"
            ) as file:
                filenames += [line.strip() for line in file.readlines()]

        filenames = list(set(filenames))

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

        bboxes = []

        for obj in root.findall("object"):
            bbox = cast(ET.Element, obj.find("bndbox"))

            xmin_elem = bbox.find("x")
            assert (
                xmin_elem is not None and xmin_elem.text is not None
            ), f"Expected a value for {xmin_elem}, found None"
            xmin = int(xmin_elem.text)

            ymin_elem = bbox.find("y")
            assert (
                ymin_elem is not None and ymin_elem.text is not None
            ), f"Expected a value for {ymin_elem}, found None"
            ymin = int(ymin_elem.text)

            width_elem = bbox.find("w")
            assert (
                width_elem is not None and width_elem.text is not None
            ), f"Expected a value for {width_elem}, found None"
            xmax = xmin + int(width_elem.text)

            height_elem = bbox.find("h")
            assert (
                height_elem is not None and height_elem.text is not None
            ), f"Expected a value for {height_elem}, found None"
            ymax = ymin + int(height_elem.text)

            bboxes.append([xmin, ymin, xmax, ymax])

        class_id_person = self.coco_labels["person"]
        class_ids = torch.full((len(bboxes),), class_id_person, dtype=torch.int)

        if len(bboxes) > 0:
            bboxes = torch.tensor(bboxes)
        else:
            bboxes = torch.empty(0, 4)
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

        data_dict_path = os.path.join(self.annotations_path, f"annotations_{self.split}.json")

        if os.path.exists(data_dict_path):
            with open(data_dict_path, "r") as json_file:
                data_dict = json.load(json_file, object_hook=LabelsEncoderDecoder.from_json)

            data_dict = {int(key): value for key, value in data_dict.items()}
            logger.info(f"Loaded pre-setuped dataset KAIST {self.split} from {data_dict_path}")

        else:
            data_dict = {}

            for idx, filename in enumerate(tqdm(self.filenames, desc=f"Setup dataset KAIST {self.split}")):
                set_number, version, img_id = filename.split("/")

                data_dict[idx] = {
                    "rgb_path": None,
                    "thermal_path": None,
                    "annotations": None,
                }
                data_dict[idx]["rgb_path"] = os.path.join(self.images_path, set_number, version, "visible", f"{img_id}.jpg")
                data_dict[idx]["thermal_path"] = os.path.join(self.images_path, set_number, version, "lwir", f"{img_id}.jpg")
                data_dict[idx]["annotations"] = self._extract_annotations(os.path.join(self.annotations_path, set_number, version, f"{img_id}.xml"))

            with open(data_dict_path, "w") as json_file:
                json.dump(data_dict, json_file, cls=LabelsEncoderDecoder)

            logger.info(f"Saved dataset setup of KAIST {self.split} to {data_dict_path}")

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
        return {cls_id: KAIST.coco_labels[cls_str] for cls_str, cls_id in KAIST.supported_classes.items()}

    def __len__(self):
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
