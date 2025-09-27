import json
import logging
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from hydra import compose, initialize
from tqdm import tqdm

import data.data_interface as data_interface
from data.data_interface import ImageLabels, LabelsEncoderDecoder


logger = logging.getLogger(__name__)

with initialize(config_path=os.path.join("..", "conf"), version_base="1.3"):
    cfg_data = compose(config_name="config").data


class RLiViT(data_interface.Dataset):
    supported_classes = {
        "person": 1,
        "bicycle": 2,
        "car": 3,
        "motorcycle": 4,
        "bus": 5,
        "tramway": 6,
        "truck": 7,
        "escooter": 8,
    }

    def __init__(
        self,
        split: str,
        data_dir: str = cfg_data.dataset_paths.rlivit,
        transform: Optional[A.Compose] = None,
        original_class_ids: bool = False,
        only_verified: bool = True,
        consider_only_cropped_frame: bool = False,
        ensure_yolo_compatible_size: bool = False,
        resize_image_to_square: bool = False,
        daytime: str = "daynight",
    ):
        """
        Initializes the R-LiViT dataset class with various options for data loading and processing.

        Args:
            split (str): The dataset split to use, "train", "test", or "all".
            data_dir (str): Directory of the dataset.
            transform (transforms.Compose): Transformations to apply to the images.
            original_class_ids (bool): Whether to use the original class IDs defined in supported_classes.
            only_verified (bool): Whether to use only verified annotations.
            consider_only_cropped_frame (bool): Whether to consider only the (smaller) IR frame of the images (including annotations).
            ensure_yolo_compatible_size (bool): Whether to resize images to a specific size suitable for YOLO.
            resize_image_to_square (bool): Resizes the image to 896x896, which is beneficial when models are finetuned on that size.
            daytime (str): The daytime filter, can be "day", "night", or "daynight".
        """
        super().__init__(split, data_dir, transform, original_class_ids)
        self.data_dir = data_dir
        self.rgb_ir_path = os.path.join(self.data_dir, "images")
        self.rgb_path = os.path.join(self.rgb_ir_path, "rgb")
        self.ir_path = os.path.join(self.rgb_ir_path, "thermal")
        self.annotations_path = os.path.join(self.rgb_ir_path, "annotations")
        self.sequence_metadata_path = os.path.join(self.data_dir, "sequences.xml")

        self.only_verified = only_verified
        self.consider_only_cropped_frame = consider_only_cropped_frame
        self.ensure_yolo_compatible_size = ensure_yolo_compatible_size
        self.daytime = daytime

        self.ir_frame_xymin_xymax = (296, 26, 1096, 626)  # mew cropped image size: (800, 600)

        if resize_image_to_square or self.ensure_yolo_compatible_size:
            if resize_image_to_square:
                resize_width, resize_height = (896, 896)
            else:
                resize_width, resize_height = ((1248, 704) if not self.consider_only_cropped_frame else (800, 608))

            self.transform = A.Compose(
                [A.Resize(height=resize_height, width=resize_width),
                ToTensorV2()],
                bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_ids"]),
            )
        
        self.sequence_metadata = self._load_sequence_metadata()
        self.img_label_data = self._get_imgs_labels_dict()

    def _load_sequence_metadata(self) -> Dict[str, Dict[str, str]]:
        """
        Loads sequence metadata from an XML file into a dictionary.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary where keys are sequence numbers (as strings),
            and values are dictionaries with:
                - "daytime" (str): The time of day ("day", "night", etc.).
                - "location" (str): The location where the sequence was recorded.
        """

        sequence_metadata = {}

        tree = ET.parse(self.sequence_metadata_path)
        root = tree.getroot()

        for sequence in root.findall("sequence"):
            number = sequence.find("number").text
            daytime = sequence.find("daytime").text
            location = sequence.find("location").text

            sequence_metadata[number] = {"daytime": daytime, "location": location}

        return sequence_metadata

    def _adjust_annotations_to_ir_frame(
        self, class_ids: List[int], bboxes: List[List[int]]
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Adjusts and select annotation bounding boxes to ensure they fit within the IR frame.

        Args:
            class_ids (List[int]): The list of class IDs corresponding to each bounding box.
            bboxes (List[List[int]]): The list of bounding boxes, where each box is a list [xmin, ymin, xmax, ymax].

        Returns:
            Tuple[List[int], List[List[int]]]: Adjusted class IDs and bounding boxes matching the IR frame.
        """
        xmin_crop, ymin_crop, xmax_crop, ymax_crop = self.ir_frame_xymin_xymax
        adjusted_class_ids = []
        adjusted_bboxes = []

        for bbox, class_id in zip(bboxes, class_ids):
            xmin, ymin, xmax, ymax = bbox

            new_xmin = max(xmin, xmin_crop)
            new_ymin = max(ymin, ymin_crop)
            new_xmax = min(xmax, xmax_crop)
            new_ymax = min(ymax, ymax_crop)

            if new_xmin < new_xmax and new_ymin < new_ymax:
                adjusted_class_ids.append(class_id)
                adjusted_bboxes.append(
                    [
                        new_xmin - xmin_crop,
                        new_ymin - ymin_crop,
                        new_xmax - xmin_crop,
                        new_ymax - ymin_crop,
                    ]
                )

        return adjusted_class_ids, adjusted_bboxes

    def _extract_annotations(self, root: ET.Element) -> ImageLabels:
        """
        Extracts and transforms all annotations of one image from the original format to the specified format of the data_interface

        Args:
            root (ET.Element): The root element of the annotation XML file.

        Returns:
            annotations (ImageLabels): the standardized label object with all the labels of one image
        """

        class_ids = []
        bboxes = []

        for obj in root.findall("object"):
            class_name = obj.find("name").text

            if self.original_class_ids:
                class_id = self.supported_classes[class_name]
            else:
                class_id = self.coco_labels[class_name]

            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            class_ids.append(class_id)
            bboxes.append([xmin, ymin, xmax, ymax])

        if self.consider_only_cropped_frame:
            class_ids, bboxes = self._adjust_annotations_to_ir_frame(class_ids, bboxes)

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
            f"annotations_{self.split}_{self.daytime}{'_irframe' if self.consider_only_cropped_frame else ''}{'_originalid' if self.original_class_ids else ''}.json",
        )

        if os.path.exists(data_dict_path):
            with open(data_dict_path, "r") as json_file:
                data_dict = json.load(
                    json_file, object_hook=LabelsEncoderDecoder.from_json
                )

            data_dict = {int(key): value for key, value in data_dict.items()}
            logger.info(
                f"Loaded pre-setuped dataset R-LiViT {self.split} from {data_dict_path}"
            )

        else:
            data_dict = {}

            if self.split == "all":
                splits = ["train", "test"]
            elif self.split == "train" or self.split == "test":
                splits = [self.split]
            elif self.split == "val":
                raise ValueError("No explicit val set for this dataset available.")

            sequences = []
            for split in splits:
                file_list_path = os.path.join(self.data_dir, f"{split}.txt")
                with open(file_list_path, "r") as file:
                    sequences += file.read().splitlines()

            idx = 0
            for seq in tqdm(sequences, desc=f"Setup dataset R-LiViT {self.split}"):
                if (
                    self.daytime != "daynight"
                    and self.sequence_metadata[seq]["daytime"] != self.daytime
                ):
                    continue
                for file in range(50):
                    if (file + 2) % 4 != 0:
                        continue
                    tree = ET.parse(
                        os.path.join(self.annotations_path, seq, f"{file:02}.xml")
                    )
                    root = tree.getroot()

                    if self.only_verified and root.attrib.get("verified") != "yes":
                        continue

                    data_dict[idx] = {
                        "rgb_path": None,
                        "thermal_path": None,
                        "annotations": None,
                    }
                    data_dict[idx]["rgb_path"] = os.path.join(
                        self.rgb_path, seq, f"{file:02}.png"
                    )
                    data_dict[idx]["thermal_path"] = os.path.join(
                        self.ir_path, seq, f"{file:02}.png"
                    )
                    data_dict[idx]["annotations"] = self._extract_annotations(root)
                    idx += 1

        with open(data_dict_path, "w") as json_file:
            json.dump(data_dict, json_file, cls=LabelsEncoderDecoder)

        logger.info(f"Saved dataset setup of R-LiViT {self.split} to {data_dict_path}")

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
            Tuple[str, str]: A tuple containing the train and validation split names.
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
        return {
            cls_id: RLiViT.coco_labels[cls_str]
            for cls_str, cls_id in RLiViT.supported_classes.items()
        }

    def __len__(self) -> int:
        """
        Returns:
            len (int): The number of images in the dataset.
        """
        return len(self.img_label_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Returns:
            tuple: (rgb_img, ir_img, target) where
                - rgb_img (torch.tensor) of size (3, 512, 640)
                - ir_img (torch.tensor) of size (1, 512, 640)
                - target (Dict[str, Any]) of which one entry is "imageLabels": ImageLabels
        """

        img_rgb = cv2.cvtColor(cv2.imread(self.img_label_data[idx]["rgb_path"]), cv2.COLOR_BGR2RGB)
        img_ir = cv2.imread(self.img_label_data[idx]["thermal_path"], cv2.IMREAD_GRAYSCALE)
        labels = self.img_label_data[idx]["annotations"]

        if self.consider_only_cropped_frame:
            x_min, y_min, x_max, y_max = self.ir_frame_xymin_xymax
            img_rgb = img_rgb[y_min:y_max, x_min:x_max]
            img_ir = img_ir[y_min:y_max, x_min:x_max]

        img_rgb, img_ir, labels = self.transform_images_and_annotations(img_rgb, img_ir, labels)

        target = {
            "image_id": idx,
            "imageLabels": labels,
        }

        return img_rgb, img_ir, target
