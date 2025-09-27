import datetime
import os
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from hydra import compose, initialize

from ultralytics.data.augment import (
    Albumentations,
    Compose,
    CopyPaste,
    Format,
    LetterBox,
    MixUp,
    Mosaic,
    RandomHSV,
    RandomFlip,
    RandomPerspective,
)
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils.instance import Instances

with initialize(config_path=os.path.join("..", "..", "conf"), version_base="1.3"):
    cfg = compose(config_name="config")
    cfg_setup = cfg.setup
    cfg_data = cfg.data


def finetune_rtdetr(
    model: nn.Module,
    datasets: Dict[str, torch.utils.data.Dataset],
    dataset_name: str,
    modality: str,
    model_size: str,
    seed: int,
    run_name: str
) -> None:
    """
    Finetunes RT-DETR models leveraging the ultralytics training implementation.

    Args:
        model (nn.Module): The RT-DETR model to be finetuned.
        datasets (Dict[str, Dataset]): A dictionary containing the training ("train") and evaluation ("eval") datasets.
        dataset_name (str): The name of the dataset.
        modality (str): The input modality ('rgb' or 'ir').
        model_size (str): The model size identifier (e.g., 'L', 'X').
        seed (int): Random seed for reproducibility.
        name (str): Name of the training run for logging and saving model checkpoints.
    """

    finetune_path = os.path.join(cfg_data.results_path, "finetuning", "rtdetr")

    override_args = {
        "epochs": 150,
        "batch": 8,
        "lr0": 0.001,
        "imgsz": 896,
        "optimizer": "AdamW",
        "plots": False,
        "project": finetune_path,
        "amp": False,
        "translate": 0.0,
        "scale": 0.0,
        "seed": seed,
        "name": run_name
    }

    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    final_weights_name = f"state_dict-{dataset_name.lower()}-{modality}-size_{model_size.lower()}-{seed}-{start_time}-{run_name}.pt"

    trainer = RTDETRTrainerWrapper(
        overrides=override_args,
        model=model,
        datasets=datasets,
        modality=modality,
        final_weights_name=final_weights_name,
    )

    trainer.model = trainer.get_model(weights=None, cfg=model.yaml)
    model = trainer.model

    trainer._do_train(0 if next(model.parameters()).device == "cpu" else 1)


class DatasetWrapper(Dataset):
    """
    Wrapper that transforms our dataset format to the required ultralytics YOLODataset format.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, modality: str, mode: str, args: dict):
        self.dataset = dataset
        self.dataset.get_image_and_label = lambda index: self.__getitem__(index, get_image_and_label=True)
        self.modality = modality
        self.train_mode = mode == "train"
        self.args = args
        self.imgsz = self.args.imgsz
        self.resized_shape = (self.imgsz, self.imgsz)
        self.transforms = self._build_transforms()

    def _build_transforms(self) -> Compose:
        """
        Builds the transformation pipeline for data augmentation.

        Returns:
            Compose: A composed set of transformations.
        """
        hyp = self.args
        if self.train_mode:
            pre_transform = Compose(
                [
                    MosaicWrapper(self.dataset, imgsz=self.imgsz, p=hyp.mosaic),
                    CopyPaste(p=hyp.copy_paste),
                    RandomPerspective(
                        degrees=hyp.degrees,
                        translate=hyp.translate,
                        scale=hyp.scale,
                        shear=hyp.shear,
                        perspective=hyp.perspective,
                        pre_transform=LetterBox(new_shape=self.resized_shape),
                    ),
                ]
            )
            transforms = Compose(
                [
                    pre_transform,
                    MixUp(self.dataset, pre_transform=pre_transform, p=hyp.mixup),
                    Albumentations(p=1.0),
                    RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                    RandomFlip(direction="vertical", p=hyp.flipud),
                    RandomFlip(direction="horizontal", p=hyp.fliplr),
                ]
            )
        else:
            transforms = Compose([])

        transforms.append(
            Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=False,
                return_keypoint=False,
                batch_idx=True,
                # bgr=1.0 - hyp.bgr if self.train_mode else 1.0,
            )
        )

        return transforms

    def __len__(self) -> int:
        return self.dataset.__len__()

    def __getitem__(self, index: int, get_image_and_label: bool = False) -> Dict[str, Any]:
        """
        Retrieves an item from the dataset and applies transformations.

        Args:
            index (int): The index of the item to retrieve.
            get_image_and_label (bool, optional): If True, returns the raw image and label. Defaults to False.

        Returns:
            Dict[str, Any]: The transformed dataset sample.
        """
        img_rgb, img_ir, target = self.dataset.__getitem__(index)

        if self.modality == "rgb":
            img = img_rgb
        elif self.modality == "ir":
            img = img_ir.repeat(3, 1, 1)

        label = {}
        img = np.transpose(img.numpy(), (1, 2, 0)) * 255
        label["ori_shape"] = img.shape[:2]
        label["resized_shape"] = (
            self.resized_shape if self.train_mode else label["ori_shape"]
        )
        label["img"] = cv2.resize(
            img, label["resized_shape"][::-1], interpolation=cv2.INTER_LINEAR
        )

        ratio_pad_height = label["resized_shape"][0] / label["ori_shape"][0]
        ratio_pad_width = label["resized_shape"][1] / label["ori_shape"][1]
        label["ratio_pad"] = (ratio_pad_height, ratio_pad_width)
        label["im_file"] = "dummy"

        bboxes = target["imageLabels"].bboxes.numpy().astype(float)
        bboxes[:, [0, 2]] *= ratio_pad_width
        bboxes[:, [1, 3]] *= ratio_pad_height
        segments_dummy = np.empty((0, 1, 2)) # Segments are not considered but the ultralytics code
        # needs an object here, it does not properly support None here

        label["instances"] = Instances(
            bboxes=bboxes,
            bbox_format="xyxy",
            segments=segments_dummy,
            normalized=False,
        )
        label["cls"] = target["imageLabels"].class_ids.numpy() -1


        if get_image_and_label:
            return label
        else:
            return self.transforms(label)


class MosaicWrapper(Mosaic):
    def get_indexes(self, buffer=False):
        # Overrides the get_indexes function so no buffer is required
        return super().get_indexes(buffer=buffer)


class RTDETRTrainerWrapper(RTDETRTrainer):
    def __init__(
        self,
        overrides: Dict[str, Any],
        model: nn.Module,
        datasets: Dict[str, torch.utils.data.Dataset],
        modality: str,
        final_weights_name: str,
    ):
        """
        Wrapper of the ultralytics DetectionTrainer to bring everything to the correct format and ovveride functions where needed.

        Args:
            overrides (Dict[str, Any]): Configuration overrides for the training process.
            model (nn.Module): The YOLO model.
            datasets (Dict[str, torch.utils.data.Dataset]): A dictionary containing training and evaluation datasets.
            modality (str): The input modality ('rgb' or 'ir').
            final_weights_name (str): The name of the file to save final model weights.
        """
        super().__init__(overrides=overrides)
        self.wdir = Path(self.save_dir).parent
        self.best = self.wdir / final_weights_name
        self.model = model
        self.datasets = datasets

        self.data = {}
        class_id_to_index = {
            class_id: index
            for index, class_id in enumerate(
                datasets["train"].supported_classes.values()
            )
        }
        self.data["names"] = {
            class_id_to_index[v]: k
            for k, v in datasets["train"].supported_classes.items()
        }
        self.data["nc"] = len(self.data["names"])

        self.modality = modality

    def final_eval(self):
        pass  # Ignore final evaluation

    def get_dataset(self):
        return (None, None)  # Manually set the datasets and skip the ultralytics build-up process

    def get_dataloader(self, dataset_path: str, batch_size: int, rank: int, mode: str) -> InfiniteDataLoader:
        # Overrides the dataloader build-up process and directly returns the dataloader in the required format
        dataset = DatasetWrapper(self.datasets[mode], self.modality, mode, self.args)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(mode == "train"),
            collate_fn=RTDETRDataset.collate_fn,
            num_workers=4 if not cfg_setup.device == "cpu" else 0,
        )
