import math
import os
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from hydra import compose, initialize

from data.data_interface import ImageLabels
from models.model_interface import ImageModelPrediction


with initialize(config_path=os.path.join("..", "conf"), version_base="1.3"):
    cfg_data = compose(config_name="config").data


def plot_detections(
    rgb_batch: torch.Tensor,
    ir_batch: torch.Tensor,
    annotations: Sequence[Union[ImageLabels, ImageModelPrediction]],
    plot_modality: str = "rgb",
    min_score: float = 0.5,
    display_n_images: int = 6,
    plot_name: str = "test_plot",
) -> None:
    """
    Plots a batch of images with detections.

    Args:
        rgb_batch (torch.Tensor): Tensor containing the RGB image data (shape: Bx3xHxW).
        ir_batch (torch.Tensor): Tensor containing the infrared image data (shape: Bx1xHxW).
        annotations (Sequence[Union[ImageLabels, ImageModelPrediction]]): 
            List of object annotations for the batch, either ground truth or model predictions.
        plot_modality (str): The modality to be plotted, either "rgb" or "ir" (default: "rgb").
        min_score (float): Minimum confidence score required to display a detection (default: 0.5).
        display_n_images (int): Number of images to plot (default: 6).
        plot_name (str): Name for the saved plot file (default: "test_plot").

    """

    assert (
        rgb_batch.shape[0] >= display_n_images and ir_batch.shape[0] >= display_n_images
    ), "Can not display more images than in the current batch"

    n_row = int(math.sqrt(display_n_images))
    n_col = display_n_images // n_row

    _, ax = plt.subplots(n_row, n_col, figsize=(20, 25))

    for idx in range(display_n_images):
        img = ir_batch[idx] if plot_modality == "ir" else rgb_batch[idx]
        draw_img_with_boxes(
            img,
            annotations[idx],
            ax[idx // n_col, idx % n_col],
            min_score=min_score,
        )

    plt.tight_layout()
    plt.savefig(f"{plot_name}.jpg")
    plt.show()


def draw_img_with_boxes(
    img: torch.Tensor,
    annotations: Union[ImageLabels, ImageModelPrediction],
    ax: Axes,
    min_score: float,
) -> None:
    """
    Draws bounding boxes and labels on an image.

    Args:
        img (torch.Tensor): Image tensor (CxHxW).
        annotations (Union[ImageLabels, ImageModelPrediction]): Object annotations (ground truth or predictions).
        ax (Axes): Matplotlib axis to draw on.
        min_score (float): Minimum confidence score required to display a detection.
    """

    if img.is_cuda:
        img = img.cpu()

    img = np.transpose(img.numpy(), (1, 2, 0))

    ax.imshow(img, cmap="gray")

    with open(cfg_data.labels_path, "r") as file:
        coco_labels = eval(file.read())

    n_objects = annotations.class_ids.size(0)

    for idx in range(n_objects):
        class_id = annotations.class_ids[idx].item()
        bbox = annotations.bboxes[idx].tolist()

        text = f"{coco_labels[class_id]}" if class_id > 0 else "_non_coco_object_"

        if isinstance(annotations, ImageModelPrediction):
            text += f", {round(annotations.scores[idx].item(), 2)}"

            if annotations.scores[idx].item() < min_score:
                continue

        ax.text(
            bbox[0], bbox[1], text, fontsize=6, bbox=dict(facecolor="yellow", alpha=0.2)
        )

        rect = Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)
    ax.axis("off")


def plot_evaluation_results(
    evaluation_results: List[Tuple[str, str, Dict[str, Any]]],
    plot_name: str = "test_plot",
):
    """
    Plots evaluation results of multiple models.

    Args:
        evaluation_results (List[Tuple[str, str, Dict[str, Any]]]): 
            A list of tuples containing:
            - model (str): Model name.
            - modality (str): Input modality used (e.g., 'rgb', 'ir').
            - results (Dict[str, Any]): Computed evaluation metrics.
        plot_name (str): Name for the saved plot file (default: "test_plot").
    """

    evaluation_results_dict = [
        pd.DataFrame([dict(model=model, modality=modality, **results)])
        for model, modality, results in evaluation_results
    ]
    evaluation_results_df = pd.concat(evaluation_results_dict, ignore_index=True)

    print(evaluation_results_df)

    evaluation_results_df.set_index("model").plot(
        kind="bar", subplots=True, layout=(-1, 3), sharex=False, figsize=(12, 14)
    )
    plt.suptitle(plot_name)
    plt.savefig(f"{plot_name}.jpg")
    plt.show()
