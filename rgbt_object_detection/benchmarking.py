"""
Benchmarking Script for RGB-T Object Detection Models

This script evaluates object detection models on a selected dataset. It runs both:
- Detection evaluation using Mean Average Precision (mAP).
- Inference benchmarking to measure speed performance.

Usage:
1. Modify the `models` list to specify models and parameters.
2. Adjust dataset and its parameters as needed.
3. Run the script: `python benchmark.py`

Outputs:
- Logs are saved in the `log/` directory.
- CSV files with evaluation results are stored in `results/`.
- Optionally, performance plots are generated.

"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Type

from tqdm import tqdm
from hydra import compose, initialize

import data.data_interface as data_interface
from data import *
import models.model_interface as model_interface
from models import *
from utils.evaluation_engine import evaluate, evaluate_inference
from utils.evaluator import Evaluator
from utils.logging import save_evaluation_results, setup_logging
from utils.visualization import plot_evaluation_results

logger = logging.getLogger(__name__)

with initialize(config_path=os.path.join("conf"), version_base="1.3"):
    cfg = compose(config_name="config")


def benchmark_model_on_dataset(
    model: model_interface.Model,
    dataset: data_interface.Dataset,
    n_images_evaluate: Optional[int] = None,
    n_repetitions_inference: int = 1000,
) -> Dict[str, Any]:
    """
    Benchmarks a model on a given dataset by performing both detection and inference evaluations.

    Args:
        model (model_interface.Model): The model to be benchmarked.
        dataset (data_interface.Dataset): The dataset to run the benchmark on.
        n_images_evaluate (Optional[int]): Number of images to evaluate the model on (default: all).
        n_repetitions_inference (int): Number of repetitions for inference time evaluation (default: 1000).

    Returns:
        Dict[str, Any]: A dictionary merging the results of both detection and inference evaluations.
    """

    evaluator = Evaluator(
        considered_classes=dataset.classes(),
        group_similar_labels_mapping=dataset.group_similar_labels_mapping(),
        class_metrics=True,
    )

    results_detection = evaluate(model=model, dataset=dataset, evaluator=evaluator, n_images=n_images_evaluate)

    results_inference = evaluate_inference(model=model, dataset=dataset, n_repetitions=n_repetitions_inference)

    results = results_detection | results_inference

    return results


def run_benchmarking(
    models: List[Tuple[str, Type[model_interface.Model], Dict[str, Any]]],
    dataset: data_interface.Dataset,
    save_results_csv: bool = True,
    plot_results: bool = True,
    benchmarking_infos: Dict[str, Any] = {},
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Runs benchmarking on multiple models on a given dataset by performing both detection and inference evaluations.

    Args:
        models (List[Tuple[str, Type[model_interface.Model], Dict[str, Any]]]): 
            A list of tuples where:
            - First element (str): Model name.
            - Second element (Type[model_interface.Model]): Model class.
            - Third element (Dict[str, Any]): Model initialization parameters.
        dataset (data_interface.Dataset): The dataset to run the benchmarks on.
        save_results_csv (bool): Whether to save benchmarking results to a CSV file (default: True).
        plot_results (bool): Whether to generate and display plots of the results (default: True).
        benchmarking_infos (Optional[Dict[str, Any]]): Additional metadata for benchmarking (default: None).

    Returns:
        List[Tuple[str, str, bool, Dict[str, Any]]]: 
            A list of tuples where:
            - First element (str): Model name.
            - Second element (str): Modality used (e.g., 'rgb', 'ir').
            - Third element (bool): Whether the model was fine-tuned.
            - Fourth element (Dict[str, Any]): Benchmarking results.
    """

    results = []
    start_time = datetime.now()
    for model_name, model_class, model_params in tqdm(
        models, desc="Model Benchmarking"
    ):
        model = model_class(**model_params)
        modality = model.modality
        fine_tuned = model.pre_trained_weights is not None

        logger.info(f"Starting benchmarking for model: {model_name}, {model}")
        model_result = benchmark_model_on_dataset(model, dataset)
        logger.info(
            f"Finished benchmarking for model: {model_name}, modality: {modality}, results: {model_result}"
        )

        results.append((model_name, modality, fine_tuned, model_result))

        if save_results_csv:
            csv_file_path = save_evaluation_results("benchmarking", results, benchmarking_infos, start_time)
            logger.info(f"Saved benchmarking results to: {csv_file_path}")

    if plot_results:
        plot_evaluation_results(results)

    return results


def main() -> None:
    setup_logging("benchmarking")

    models: List[Tuple[str, Type[model_interface.Model]], Dict[str, Any]] = [
        # List models to be benchmarked here in the format:
        # (model_name (str), model_class (Type[model_interface.Model]]), model_params (Dict[str, Any])),
        # e.g. ("YOLOv8_RGBIR", YOLOv8, {"modality": "rgbir", "modality_rgbir_use_individual_models": True}),

        # ("RTDETR_RGB", RTDETR, {"modality": "rgb"}),
        # ("FasterRCNN_RGB", FasterRCNN, {"modality": "rgb"}),
        ("YOLOv8_RGBIR", YOLOv8, {"modality": "rgbir", "modality_rgbir_use_individual_models": True}),
    ]

    # For evaluation on RLiViT -> For other datasets adjust dataset parameters accordingly
    # (i.e. ensure_yolo_compatible_size, resize_image_to_square, and consider_only_cropped_frame are only implemented in RLiViT)
    dataset_class: data_interface.Dataset = RLiViT # Select dataset for the benchmarking
    split = dataset_class.standard_dataset_split_names()[1] # Overwrite, if a different set than the standard validation is desired
    for daytime in ["day", "night", "daynight"]:
        consider_only_cropped_frame = True
        ensure_yolo_compatible_size = True
        resize_image_to_square = False
        benchmarking_infos = {
            "dataset": dataset_class.__name__,
            "split": split,
            "daytime": daytime,
            "consider_only_cropped_frame": consider_only_cropped_frame,
            "ensure_yolo_compatible_size": ensure_yolo_compatible_size,
            "resize_image_to_square": resize_image_to_square,
        }
        dataset = dataset_class(split=split,
                                daytime=daytime,
                                consider_only_cropped_frame=consider_only_cropped_frame,
                                ensure_yolo_compatible_size=ensure_yolo_compatible_size,
                                resize_image_to_square=resize_image_to_square)

        logger.info(f"Starting benchmarking script with {dataset}")
        res = run_benchmarking(
            models=models,
            dataset=dataset,
            plot_results=False,
            benchmarking_infos=benchmarking_infos,
        )
        logger.info(f"Finished benchmarking script, results: {res}")


if __name__ == "__main__":
    main()
