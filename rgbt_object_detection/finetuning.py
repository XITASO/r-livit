"""
Finetuning Script for Object Detection Models

This script finetunes a specified object detection model on a given dataset.

Usage:
1. Modify the dataset and model parameters as needed.
2. Run the script from the command line, e.g.:

python finetuning.py --dataset RLiViT --model YOLOv8 --model_size M --modality rgb --seed 42 --name finetune_1

Arguments:
- `--dataset` (str): Name of the dataset to use (e.g., `RLiViT`).
- `--model` (str): Name of the model to use (e.g., `FasterRCNN`).
- `--model_size` (str): Model size specification (e.g., `L`).
- `--modality` (str): Modality type (`rgb`, or `ir`).
- `--seed` (int): Random seed for reproducibility.
- `--name` (str): Name for the experiment (used in logs and saved files).

Outputs:
- Training progress and results are logged.
- The finetuned model is saved in the configured results directory.
"""

import argparse
import logging
import os

from hydra import compose, initialize

from utils.logging import setup_logging, set_seeds
from data import * 
from models import *

logger = logging.getLogger(__name__)

with initialize(config_path=os.path.join("conf"), version_base="1.3"):
    cfg = compose(config_name="config")


def main(dataset_name: str, model_name: str, model_size: str, modality: str, seed: int, name: str) -> None:
    """
    Finetunes a specified model on a given dataset.

    Args:
        dataset_name (str): Name of the dataset to use (must match a class in `globals()`).
        model_name (str): Name of the model to use (must match a class in `globals()`).
        model_size (str): The model size specification (e.g., "small", "large").
        modality (str): The modality type (e.g., "rgb", "ir", "rgbir").
        seed (int): The random seed for reproducibility.
        name (str): The name of the experiment (used for logging and saving results).
    """

    setup_logging("finetuning")
    set_seeds(seed)

    dataset_class = globals().get(dataset_name)

    splits = dataset_class.standard_dataset_split_names()

    model_class = globals().get(model_name)
    model = model_class(modality=modality, model_size=model_size)

    logger.info(f"Starting finetuning for {model_name} (modality: {modality}, size {model_size}) with {dataset_name}, seed {seed}")

    model.finetune(dataset_name, dataset_class, splits, seed, name)

    logger.info(f"Finished finetuning for {model_name} (modality: {modality}, size {model_size}) with {dataset_name}, seed {seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a model on a specified dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--model_size", type=str, required=True, help="Size of the model to use.")
    parser.add_argument("--modality", type=str, required=True, help="Modality to use for the model.")
    parser.add_argument("--seed", type=int, required=True, help="Seed for the finetuning run.")
    parser.add_argument("--name", type=str, required=True, help="Experiment Name.")
    args = parser.parse_args()
    main(args.dataset, args.model, args.model_size, args.modality, args.seed, args.name)
