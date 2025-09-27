import os
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize
from torch.utils.data import DataLoader
from tqdm import tqdm

import data.data_interface as data_interface
from data.data_interface import collate_fn_benchmarking
import models.model_interface as model_interface
from utils.evaluator import Evaluator


with initialize(config_path=os.path.join("..", "conf"), version_base="1.3"):
    cfg_setup = compose(config_name="config").setup


def evaluate(
    model: model_interface.Model,
    dataset: data_interface.Dataset,
    evaluator: Evaluator,
    n_images: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Runs evaluation based on model predictions using provided data and evaluator.

    Args:
        model (model_interface.Model): An instance of a model to evaluate.
        dataset (data_interface.Dataset): Dataset to be considered for the evaluation.
        evaluator (Evaluator): An instance of an evaluator to compute the metrics.
        n_images (Optional[int]): Number of images to use for evaluation. If None, the entire dataset is used.
    Returns:
        computing_results (Dict[str, torch.Tensor]): A dict with all the computed metrics:
    """

    model.eval()
    model.to(cfg_setup.device)

    evaluator.to(cfg_setup.device)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg_setup.evaluation_batch_size,
        collate_fn=collate_fn_benchmarking,
        shuffle=False,
        pin_memory=True,
    )

    if n_images is not None:
        n_batches = int(np.ceil(n_images / cfg_setup.evaluation_batch_size))
    else:
        n_batches = len(data_loader)

    tqdm_object = tqdm(data_loader, total=n_batches, desc="Model Evaluation")

    with torch.no_grad():
        for i, (rgb_batch, ir_batch, targets) in enumerate(tqdm_object):
            if i == n_batches:
                break
            rgb_batch = rgb_batch.to(cfg_setup.device)
            ir_batch = ir_batch.to(cfg_setup.device)

            input_data = (rgb_batch, ir_batch)

            model_input = model.pre_process(input_data)
            output = model.predict(model_input)
            predictions = model.post_process(output)

            for prediction in predictions:
                prediction.to(cfg_setup.device)

            for target in targets:
                target.to(cfg_setup.device)

            evaluator.update(predictions, targets)

        computing_results = evaluator.compute()

    return computing_results


def evaluate_inference(
    model: model_interface.Model,
    dataset: data_interface.Dataset,
    n_repetitions: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Measure the inference time of a model. For this cuda is needed!

    Args:
        model (model_interface.Model): An instance of a model to evaluate.
        dataset (data_interface.Dataset): Dataset to be considered for the evaluation.
        n_repetitions (int): Number of runs for evaluation, default is the whole provided dataset.

    Returns:
        Dict[str, Any]: A dictionary containing inference time statistics:
            - "device" (str): The CUDA device name.
            - "inference_time_mean" (float): Mean inference time (ms).
            - "inference_time_std" (float): Standard deviation of inference time (ms).
    """

    if cfg_setup.device == "cpu":
        warnings.warn("CUDA is not available. Inference time will not be evaluated.")
        return {}

    model.eval()
    model.to(cfg_setup.device)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn_benchmarking,
        shuffle=True,
        pin_memory=True,
    )

    n_repetitions = min(
        len(data_loader) if n_repetitions is None else n_repetitions, len(data_loader)
    )

    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    timings = np.zeros((n_repetitions, 1))

    warmup_input = next(iter(data_loader))
    warmup_input = (
        warmup_input[0].to(cfg_setup.device),
        warmup_input[1].to(cfg_setup.device),
    )

    for _ in range(10):
        _ = model.predict(model.pre_process(warmup_input))

    tqdm_object = tqdm(data_loader, total=n_repetitions, desc="Inference Evaluation")

    with torch.no_grad():
        for i, (rgb_batch, ir_batch, _) in enumerate(tqdm_object):
            if i == n_repetitions:
                break

            rgb_batch = rgb_batch.to(cfg_setup.device)
            ir_batch = ir_batch.to(cfg_setup.device)
            input_data = (rgb_batch, ir_batch)
            model_input = model.pre_process(input_data)

            starter.record()
            _ = model.predict(model_input)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time

    mean_syn = np.mean(timings)
    std_syn = np.std(timings)

    inference_results = {
        "device": torch.cuda.get_device_name(cfg_setup.device),
        "inference_time_mean": mean_syn,
        "inference_time_std": std_syn,
    }

    return inference_results
