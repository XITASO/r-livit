import logging
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize

with initialize(config_path=os.path.join("..", "conf"), version_base="1.3"):
    cfg_data = compose(config_name="config").data


def set_seeds(seed: int) -> None:
    """
    Set seed for reproducibility in experiments using numpy, torch, CUDA, and onnxruntime.
    
    Parameters:
    seed (int): The seed value to set for all libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ['ORT_SEED'] = str(seed)

def setup_logging(name: str, level: int = logging.INFO) -> None:
    """
    Creates a logger with the specified configuration.

    Args:
        name (str): Name of the logger. It's assumed to be the name of the script
        level (int): Level of the logger. It's assumed to be the logging level
    """
    formatted_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_name = f"{formatted_date_time}_{name}.log"
    log_file_path = os.path.join(cfg_data.log_path, "log", log_file_name)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(filename)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler()],
        force=True,
    )


def save_evaluation_results(
    name: str,
    evaluation_results: List[Tuple[str, str, Dict[str, Any]]],
    benchmarking_infos: Dict[str, Any] = {},
    time: Optional[datetime] = None,
) -> str:
    """
    Saves evaluation results as a CSV file in the results folder.

    Args:
        name (str): Name identifier for the evaluation results file.
        evaluation_results (List[Tuple[str, str, bool, Dict[str, Any]]]): 
            A list of tuples containing:
            - model (str): Model name.
            - modality (str): Input modality used (e.g., 'rgb', 'ir', 'rgbir').
            - fine_tuned (bool): Whether the model was fine-tuned.
            - ...
            - results (Dict[str, Any]): Computed benchmark results.
        benchmarking_infos (Optional[Dict[str, Any]]): Additional benchmark metadata (default: None).
        time (Optional[datetime]): Timestamp for file naming (default: current time).

    Returns:
        str: Path where the CSV file is stored.
    """
    if time is None:
        time = datetime.now()
    formatted_date_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    csv_file_name = f"{formatted_date_time}_{name}.csv"
    csv_file_path = os.path.join(cfg_data.results_path, "results", csv_file_name)
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    evaluation_results_dict = [
        pd.DataFrame(
            [
                dict(
                    model=model,
                    modality=modality,
                    fine_tuned=fine_tuned,
                    **benchmarking_infos,
                    **results,
                )
            ]
        )
        for model, modality, fine_tuned, results in evaluation_results
    ]
    evaluation_results_df = pd.concat(evaluation_results_dict, ignore_index=True)

    evaluation_results_df.to_csv(csv_file_path, index=False)

    return csv_file_path
