# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

"""
Evaluate a trained model using implementations from `phyclip.evaluation` module.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf

from phyclip.config import LazyConfig, LazyFactory
from phyclip.utils.checkpointing import CheckpointManager

parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA(
    "--config",
    help="Path to an evaluation config file (.py). If not provided, all evaluations (classification, retrieval, hierarchical) will be run.",
)
_AA(
    "--checkpoint-path",
    help="Path to checkpoint of a trained HyCoCLIP/MERU/CLIP model.",
)
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")


def main(_A: argparse.Namespace):
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create evaluation and training config objects.
    _C_TRAIN = LazyConfig.load(_A.train_config)

    # If no config is provided, run all evaluations
    if _A.config is None:
        config_paths = [
            "configs/eval_zero_shot_classification.py",
            "configs/eval_zero_shot_retrieval.py",
            "configs/eval_hierarchical_metrics.py",
        ]

        all_results = {}
        for config_path in config_paths:
            logger.info(f"Running evaluation with config: {config_path}")
            _C = LazyConfig.load(config_path)
            logger.info(OmegaConf.to_yaml(_C))

            # Get evaluation task name for results key
            eval_task_name = (
                os.path.basename(config_path).replace("eval_", "").replace(".py", "")
            )

            # Run evaluation and save individual results
            result = run_single_evaluation(_A, _C, _C_TRAIN, device, eval_task_name)
            save_individual_result(_A, _C, result, eval_task_name)
            all_results.update(result)

        # Output combined results
        output_combined_results(_A, all_results)

    else:
        _C = LazyConfig.load(_A.config)
        logger.info(OmegaConf.to_yaml(_C))
        eval_task_name = (
            os.path.basename(_A.config).replace("eval_", "").replace(".py", "")
        )
        result = run_single_evaluation(_A, _C, _C_TRAIN, device, eval_task_name)
        save_individual_result(_A, _C, result, eval_task_name)
        output_combined_results(_A, result)


def run_single_evaluation(_A, _C, _C_TRAIN, device, eval_task_name):
    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    logger.info(f"Evaluating checkpoint in {_A.checkpoint_path}...")

    # Create a fresh model and evaluator for every checkpoint, so the evaluator
    # is free to modify the model weights (e.g. remove projection layers).
    evaluator = instantiate(_C.evaluator)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()

    # Explicitly move model to device
    model = model.to(device)
    logger.info(f"Model moved to device: {next(model.parameters()).device}")

    CheckpointManager(model=model).load(_A.checkpoint_path)

    results_dict = evaluator(model)

    # Convert Tensor types to serializable format using item()
    def to_serializable(val):
        if isinstance(val, torch.Tensor) and val.numel() == 1:
            return val.item()
        if isinstance(val, dict):
            return {k: to_serializable(v) for k, v in val.items()}
        return val

    results_dict = {k: to_serializable(v) for k, v in results_dict.items()}

    # Add evaluation task prefix to results
    prefixed_results = {f"{eval_task_name}_{k}": v for k, v in results_dict.items()}

    return prefixed_results


def save_individual_result(_A, _C, results_dict, eval_task_name):
    # Remove prefix from results for individual saving
    original_results = {}
    prefix = f"{eval_task_name}_"
    for k, v in results_dict.items():
        if k.startswith(prefix):
            original_results[k[len(prefix) :]] = v
        else:
            original_results[k] = v

    # Additional information: evaluation task name
    eval_task = None
    if hasattr(_C.evaluator, "_target_"):
        eval_task = _C.evaluator._target_

    save_obj = {
        "eval_task": eval_task,
        "checkpoint_path": _A.checkpoint_path,
        "results": original_results,
    }

    # Save results as JSON in the experiment directory (append as list)
    exp_dir = os.path.dirname(os.path.abspath(_A.checkpoint_path))
    json_path = os.path.join(exp_dir, "evaluate.json")
    try:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []
            if isinstance(existing, dict):
                existing = [existing]
            elif not isinstance(existing, list):
                existing = []
            existing.append(save_obj)
            with open(json_path, "w") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        else:
            with open(json_path, "w") as f:
                json.dump([save_obj], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evaluation results to {json_path}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results to {json_path}: {e}")


def output_combined_results(_A, results_dict):
    # If any value is dict type, treat as multi-dataset/metrics format
    if any(isinstance(v, dict) for v in results_dict.values()):
        # Collect metric names from dict values
        all_metrics = set()
        for v in results_dict.values():
            if isinstance(v, dict):
                all_metrics.update(v.keys())
        all_metrics = sorted(all_metrics)
        header = "dataset," + ",".join(all_metrics)
        lines = []
        for dset, metrics in results_dict.items():
            row = [dset]
            if isinstance(metrics, dict):
                for m in all_metrics:
                    val = metrics.get(m, float("nan"))
                    row.append(f"{val:.3f}" if isinstance(val, float) else str(val))
            else:
                # If not dict type, use as is
                row += [str(metrics)] + ["" for _ in range(len(all_metrics) - 1)]
            lines.append(",".join(row))
        numbers = "\n".join(lines)
    else:
        header = ",".join(results_dict.keys())
        numbers = ",".join(
            [
                f"{num:.3f}" if isinstance(num, float) else str(num)
                for num in results_dict.values()
            ]
        )

    logger.info(f"copypaste: {_A.checkpoint_path}")
    logger.info(f"\ncopypaste below:\n{header}\n{numbers}")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
