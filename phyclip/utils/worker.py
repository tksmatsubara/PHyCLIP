import json
import os
import subprocess

from loguru import logger
from torch.utils.tensorboard.writer import SummaryWriter


def tensorboard_worker(input_queue, output_dir):
    """Worker for asynchronous TensorBoard logging"""
    tboard = SummaryWriter(log_dir=output_dir)

    while True:
        try:
            data = input_queue.get()
            if data is None:  # termination signal
                break

            iteration = data["iteration"]

            # Process hierarchy visualization data
            match data.get("type", ""):
                case "scalar":
                    log_data = data.get("scalar", {})
                    for name, value in log_data.items():
                        tboard.add_scalar(name, value, iteration)
                    continue

        except Exception as e:
            logger.error(f"Error in tensorboard worker: {str(e)}")

    tboard.close()


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def evaluation_worker(input_queue, tensorboard_input_queue):
    """
    input_queue: multiprocessing.Queue, receives (checkpoint_path, config_path, iteration)
    tensorboard_input_queue: multiprocessing.Queue, receives log dict for tensorboard_worker
    """
    logger.info("Evaluation worker started")

    while True:
        job = input_queue.get()
        if job is None:
            break
        checkpoint_path, train_config_path, iteration = job
        try:
            for config in [
                "eval_zero_shot_classification.py",
                "eval_zero_shot_retrieval.py",
                "eval_hierarchical_metrics.py",
            ]:
                logger.info(
                    f"Running evaluation for {checkpoint_path} with config {config}"
                )
                result = subprocess.run(
                    [
                        "python",
                        "scripts/evaluate.py",
                        "--checkpoint-path",
                        checkpoint_path,
                        "--train-config",
                        train_config_path,
                        "--config",
                        os.path.join("configs", config),
                    ],
                    capture_output=True,  # Capture stdout and stderr
                    text=True,
                )

                if result.returncode != 0:
                    logger.error(
                        f"evaluate.py failed with exit code {result.returncode}"
                    )
                    logger.error(f"evaluate.py stdout: {result.stdout}")
                    logger.error(f"evaluate.py stderr: {result.stderr}")
                    metrics = {}
                else:
                    # Determine the experiment directory to find evaluate.json
                    # This logic mirrors how evaluate.py determines exp_dir
                    exp_dir = os.path.dirname(os.path.abspath(checkpoint_path))
                    json_path = os.path.join(exp_dir, "evaluate.json")

                    try:
                        with open(json_path, "r") as f:
                            all_results = json.load(f)
                            metrics = all_results[-1]["results"] if all_results else {}
                    except Exception as e:
                        print(f"Failed to read or parse evaluate.json: {e}")
                        metrics = {}

                logger.info(f"Parsed metrics: {metrics}")

                # Extract task name from config file name
                # Example: eval_zero_shot_classification.py → zero_shot_classification
                if config.startswith("eval_") and config.endswith(".py"):
                    task_name = config[len("eval_") : -len(".py")]
                else:
                    task_name = os.path.splitext(config)[0]

                # Flatten nested dict
                flat_metrics = flatten_dict(metrics)
                # Create log data for tensorboard_worker (add "evaluate/{task_name}/" prefix to keys)
                eval_metrics = {
                    f"evaluate/{task_name}/{k}": v for k, v in flat_metrics.items()
                }
                log_data = {
                    "iteration": iteration,
                    "scalar": eval_metrics,
                    "type": "scalar",
                }
                tensorboard_input_queue.put(log_data)

        except Exception as e:
            logger.error(f"Evaluation failed for {checkpoint_path}: {e}")

    logger.info("Evaluation worker finished")
