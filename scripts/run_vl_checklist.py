#!/usr/bin/env python3
"""
VL-Checklist evaluation script for PHyCLIP models.

This script evaluates PHyCLIP, MERU, HyCoCLIP, and CLIP baseline models
using the VL-Checklist evaluation framework.
"""

import argparse
import os

from vl_checklist.evaluate import Evaluate

from phyclip.evaluation.vl_checklist_wrapper import VLChecklistWrapper


def main():
    parser = argparse.ArgumentParser(description="VL-Checklist evaluation for PHyCLIP")

    # Model configuration
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        required=True,
        help="Path to the model configuration file",
    )

    # VL-Checklist configuration
    parser.add_argument(
        "--vl-checklist-config",
        type=str,
        default="configs/vl_checklist_config.yaml",
        help="Path to VL-Checklist configuration file",
    )

    # Hardware configuration
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to run the model on"
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.train_config):
        raise FileNotFoundError(f"Config file not found: {args.train_config}")

    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    if not os.path.exists(args.vl_checklist_config):
        print(f"Warning: VL-Checklist config not found: {args.vl_checklist_config}")
        print(
            "You may need to create this config file or use the default VL-Checklist configuration."
        )

    # Create model wrapper
    model = VLChecklistWrapper(
        checkpoint_path=args.checkpoint_path,
        config_path=args.train_config,
        device=args.device,
        batch_size=32,
    )

    # Run VL-Checklist evaluation
    print("Starting VL-Checklist evaluation...")
    evaluator = Evaluate(config_file=args.vl_checklist_config, model=model)
    evaluator.start()

    # Print completion message
    print("Evaluation completed! Check VL-Checklist output for results.")


if __name__ == "__main__":
    main()
