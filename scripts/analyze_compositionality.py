import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from phyclip import lorentz as L
from phyclip.config import LazyConfig, LazyFactory
from phyclip.models import HyCoCLIP, PHyCLIP
from phyclip.tokenizer import Tokenizer
from phyclip.utils.checkpointing import CheckpointManager

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.eval_compositionality_test import COMPOSITION_TEST_CASES

parser = argparse.ArgumentParser(
    description="Compare norms of concept compositions across subspaces"
)
_AA = parser.add_argument
_AA(
    "--checkpoint-path",
    required=True,
    help="Path to checkpoint of a trained PHyCLIP model.",
)
_AA(
    "--train-config",
    required=True,
    help="Path to train config (.yaml/py) for given checkpoint.",
)

_AA("--seed", type=int, default=42, help="Seed for reproducibility.")

COLORS = [
    "#ff7f0e",
    "#1f77b4",
    "#2ca02c",
]  # blue, orange, green (matplotlib default colors)
Y_LIM_MARGIN = 0.9  # margin for y-axis lower limit


def setup_model(checkpoint_path, train_config_path):
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    _C_TRAIN = LazyConfig.load(train_config_path)
    logger.info(f"Train config loaded from: {train_config_path}")
    logger.info(f"Checkpoint path: {checkpoint_path}")

    tokenizer = Tokenizer()
    logger.info(f"Loading model from checkpoint: {checkpoint_path}...")

    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(checkpoint_path)
    model = model.eval()

    is_phyclip = isinstance(model, PHyCLIP)
    if not is_phyclip and not isinstance(model, HyCoCLIP):
        raise ValueError("This script only supports PHyCLIP or HyCoCLIP models.")

    return model, tokenizer, is_phyclip


def collect_norm_data(model, tokenizer, categories, is_phyclip):
    """Collect norm data for each category"""
    logger.info("Collecting norms for dog, car, and 'dog and car' embeddings...")

    category_norms_data = {}
    num_subspaces = model.num_subspaces if is_phyclip else 1

    for category_name, prompts in categories.items():
        logger.info(f"Processing category: {category_name}")
        all_norms_for_category = []

        for prompt in tqdm(prompts, desc=f"Processing {category_name}", leave=False):
            with torch.inference_mode():
                tokens = tokenizer([prompt])
                text_feats = model.encode_text(tokens, project=True)

                if is_phyclip:
                    text_subspace_feats = text_feats.view(
                        -1, model.num_subspaces, model.subspace_dim
                    )
                    batch_text_norms = L.get_space_norm(text_subspace_feats)
                else:  # HyCoCLIP case
                    batch_text_norms = L.get_space_norm(text_feats.unsqueeze(1))

                all_norms_for_category.append(batch_text_norms.squeeze().cpu().numpy())

        if all_norms_for_category:
            category_norms_data[category_name] = np.array(all_norms_for_category)

    if not category_norms_data:
        raise ValueError("No norm data was collected. Aborting.")

    return category_norms_data, num_subspaces


def calculate_difference_based_alphas(category_norms_data, num_subspaces, top_k=5):
    """Highlight only 2 spaces: the space with max norm for concept1 and the space with max norm for concept2"""
    # Get category names (use first 2 categories)
    category_names = list(category_norms_data.keys())
    if len(category_names) < 2:
        return [0.7] * num_subspaces

    concept1_name = category_names[0]
    concept2_name = category_names[1]

    # Get data for concept1 and concept2
    concept1_means = [
        np.mean(category_norms_data[concept1_name][:, i])
        for i in range(num_subspaces)
        if category_norms_data[concept1_name][:, i].size > 0
    ]
    concept2_means = [
        np.mean(category_norms_data[concept2_name][:, i])
        for i in range(num_subspaces)
        if category_norms_data[concept2_name][:, i].size > 0
    ]

    if not concept1_means or not concept2_means:
        return [0.1] * num_subspaces

    # Index of space with largest norm for concept1
    concept1_max_idx = np.argmax(concept1_means)

    # Index of space with largest norm for concept2
    concept2_max_idx = np.argmax(concept2_means)

    # Initialize alpha values (default is light)
    alphas = [0.1] * len(concept1_means)

    # Make the max norm spaces for concept1 and concept2 darker
    alphas[concept1_max_idx] = 1.0
    alphas[concept2_max_idx] = 1.0

    return alphas


def create_overlapped_comparison_plot(
    category_norms_data, num_subspaces, plot_output_dir
):
    logger.info("Generating overlapped comparison plot...")

    plt.figure(figsize=(10, 6))

    alphas = calculate_difference_based_alphas(category_norms_data, num_subspaces)
    highlighted_indices = [i for i, alpha in enumerate(alphas) if alpha > 0.5]

    # Control plot order: composition, concept1, concept2
    category_list = list(category_norms_data.items())
    # Change plot order (composition first)
    plot_order = [2, 0, 1]  # composition, concept1, concept2

    # Create dummy plots for legend to maintain original order
    for legend_idx, (category_name, _) in enumerate(category_list):
        # Set marker type based on legend order
        if legend_idx == 0:  # concept1
            marker = "^"  # triangle
        elif legend_idx == 1:  # concept2
            marker = "o"  # circle
        else:  # composition
            marker = "s"  # square

        # Dummy plot for legend
        plt.plot(
            [],  # empty data
            [],  # empty data
            marker,
            color=COLORS[legend_idx],
            markersize=8,
            label=f"{category_name.replace('_', ' ')}",
            alpha=1.0,
        )

    # Plot actual data in plot order
    for plot_idx in plot_order:
        if plot_idx < len(category_list):
            category_name, norms_matrix = category_list[plot_idx]
            space_data, space_labels = prepare_space_data(norms_matrix, num_subspaces)

            if space_data:
                means = [np.mean(data) for data in space_data]

                alphas = calculate_difference_based_alphas(
                    category_norms_data, num_subspaces
                )

                # Set marker type based on legend order
                if plot_idx == 0:  # concept1
                    marker = "^"  # triangle
                elif plot_idx == 1:  # concept2
                    marker = "o"  # circle
                else:  # composition
                    marker = "s"  # square

                # Set different alpha values for each point (no label)
                for i, (x, mean, alpha) in enumerate(
                    zip(range(len(means)), means, alphas)
                ):
                    plt.plot(
                        x,
                        mean,
                        marker,
                        color=COLORS[plot_idx],
                        markersize=8,
                        alpha=alpha,
                    )

    # draw vertical line behind emphasized axis
    if highlighted_indices:
        y_min, y_max = plt.ylim()
        for idx in highlighted_indices:
            plt.axvline(
                x=idx, color="orange", linestyle="-", alpha=0.3, linewidth=4, zorder=0
            )

    setup_overlapped_plot_axes(space_labels, category_norms_data, num_subspaces)

    overlapped_output_path = os.path.join(plot_output_dir, "overlapped_comparison.png")
    plt.savefig(overlapped_output_path, bbox_inches="tight", dpi=300)
    plt.close()

    logger.info(f"Overlapped comparison plot saved to {overlapped_output_path}")


def prepare_space_data(norms_matrix, num_subspaces):
    """Prepare space data and labels"""
    space_data = []
    space_labels = []

    for space_idx in range(num_subspaces):
        if norms_matrix[:, space_idx].size > 0:
            space_data.append(norms_matrix[:, space_idx])
            space_labels.append(f"{space_idx}")

    return space_data, space_labels


def setup_subplot_axes(ax, space_labels, means, category_name):
    if len(space_labels) > 10:
        tick_positions = []
        tick_labels = []
        for i, label in enumerate(space_labels):
            if i % 10 == 0 or i == len(space_labels) - 1:
                tick_positions.append(i)
                tick_labels.append(label)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
    else:
        ax.set_xticks(range(len(space_labels)))
        ax.set_xticklabels(space_labels)

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    if means:
        min_mean = min(means)
        ax.set_ylim(bottom=min_mean * Y_LIM_MARGIN)


def setup_overlapped_plot_axes(space_labels, category_norms_data, num_subspaces):
    """Configure axes for overlaid comparison plot"""
    # display x-axis labels every 10 units
    if len(space_labels) > 10:
        tick_positions = []
        tick_labels = []
        for i, label in enumerate(space_labels):
            if i % 10 == 0 or i == len(space_labels) - 1:
                tick_positions.append(i)
                tick_labels.append(label)
        plt.xticks(tick_positions, labels=tick_labels)
    else:
        plt.xticks(range(len(space_labels)), labels=space_labels)

    plt.legend(fontsize=12, loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)

    all_means = []
    for category_name, norms_matrix in category_norms_data.items():
        for space_idx in range(num_subspaces):
            if norms_matrix[:, space_idx].size > 0:
                all_means.append(np.mean(norms_matrix[:, space_idx]))

    if all_means:
        min_mean = min(all_means)
        plt.ylim(bottom=min_mean * Y_LIM_MARGIN)


def main(_A: argparse.Namespace):
    try:
        model, tokenizer, is_phyclip = setup_model(_A.checkpoint_path, _A.train_config)
        num_subspaces = model.num_subspaces if is_phyclip else 1

        checkpoint_dir = os.path.dirname(_A.checkpoint_path)
        output_dir = os.path.join(checkpoint_dir, "composition_analysis")

        logger.info("Creating composition analysis plots...")
        composition_dir = os.path.join(output_dir, "composition")
        os.makedirs(composition_dir, exist_ok=True)

        # run all test cases
        for test_case_name, test_case in COMPOSITION_TEST_CASES.items():
            logger.info(f"Processing test case: {test_case_name}")

            # Prepare categories for test case
            categories = {
                test_case["concept1"]["name"]: test_case["concept1"]["prompts"],
                test_case["concept2"]["name"]: test_case["concept2"]["prompts"],
                test_case["composition"]["name"]: test_case["composition"]["prompts"],
            }

            # Collect norm data
            category_norms_data, num_subspaces = collect_norm_data(
                model, tokenizer, categories, is_phyclip
            )

            # create test case subfolders within composition folder
            test_case_dir = os.path.join(composition_dir, test_case_name)
            os.makedirs(test_case_dir, exist_ok=True)

            create_overlapped_comparison_plot(
                category_norms_data, num_subspaces, test_case_dir
            )

            logger.info(
                f"Test case {test_case_name} completed successfully in {test_case_dir}"
            )

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
