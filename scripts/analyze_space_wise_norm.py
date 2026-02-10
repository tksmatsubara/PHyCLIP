import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from loguru import logger
from tqdm import tqdm

from phyclip import lorentz as L
from phyclip.config import LazyConfig, LazyFactory
from phyclip.models import HyCoCLIP, PHyCLIP
from phyclip.tokenizer import Tokenizer
from phyclip.utils.checkpointing import CheckpointManager
from phyclip.utils.word_generation import generate_wordnet_words, get_root_synsets

parser = argparse.ArgumentParser(description=__doc__)
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
_AA(
    "--output-dir",
    required=True,
    help="Path to the directory where the output folder will be created.",
)
_AA(
    "--num-words",
    type=int,
    default=1000,
    help="Number of words to generate per group.",
)
_AA("--seed", type=int, default=42, help="Seed for word generation.")
_AA(
    "--level",
    type=int,
    default=1,
    help="Level of WordNet synsets to analyze (1=root, 2=more specific, etc.).",
)


def main(_A: argparse.Namespace):
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # --- Setup ---
    _C_TRAIN = LazyConfig.load(_A.train_config)
    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    tokenizer = Tokenizer()

    logger.info(f"Loading model from checkpoint: {_A.checkpoint_path}...")
    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)
    model = model.eval()

    is_phyclip = isinstance(model, PHyCLIP)
    if not is_phyclip and not isinstance(model, HyCoCLIP):
        logger.error("This script only supports PHyCLIP or HyCoCLIP models.")
        return

    num_subspaces = model.num_subspaces if is_phyclip else 1

    # --- Data Collection ---
    logger.info("Collecting norms for all word groups. This may take a while...")
    root_synsets_map = get_root_synsets(level=_A.level)
    group_norms_data = {}

    for group_name, synset in root_synsets_map.items():
        word_list = generate_wordnet_words(
            num_words=_A.num_words,
            seed=_A.seed,
            root_synset=synset,
            include_compounds=True,
        )
        if not word_list:
            logger.warning(
                f"Could not generate words for group {group_name}. Skipping."
            )
            continue

        all_norms_for_group = []
        for word in tqdm(word_list, desc=f"Processing {group_name}", leave=False):
            with torch.inference_mode():
                tokens = tokenizer([f"a photo of a {word}"])
                text_feats = model.encode_text(tokens, project=True)  # type: ignore

                if is_phyclip:
                    text_subspace_feats = text_feats.view(
                        -1, model.num_subspaces, model.subspace_dim
                    )
                    batch_text_norms = L.get_space_norm(text_subspace_feats)
                else:  # HyCoCLIP case
                    # Calculate norm for the single space and reshape for consistency
                    batch_text_norms = L.get_space_norm(
                        text_feats.unsqueeze(1)
                    )  # Add concept dim

                all_norms_for_group.append(batch_text_norms.squeeze().cpu().numpy())

        if all_norms_for_group:
            group_norms_data[group_name] = np.array(all_norms_for_group)

    if not group_norms_data:
        logger.error("No norm data was collected. Aborting.")
        return

    # --- Plotting ---
    logger.info("All norms collected. Now generating plots for each concept space.")

    num_subspaces = model.num_subspaces if is_phyclip else 1
    for space_idx in tqdm(range(num_subspaces), desc="Generating plots per space"):
        plt.figure(figsize=(12, 8))

        # Keep track if any data was plotted for this space
        plotted_data_for_space = False

        for group_name, norms_matrix in group_norms_data.items():
            # norms_matrix shape: (num_words, num_subspaces)
            # Extract norms for the current space index
            norms_for_space = norms_matrix[:, space_idx] if is_phyclip else norms_matrix

            # Check if norms_for_space is empty
            if norms_for_space.size == 0:
                logger.warning(
                    f"No data for group {group_name} in concept space {space_idx}. Skipping plot for this group."
                )
                continue

            num_words_in_group = norms_matrix.shape[0]

            sns.histplot(
                data=norms_for_space,
                bins="auto",
                stat="percent",
                kde=True,
                element="step",
                alpha=0.5,
                label=f"{group_name} (n={num_words_in_group})",
            )
            plotted_data_for_space = True

        if not plotted_data_for_space:
            logger.warning(
                f"No data to plot for concept space {space_idx}. Skipping plot for this space."
            )
            plt.close()  # Close the empty figure
            continue

        title = (
            f"Norm Distribution in Concept Space {space_idx}"
            if is_phyclip
            else "Norm Distribution in Hyperbolic Space"
        )
        plt.title(title, fontsize=16)
        plt.xlabel(
            r"$\Vert \mathbf{\tilde{z}} \Vert$", fontsize=14
        )  # Using z for general embedding
        plt.ylabel("Percent (%)", fontsize=14)
        plt.legend(title="WordNet Groups", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Create a dedicated subdirectory for the plots
        plot_output_dir = os.path.join(
            _A.output_dir, f"space_wise_norm_plots_level_{_A.level}"
        )
        os.makedirs(plot_output_dir, exist_ok=True)

        # Save the plot to the subdirectory
        output_path = os.path.join(
            plot_output_dir, f"space_{space_idx:02d}_distribution.png"
        )
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()

    logger.info(f"All plots have been generated successfully in {plot_output_dir}.")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
