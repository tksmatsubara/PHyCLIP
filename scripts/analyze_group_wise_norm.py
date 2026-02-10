import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from phyclip import lorentz as L
from phyclip.config import LazyConfig, LazyFactory
from phyclip.models import PHyCLIP
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
_AA("--output-dir", required=True, help="Path to the directory to save the plots.")
_AA("--num-words", type=int, default=100, help="Number of words to generate per group.")
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

    if not isinstance(model, PHyCLIP):
        logger.error("This script only supports PHyCLIP models.")
        return

    # --- Define Word Groups from WordNet root synsets ---
    root_synsets_map = get_root_synsets(level=_A.level)

    os.makedirs(_A.output_dir, exist_ok=True)

    # --- Loop over each group, generate words, calculate norms, and plot ---
    for group_name, synset in root_synsets_map.items():
        logger.info(f"--- Processing group: {group_name} ---")

        # --- Word Generation ---
        word_list = generate_wordnet_words(
            num_words=_A.num_words, seed=_A.seed, root_synset=synset
        )
        if not word_list:
            logger.warning(
                f"Could not generate words for group {group_name}. Skipping."
            )
            continue
        logger.info(f"Generated {len(word_list)} words: {word_list}")

        # --- Norm Calculation ---
        num_subspaces = model.num_subspaces
        all_norms = []

        for word in tqdm(word_list, desc=f"Calculating norms for {group_name}"):
            with torch.inference_mode():
                tokens = tokenizer(f"a photo of a {word}")
                text_feats = model.encode_text(tokens, project=True)  # type: ignore
                text_subspace_feats = text_feats.view(
                    -1, model.num_subspaces, model.subspace_dim
                )
                batch_text_norms = L.get_space_norm(text_subspace_feats)
                all_norms.append(batch_text_norms.squeeze().cpu().numpy())

        if not all_norms:
            logger.warning(
                f"No norms were calculated for group {group_name}. Skipping plot."
            )
            continue

        # --- Averaging and Plotting ---
        all_norms_np = np.array(all_norms)
        avg_norms = np.mean(all_norms_np, axis=0)

        plt.figure(figsize=(12, 6))
        space_indices = np.arange(num_subspaces)
        plt.bar(space_indices, avg_norms, width=0.8)
        # Adjust Y-axis to better show differences
        min_norm = np.min(avg_norms)
        max_norm = np.max(avg_norms)
        margin = (max_norm - min_norm) * 0.1  # 10% margin
        plt.ylim(min_norm - margin, max_norm + margin)

        plt.xlabel("Concept Space Index")
        plt.ylabel("Average Norm")
        plt.title(
            f"Average Norm per Concept Space\nGroup: {group_name} (Words: {len(word_list)}, Seed: {_A.seed})"
        )
        # Set x-ticks every 10 indices
        xtick_indices = np.arange(0, num_subspaces, 10)
        plt.xticks(xtick_indices)
        plt.grid(axis="y", linestyle="--")

        # # --- Adjust figure width to prevent x-label overlap ---
        # plt.gcf().set_size_inches(max(12, num_subspaces * 0.7), 6)
        # plt.tight_layout()

        # Create a dedicated subdirectory for the plots
        plot_output_dir = os.path.join(_A.output_dir, "group_wise_norm_plots")
        os.makedirs(plot_output_dir, exist_ok=True)
        logger.info(f"Plots will be saved in: {plot_output_dir}")

        # --- Save Plot ---
        sanitized_group_name = group_name.replace(".", "_")
        output_path = os.path.join(
            plot_output_dir, f"avg_norm_{sanitized_group_name}.png"
        )
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved plot for group {group_name} to {output_path}")
        plt.close()


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
