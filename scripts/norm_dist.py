from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision.utils as vutils
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from phyclip import lorentz as L
from phyclip.config import LazyConfig, LazyFactory
from phyclip.models import PHyCLIP
from phyclip.tokenizer import Tokenizer
from phyclip.utils.checkpointing import CheckpointManager

PLOT_COLORS = {
    "images": "#2E86AB",  # blue
    "image_boxes": "#A23B72",  # purple
    "texts": "#F18F01",  # orange
    "text_boxes": "#C73E1D",  # red
}


def create_norm_distribution_plot(
    image_norms,
    box_image_norms,
    text_norms,
    box_text_norms,
    xlabel,
    figsize=(5, 2),
    save_path=None,
):
    plt.figure(figsize=figsize)

    # Plot histograms for each embedding type
    sns.histplot(
        data=image_norms.squeeze(),
        bins="auto",
        stat="density",
        kde=True,
        element="step",
        alpha=0.6,
        color=PLOT_COLORS["images"],
        label="Images",
    )
    sns.histplot(
        data=box_image_norms.squeeze(),
        bins="auto",
        stat="density",
        kde=True,
        element="step",
        alpha=0.6,
        color=PLOT_COLORS["image_boxes"],
        label="Image boxes",
    )
    sns.histplot(
        data=text_norms.squeeze(),
        bins="auto",
        stat="density",
        kde=True,
        element="step",
        alpha=0.6,
        color=PLOT_COLORS["texts"],
        label="Texts",
    )
    sns.histplot(
        data=box_text_norms.squeeze(),
        bins="auto",
        stat="density",
        kde=True,
        element="step",
        alpha=0.6,
        color=PLOT_COLORS["text_boxes"],
        label="Text boxes",
    )

    plt.xlabel(xlabel)
    plt.ylabel("density")
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Saved PNG plot to {save_path}")

    plt.close()


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA(
    "--checkpoint-path",
    help="Path to checkpoint of a trained HyCoCLIP/MERU/CLIP model.",
)
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")

_AA(
    "--top-k",
    type=int,
    default=10,
    help="Number of top/bottom images (per subspace) to keep & save (streaming).",
)


def main(_A: argparse.Namespace):
    device = (
        torch.device(f"cuda:{torch.cuda.current_device()}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # --- Setup ---
    _C_TRAIN = LazyConfig.load(_A.train_config)
    logger.info(OmegaConf.to_yaml(_C_TRAIN))
    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    dataloader = LazyFactory.build_dataloader(_C_TRAIN)
    tokenizer = Tokenizer()

    logger.info(
        f"Generating norm distribution for checkpoint in {_A.checkpoint_path}..."
    )
    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)
    model = model.eval()

    # --- Output directory structure (same as checkpoint) ---
    checkpoint_dir = os.path.dirname(_A.checkpoint_path)
    base_dir = checkpoint_dir if checkpoint_dir else "."
    plot_ext = ".png"
    SPATIAL_OUT_DIR = os.path.join(base_dir, "spatial_dist")
    os.makedirs(SPATIAL_OUT_DIR, exist_ok=True)
    logger.info(f"All spatial norm artifacts will be saved under: {SPATIAL_OUT_DIR}")

    # --- Unify cases ---
    is_phyclip = isinstance(model, PHyCLIP)
    num_subspaces = model.num_subspaces if is_phyclip else 1

    # --- Norm Calculation Loop ---
    image_norms, box_image_norms = (
        [[] for _ in range(num_subspaces)],
        [[] for _ in range(num_subspaces)],
    )
    text_norms, box_text_norms = (
        [[] for _ in range(num_subspaces)],
        [[] for _ in range(num_subspaces)],
    )
    collected_texts: list[str] = []
    collected_box_texts: list[str] = []

    # List to store norms from overall embeddings
    overall_image_norms = []
    overall_box_image_norms = []
    overall_text_norms = []
    overall_box_text_norms = []

    top_k = _A.top_k
    top_images = [[] for _ in range(num_subspaces)]  # list of (norm, key, uint8_tensor)
    bottom_images = [
        [] for _ in range(num_subspaces)
    ]  # list of (norm, key, uint8_tensor)
    processed_counts = [0 for _ in range(num_subspaces)]

    batches = 0

    # Norm Calculation and Collecting Top/Bottom Norm Images
    for batch in tqdm(dataloader, desc="Generating representation norms"):
        with torch.inference_mode():
            tokens = tokenizer(batch["text"])
            box_tokens = tokenizer(batch["box_text"])

            image_feats = model.encode_image(
                batch["image"].to(model.device), project=True
            )
            box_image_feats = model.encode_image(
                batch["box_image"].to(model.device), project=True
            )
            text_feats = model.encode_text(tokens, project=True)
            box_text_feats = model.encode_text(box_tokens, project=True)

            batch_text_list = batch["text"]
            batch_box_text_list = batch["box_text"]
            if isinstance(batch_text_list, (list, tuple)):
                collected_texts.extend([str(t) for t in batch_text_list])
            else:
                collected_texts.append(str(batch_text_list))
            if isinstance(batch_box_text_list, (list, tuple)):
                collected_box_texts.extend([str(t) for t in batch_box_text_list])
            else:
                collected_box_texts.append(str(batch_box_text_list))

            if is_phyclip:
                # Reshape features for get_space_norm
                image_feats = image_feats.view(
                    -1, model.num_subspaces, model.subspace_dim
                )
                box_image_feats = box_image_feats.view(
                    -1, model.num_subspaces, model.subspace_dim
                )
                text_feats = text_feats.view(
                    -1, model.num_subspaces, model.subspace_dim
                )
                box_text_feats = box_text_feats.view(
                    -1, model.num_subspaces, model.subspace_dim
                )

            curvs = (
                torch.stack([curv.exp() for curv in model.curvs])
                if is_phyclip
                else model.curv.exp()
            )

            batch_image_norms = L.get_distance_from_origin(
                image_feats, curvs
            ).transpose(0, 1)  # (num_subspaces, batch_size, 1)

            batch_box_image_norms = L.get_distance_from_origin(
                box_image_feats, curvs
            ).transpose(0, 1)
            batch_text_norms = L.get_distance_from_origin(text_feats, curvs).transpose(
                0, 1
            )
            batch_box_text_norms = L.get_distance_from_origin(
                box_text_feats, curvs
            ).transpose(0, 1)

            overall_image_norms.append(
                batch_image_norms.sum(0).cpu().numpy()  # (batchsize, 1)
            )
            overall_box_image_norms.append(batch_box_image_norms.sum(0).cpu().numpy())
            overall_text_norms.append(batch_text_norms.sum(0).cpu().numpy())
            overall_box_text_norms.append(batch_box_text_norms.sum(0).cpu().numpy())

            # --- Update Top/Bottom Image Candidates (streaming, optimized) ---
            batch_size = batch["image"].shape[0]
            keys = batch.get("__key__", [f"idx_{k}" for k in range(batch_size)])
            images_gpu = batch["image"].detach().clamp(0, 1)  # keep on GPU

            for subspace_idx in range(num_subspaces):
                subspace_image_norm_vals = (
                    batch_image_norms[subspace_idx].view(-1).cpu().numpy()
                )

                # Track already added image keys to avoid duplicates
                added_keys = set()
                for collection in [
                    top_images[subspace_idx],
                    bottom_images[subspace_idx],
                ]:
                    added_keys.update(key for _, key, _ in collection)

                for collection_type, (collection, reverse_sort) in [
                    ("top", (top_images[subspace_idx], True)),
                    ("bottom", (bottom_images[subspace_idx], False)),
                ]:
                    # Determine update candidates (excluding duplicates)
                    if len(collection) < top_k:
                        cand_idx = np.arange(batch_size)
                    else:
                        threshold = (
                            min(x[0] for x in collection)
                            if reverse_sort
                            else max(x[0] for x in collection)
                        )
                        if reverse_sort:
                            cand_idx = np.where(subspace_image_norm_vals > threshold)[0]
                        else:
                            cand_idx = np.where(subspace_image_norm_vals < threshold)[0]

                    # Add candidates (with duplicate check)
                    for idx in cand_idx:
                        norm_val = float(subspace_image_norm_vals[idx])
                        key = keys[idx]

                        # Skip already added keys (duplicate check)
                        if key in added_keys:
                            continue

                        img_gpu = images_gpu[idx]
                        collection.append((norm_val, key, img_gpu))
                        added_keys.add(key)

                    # Sort and keep top K
                    if len(collection) > top_k:
                        collection.sort(key=lambda x: x[0], reverse=reverse_sort)
                        del collection[top_k:]

                processed_counts[subspace_idx] += batch_size

            for i in range(num_subspaces):
                image_norms[i].append(batch_image_norms[i].to("cpu").detach().numpy())
                box_image_norms[i].append(
                    batch_box_image_norms[i].to("cpu").detach().numpy()
                )
                text_norms[i].append(batch_text_norms[i].to("cpu").detach().numpy())
                box_text_norms[i].append(
                    batch_box_text_norms[i].to("cpu").detach().numpy()
                )

            batches += 1
            if batches > 167:
                break

    # Plotting
    for i in range(num_subspaces):
        # Concatenate norms (lists of arrays shaped (B,1)); result -> (TotalBatches*B,1)
        subspace_image_norms = np.concatenate(image_norms[i], axis=0)
        subspace_box_image_norms = np.concatenate(box_image_norms[i], axis=0)
        subspace_text_norms = np.concatenate(text_norms[i], axis=0)
        subspace_box_text_norms = np.concatenate(box_text_norms[i], axis=0)

        # subspace specific directory
        subspace_dir = os.path.join(SPATIAL_OUT_DIR, f"subspace_{i}")
        os.makedirs(subspace_dir, exist_ok=True)

        # Plotting
        create_norm_distribution_plot(
            subspace_image_norms,
            subspace_box_image_norms,
            subspace_text_norms,
            subspace_box_text_norms,
            rf"$\Vert \mathbf{{x}}^{{({i})}} \Vert$",
            save_path=os.path.join(subspace_dir, f"norms{plot_ext}"),
        )

        try:
            image_values = subspace_image_norms.flatten()
            text_values = subspace_text_norms.flatten()
            box_text_values = subspace_box_text_norms.flatten()

            image_sorted_idx = np.argsort(image_values)
            text_sorted_idx = np.argsort(text_values)
            box_text_sorted_idx = np.argsort(box_text_values)
            idx_top_k = _A.top_k
            image_bottom = image_sorted_idx[:idx_top_k]
            image_top = image_sorted_idx[-idx_top_k:][::-1]
            text_bottom = text_sorted_idx[:idx_top_k]
            text_top = text_sorted_idx[-idx_top_k:][::-1]
            box_text_bottom = box_text_sorted_idx[:idx_top_k]
            box_text_top = box_text_sorted_idx[-idx_top_k:][::-1]

            def pack(indices, values):
                return [
                    {"rank": r + 1, "index": int(idx), "value": float(values[idx])}
                    for r, idx in enumerate(indices)
                ]

            def pack_with_text(indices, values, text_store):
                out = []
                for r, idx in enumerate(indices):
                    txt = text_store[idx] if 0 <= idx < len(text_store) else None
                    out.append(
                        {
                            "rank": r + 1,
                            "index": int(idx),
                            "value": float(values[idx]),
                            "text": txt,
                        }
                    )
                return out

            # Streamed top/bottom images
            top_stream = sorted(top_images[i], key=lambda x: x[0], reverse=True)
            bottom_stream = sorted(bottom_images[i], key=lambda x: x[0])

            # save images (individual + grid)
            saved_top = []
            saved_bottom = []
            for rank, (norm_val, key, img_gpu) in enumerate(top_stream, start=1):
                out_path = os.path.join(subspace_dir, f"image_top_rank{rank}_{key}.png")
                vutils.save_image(img_gpu, out_path)
                saved_top.append(
                    {"rank": rank, "key": key, "norm": norm_val, "file": out_path}
                )
            for rank, (norm_val, key, img_gpu) in enumerate(bottom_stream, start=1):
                out_path = os.path.join(
                    subspace_dir, f"image_bottom_rank{rank}_{key}.png"
                )
                vutils.save_image(img_gpu, out_path)
                saved_bottom.append(
                    {"rank": rank, "key": key, "norm": norm_val, "file": out_path}
                )

            def _save_grid(stream_list, grid_path):
                if not stream_list:
                    return None
                tensors = [rec[2] for rec in stream_list]
                grid = vutils.make_grid(
                    torch.stack(tensors, dim=0),
                    nrow=min(len(tensors), 5),
                    normalize=True,
                    value_range=(0, 1),
                )
                vutils.save_image(grid, grid_path)
                return grid_path

            _save_grid(top_stream, os.path.join(subspace_dir, "image_top_grid.png"))
            _save_grid(
                bottom_stream, os.path.join(subspace_dir, "image_bottom_grid.png")
            )

            summary = {
                "subspace": int(i),
                "image_norms_top": pack(image_top, image_values),
                "image_norms_bottom": pack(image_bottom, image_values),
                "text_norms_top": pack_with_text(
                    text_top, text_values, collected_texts
                ),
                "text_norms_bottom": pack_with_text(
                    text_bottom, text_values, collected_texts
                ),
                "box_text_norms_top": pack_with_text(
                    box_text_top, box_text_values, collected_box_texts
                ),
                "box_text_norms_bottom": pack_with_text(
                    box_text_bottom, box_text_values, collected_box_texts
                ),
                "total_image_samples": int(image_values.shape[0]),
                "total_text_samples": int(text_values.shape[0]),
                "total_box_text_samples": int(box_text_values.shape[0]),
            }

            json_path = os.path.join(subspace_dir, "top_bottom.json")
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved top/bottom norm summary & images to {json_path}")
        except Exception as e:
            logger.error(
                f"Failed to save top/bottom norm summary for subspace {i}: {e}"
            )

    # --- Save overall spatial norm dist plot ---
    if overall_image_norms:  # If data exists
        logger.info("Generating overall spatial norm distribution plot...")

        overall_image_norms_combined = np.concatenate(overall_image_norms, axis=0)
        overall_box_image_norms_combined = np.concatenate(
            overall_box_image_norms, axis=0
        )
        overall_text_norms_combined = np.concatenate(overall_text_norms, axis=0)
        overall_box_text_norms_combined = np.concatenate(overall_box_text_norms, axis=0)

        # save overall plot
        create_norm_distribution_plot(
            overall_image_norms_combined,
            overall_box_image_norms_combined,
            overall_text_norms_combined,
            overall_box_text_norms_combined,
            r"$\Vert \mathbf{x} \Vert$",
            save_path=os.path.join(SPATIAL_OUT_DIR, f"overall_norms{plot_ext}"),
        )


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
