from __future__ import annotations

import argparse
import pickle

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch import Tensor
from tqdm import tqdm

from phyclip.config import LazyConfig, LazyFactory
from phyclip.tokenizer import Tokenizer
from phyclip.utils.checkpointing import CheckpointManager

parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA(
    "--checkpoint-path",
    help="Path to checkpoint of a trained HyCoCLIP/MERU/CLIP model.",
)
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--embed-save-path", help="Path to save embeddings in .pkl format.")


def create_hyperboloid_embed(x: Tensor, curv: float | Tensor = 1.0):
    """
    Compute the time dimension from spatial coordinates and return as Lorentzian N+1 dim vector.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, D+1)` giving full hyperboloid vector.
    """

    x_time = torch.sqrt(1 / curv + torch.sum(x**2, dim=-1, keepdim=True))
    x_full = torch.cat([x_time, x], dim=-1)
    return x_full


def get_space_norm(x: Tensor):
    return torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))


def main(_A: argparse.Namespace):
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create evaluation and training config objects.
    _C_TRAIN = LazyConfig.load(_A.train_config)
    logger.info(OmegaConf.to_yaml(_C_TRAIN))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    dataloader = LazyFactory.build_dataloader(_C_TRAIN)
    tokenizer = Tokenizer()

    logger.info(f"Generating embeddings for checkpoint in {_A.checkpoint_path}...")

    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)
    model = model.eval()

    all_image_feats, all_box_image_feats = [], []
    all_text_feats, all_box_text_feats = [], []
    batches = 0

    for batch in tqdm(dataloader, desc="Generating representations..."):
        with torch.inference_mode():
            tokens = tokenizer(batch["text"])
            box_tokens = tokenizer(batch["box_text"])

            image_feats = model.encode_image(
                batch["image"].to(model.device), project=True
            )
            image_feats = create_hyperboloid_embed(image_feats, model.curv.exp())

            box_image_feats = model.encode_image(
                batch["box_image"].to(model.device), project=True
            )
            box_image_feats = create_hyperboloid_embed(
                box_image_feats, model.curv.exp()
            )

            text_feats = model.encode_text(tokens, project=True)
            text_feats = create_hyperboloid_embed(text_feats, model.curv.exp())

            box_text_feats = model.encode_text(box_tokens, project=True)
            box_text_feats = create_hyperboloid_embed(box_text_feats, model.curv.exp())

            all_image_feats.append(image_feats.to("cpu").detach().numpy())
            all_box_image_feats.append(box_image_feats.to("cpu").detach().numpy())
            all_text_feats.append(text_feats.to("cpu").detach().numpy())
            all_box_text_feats.append(box_text_feats.to("cpu").detach().numpy())

            batches += 1
            if batches > 25:
                break

    all_image_feats = np.concatenate(all_image_feats, axis=0)
    all_box_image_feats = np.concatenate(all_box_image_feats, axis=0)
    all_text_feats = np.concatenate(all_text_feats, axis=0)
    all_box_text_feats = np.concatenate(all_box_text_feats, axis=0)

    embed_dict = {
        "image_feats": all_image_feats,
        "box_image_feats": all_box_image_feats,
        "text_feats": all_text_feats,
        "box_text_feats": all_box_text_feats,
    }

    with open(_A.embed_save_path, "wb") as f:
        pickle.dump(embed_dict, f)


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
