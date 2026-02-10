"""
VL-Checklist wrapper for PHyCLIP models.

This module provides a wrapper class to integrate PHyCLIP models with the VL-Checklist
evaluation framework. It implements the required predict() method and handles image-text
matching tasks.
"""

import base64
import io
import os
from typing import Dict, List, Optional, Tuple

import requests
import torch
from PIL import Image
from vl_checklist.vlp_model import VLPModel

from phyclip.config import LazyConfig, LazyFactory
from phyclip.evaluation.vl_checklist_helper import (
    LRUCache,
    chunks,
    pixelbert_transform,
)
from phyclip.tokenizer import Tokenizer
from phyclip.utils.checkpointing import CheckpointManager
from phyclip.utils.evaluation import compute_similarity_scores


class VLChecklistWrapper(VLPModel):
    """
    Wrapper class for PHyCLIP models to integrate with VL-Checklist evaluation framework.

    This wrapper provides the necessary interface methods required by VL-Checklist,
    including the predict() method for image-text matching tasks.
    """

    MAX_CACHE = 5  # Reduced cache size due to potentially large models

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda:0",
        batch_size: int = 8,
    ):
        """
        Initialize the PHyCLIP wrapper.

        Args:
            checkpoint_path: Path to the model checkpoint
            config_path: Path to the model configuration file
            device: Device to run the model on
            batch_size: Batch size for inference
        """
        self._models = LRUCache(self.MAX_CACHE)
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Initialize tokenizer
        self.tokenizer = Tokenizer()

    def load_model(self, checkpoint_path: str) -> Tuple[torch.nn.Module, Tokenizer]:
        """
        Load the PHyCLIP model and tokenizer.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Tuple of (model, tokenizer)
        """
        if checkpoint_path is None:
            raise Exception("Checkpoint path cannot be None.")

        if not self._models.has(checkpoint_path):
            if self.config_path is None or self.checkpoint_path is None:
                raise ValueError("config_path and checkpoint_path must be provided")

            # Load configuration
            config = LazyConfig.load(self.config_path)

            # Build model
            model = LazyFactory.build_model(config, self.device)

            # Load checkpoint
            if os.path.exists(self.checkpoint_path):
                checkpoint_manager = CheckpointManager(model=model)
                checkpoint_manager.load(self.checkpoint_path)

            model.eval()
            model.to(self.device)

            self._models.put(checkpoint_path, (model, self.tokenizer))

        return self._models.get(checkpoint_path)

    def load_data(self, src_type, data):
        def transform(x):
            img = x.resize((224, 224))
            img = pixelbert_transform(size=224)(img)
            img = img.unsqueeze(0).to(self.device)
            return img

        if src_type == "local":
            image_data = []
            for x in data:
                temp = Image.open(x).convert("RGB")
                image_data.append(transform(temp))

        elif src_type == "url":
            image_data = []
            for x in data:
                temp = Image.open(io.BytesIO(requests.get(x).content)).convert("RGB")
                image_data.append(transform(temp))

        elif src_type == "base64":
            image_data = []
            for x in data:
                temp = Image.open(io.BytesIO(base64.b64decode(x)))
                image_data.append(transform(temp))
        else:
            raise Exception("Unknown mode {}.".format(src_type))

        return image_data

    def predict(
        self, image_paths: List[str], texts: List[str], src_type: str = "local"
    ) -> Dict[str, List[List[float]]]:
        """
        Predict image-text matching probabilities.

        Args:
            image_paths: List of image paths/URLs/base64 strings
            texts: List of text captions
            src_type: Source type for images ('local', 'url', 'base64')

        Returns:
            Dictionary with 'probs' key containing list of [negative_prob, positive_prob] pairs
        """
        if not len(texts) == len(image_paths):
            raise Exception("# of texts and # of images should be matched")

        model, tokenizer = self.load_model(self.checkpoint_path)

        # Process by batches
        probs = []

        for chunk_images, chunk_texts in zip(
            chunks(image_paths, self.batch_size), chunks(texts, self.batch_size)
        ):
            # Load and preprocess images
            image_data = self.load_data(src_type, chunk_images)

            batch_images = []
            batch_text = []

            for i, t in zip(image_data, chunk_texts):
                batch_images.append(i)
                batch_text.append(t)

            # Convert list of image tensors to a single batch tensor
            # Each image tensor has shape (1, C, H, W), so we concatenate them
            batch_images = torch.cat(batch_images, dim=0)  # (batch_size, C, H, W)

            # Use tokenizer the same way as in retrieval.py
            # tokenizer returns a list of torch.IntTensor, which encode_text handles internally
            text_tokens = tokenizer(batch_text)

            with torch.no_grad():
                # Encode images
                image_features = model.encode_image(batch_images, project=True)

                # # Encode texts - encode_text handles the list of tensors internally
                text_features = model.encode_text(text_tokens, project=True)

                # Handle PHyCLIP concept features
                if hasattr(model, "num_subspaces") and hasattr(model, "subspace_dim"):
                    # PHyCLIP uses concept features, reshape for proper computation
                    image_features = image_features.view(
                        -1, model.num_subspaces, model.subspace_dim
                    )
                    text_features = text_features.view(
                        -1, model.num_subspaces, model.subspace_dim
                    )

                # For VL-Checklist, we need to compute similarity for each image-text pair individually
                # Unlike retrieval tasks, here we evaluate each (image[i], text[i]) pair independently
                batch_probs = []

                for i in range(len(batch_images)):
                    # Get individual image and text features
                    single_image_feat = image_features[
                        i : i + 1
                    ]  # Keep batch dimension
                    single_text_feat = text_features[i : i + 1]  # Keep batch dimension

                    # Compute similarity score for this specific pair
                    similarity_score = compute_similarity_scores(
                        model, single_image_feat, single_text_feat
                    )

                    # similarity_score should be a scalar (distance between this image-text pair)
                    if similarity_score.dim() > 0:
                        similarity_score = similarity_score.item()

                    # Use similarity score directly as positive probability
                    # Higher similarity score = better match = higher positive probability
                    # For VL-Checklist comparison, we just need relative ordering
                    positive_prob = similarity_score
                    negative_prob = -similarity_score  # Inverse for completeness

                    probs_pair = [negative_prob, positive_prob]

                    batch_probs.append(probs_pair)

                probs.extend(batch_probs)

        return {"probs": probs}
