# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

# Modified from github.com/facebookresearch/meru

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

import phyclip.utils.distributed as dist
from phyclip import lorentz as L
from phyclip.encoders.text_encoders import TransformerTextEncoder


class CLIPBaseline(nn.Module):
    """
    Re-implementation of the CLIP model that uses an image-text contrastive
    loss as a training objective and embeds images and text in a Euclidean space.

    Reference: CLIP paper (https://arxiv.org/abs/2103.00020)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_boxes: bool = False,
    ):
        """
        Args:
            visual: ConvNet or ViT image encoder to compute image features.
            textual: Transformer-based encoder to compute text features.
            embed_dim: Size of the visual and textual embedding vectors for
                computing pairwise similarity matrix.
            pixel_mean: Normalize input images by this color mean. Default value
                is of ImageNet color, set to `(0, 0, 0)` for no normalization.
            pixel_std: Normalize input images by this color std. Default value
                is of ImageNet color, set to `(1, 1, 1)` for no normalization.
        """
        super().__init__()
        self.visual = visual
        self.textual = textual
        self.embed_dim = embed_dim

        # Linear layers to project image and text features such that they have
        # same size before computing dot-product similarity.
        self.visual_proj = nn.Linear(visual.width, embed_dim, bias=False)
        self.textual_proj = nn.Linear(textual.width, embed_dim, bias=False)

        # CLIP-style initialization of projection layers.
        nn.init.normal_(self.visual_proj.weight, std=visual.width**-0.5)
        nn.init.normal_(self.textual_proj.weight, std=textual.width**-0.5)

        # Initialize a learnable logit scale parameter.
        self.logit_scale = nn.Parameter(torch.tensor(1 / 0.07).log())

        # Color mean/std to normalize image.
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1))

        # Get rank of current GPU process for gathering features.
        self._rank = dist.get_rank()

    @property
    def device(self) -> torch.device:
        return self.logit_scale.device

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Project features to a unit hypersphere through L2 normalization.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """
        images = (images - self.pixel_mean) / self.pixel_std
        image_feats = self.visual(images)
        image_feats = self.visual_proj(image_feats)

        if project:
            image_feats = F.normalize(image_feats, dim=-1)

        return image_feats

    def encode_text(self, tokens: torch.Tensor, project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Project features to a unit hypersphere through L2 normalization.
        """

        # Truncate tokens that are longer than context_length:
        for idx, inst_tokens in enumerate(tokens):
            if len(inst_tokens) > self.textual.context_length:
                eot_token = inst_tokens[-1]
                inst_tokens = inst_tokens[: self.textual.context_length]
                inst_tokens[-1] = eot_token
                tokens[idx] = inst_tokens

        # Pad all tokens on the right.
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        tokens = tokens.to(self.device)

        # shape: (batch_size, context_length, textual.width)
        text_feats = self.textual(tokens)

        # Get features for [EOS] position and apply projection. `[EOS]` token ID
        # is the largest number in the vocabulary of tokenizer.
        _eos_indices = tokens.argmax(dim=-1)
        batch_idxs = torch.arange(text_feats.shape[0])
        text_feats = text_feats[batch_idxs, _eos_indices]
        text_feats = self.textual_proj(text_feats)

        if project:
            text_feats = F.normalize(text_feats, dim=-1)

        return text_feats

    def forward(
        self, images: torch.Tensor, tokens: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Clamp temperature such that logits are not scaled more than 100x.
        # ln(100) = ~4.6052
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        # Compute logits for image-text contrastive loss: cosine similarity.
        image_logits = _scale * image_feats @ all_text_feats.T
        text_logits = _scale * text_feats @ all_image_feats.T

        # Compute cross entropy loss: we compute log probabilities and take the
        # diagonal elements as targets: image[i] should match text[i] in batch.
        # Shift the targets according to rank of GPU process (we assume that all
        # GPU processes have the same local batch size).
        batch_size = image_feats.shape[0]
        targets = torch.arange(batch_size, device=image_logits.device)
        targets = targets + batch_size * self._rank

        loss = 0.5 * (
            F.cross_entropy(image_logits, targets)
            + F.cross_entropy(text_logits, targets)
        )
        output_dict = {
            "loss": loss,
            "logging": {"contrastive_loss": loss, "logit_scale": _scale},
        }
        return output_dict


class MERU(CLIPBaseline):
    """
    Implementation of MERU model that embeds images and text in a hyperbolic space.

    Reference: MERU paper (https://arxiv.org/abs/2304.09172)
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = False,
        lorentz_eps: float = 1e-8,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Un-documented args are same as `CLIPBaseline`.

        Args:
            curv_init: Positive scalar that denotes negative Hyperboloid curvature.
            learn_curv: Whether to learn the curvature parameter during training.
            entail_weight: Weight for the entailment loss component.
            lorentz_eps: Small epsilon value for numerical stability in Lorentz operations.
        """
        super().__init__(visual, textual, embed_dim, pixel_mean, pixel_std)

        # Initialize curvature parameter. Hyperboloid curvature will be `-curv`.
        self.curv = nn.Parameter(
            torch.tensor(curv_init).log(), requires_grad=learn_curv
        )
        # When learning the curvature parameter, restrict it in this interval to
        # prevent training instability.
        self._curv_minmax = {
            "max": torch.tensor(math.log(curv_init * 10), device=self.device),
            "min": torch.tensor(math.log(curv_init / 10), device=self.device),
        }
        self.entail_weight = entail_weight

        # Lorentz operations epsilon for numerical stability
        self.lorentz_eps = lorentz_eps

        # Learnable scalars to ensure that image/text features have an expected
        # unit norm before exponential map (at initialization).
        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())
        self.textual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())

    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            project: Lift features from the encoder onto the Hyperboloid.

        Returns:
            Batch of image features of shape `(B, visual.width)`.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        image_feats = super().encode_image(images, project=False)

        # These features are space components of embeddings in the tangent
        # space of the Hyperboloid origin (which is Euclidean). Apply projection.
        if project:
            image_feats = image_feats * self.visual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                image_feats = L.exp_map0(
                    image_feats, self.curv.exp(), eps=self.lorentz_eps
                )

        return image_feats

    def encode_text(self, tokens: torch.Tensor, project: bool):
        """
        Args:
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
            project: Lift features from the encoder onto the Hyperboloid.
        """

        # Get Euclidean features from the encoder (without L2 normalization).
        text_feats = super().encode_text(tokens, project=False)

        if project:
            text_feats = text_feats * self.textual_alpha.exp()
            with torch.autocast(self.device.type, dtype=torch.float32):
                text_feats = L.exp_map0(
                    text_feats, self.curv.exp(), eps=self.lorentz_eps
                )

        return text_feats

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        self.curv.data = torch.clamp(
            self.curv.data,
            min=self._curv_minmax["min"].to(self.curv.data.device),
            max=self._curv_minmax["max"].to(self.curv.data.device),
        )
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute logits for contrastive loss.
            image_logits = -L.pairwise_dist(
                image_feats, all_text_feats, _curv, eps=self.lorentz_eps
            )
            text_logits = -L.pairwise_dist(
                text_feats, all_image_feats, _curv, eps=self.lorentz_eps
            )

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
            )

            # Hyperbolic entailment loss: text should entail matching image.
            _angle = L.oxy_angle(text_feats, image_feats, _curv, eps=self.lorentz_eps)
            _aperture = L.half_aperture(text_feats, _curv, eps=self.lorentz_eps)

            entailment_loss = torch.clamp(_angle - _aperture, min=0).mean()

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }


class HyCoCLIP(MERU):
    """
    Our HyCoCLIP model, that modifies MERU and CLIP to embed images, texts and their localized box
    information hierarchically in a hyperbolic space.
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = True,
        lorentz_eps: float = 1e-8,
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Un-documented args are same as `MERU`.

        Args:
            use_boxes: Whether to use box images and texts for training.
        """
        super().__init__(
            visual,
            textual,
            embed_dim,
            curv_init,
            learn_curv,
            entail_weight,
            use_boxes,
            lorentz_eps,
            pixel_mean,
            pixel_std,
        )
        assert use_boxes, "HyCoCLIP requires box images and texts to function."

    def forward(
        self,
        images: torch.Tensor,
        box_images: torch.Tensor,
        tokens: torch.Tensor,
        box_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Args:
            images: Image batch in BCHW format, with pixel values in `[0, 1]`.
            tokens: List of tensors, each containing text tokens. Tensors may have
                variable length (they will be padded internally).
        """

        self.curv.data = torch.clamp(
            self.curv.data,
            min=self._curv_minmax["min"].to(self.curv.data.device),
            max=self._curv_minmax["max"].to(self.curv.data.device),
        )
        _curv = self.curv.exp()

        # Clamp scaling factors such that they do not up-scale the feature norms.
        # Once `exp(scale) = 1`, they can simply be removed during inference.
        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        self.textual_alpha.data = torch.clamp(self.textual_alpha.data, max=0.0)

        # shape: (batch_size, embed_dim)
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        box_image_feats = self.encode_image(box_images, project=True)
        box_text_feats = self.encode_text(box_tokens, project=True)

        # Get features from all GPUs to increase negatives for contrastive loss.
        # These will be lists of tensors with length = world size.
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # shape: (batch_size * world_size, embed_dim)
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Compute all necessary loss components. We enclose the entire block with
        # autocast to force a higher floating point precision.
        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute logits for contrastive loss.
            image_logits = -L.pairwise_dist(
                image_feats, all_text_feats, _curv, eps=self.lorentz_eps
            )
            text_logits = -L.pairwise_dist(
                text_feats, all_image_feats, _curv, eps=self.lorentz_eps
            )
            box_image_logits = -L.pairwise_dist(
                box_image_feats, all_text_feats, _curv, eps=self.lorentz_eps
            )
            box_text_logits = -L.pairwise_dist(
                box_text_feats, all_image_feats, _curv, eps=self.lorentz_eps
            )

            # Compute cross entropy loss: we compute log probabilities and take the
            # diagonal elements as targets: image[i] should match text[i] in batch.
            # Shift the targets according to rank of GPU process (we assume that all
            # GPU processes have the same local batch size).
            batch_size = image_feats.shape[0]
            targets = torch.arange(batch_size, device=image_logits.device)
            targets = targets + batch_size * self._rank

            # Clamp temperature such that logits are not scaled more than 100x.
            # ln(100) = ~4.6052
            self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
            _scale = self.logit_scale.exp()

            contrastive_loss = 0.25 * (
                nn.functional.cross_entropy(_scale * image_logits, targets)
                + nn.functional.cross_entropy(_scale * text_logits, targets)
                + nn.functional.cross_entropy(_scale * box_image_logits, targets)
                + nn.functional.cross_entropy(_scale * box_text_logits, targets)
            )

            # Hyperbolic entailment loss: text should entail matching image.
            _angle = L.oxy_angle(text_feats, image_feats, _curv, eps=self.lorentz_eps)
            _aperture = L.half_aperture(text_feats, _curv, eps=self.lorentz_eps)

            _box_angle = L.oxy_angle(
                box_text_feats, box_image_feats, _curv, eps=self.lorentz_eps
            )
            _box_aperture = L.half_aperture(box_text_feats, _curv, eps=self.lorentz_eps)

            _cross_image_angle = L.oxy_angle(
                box_image_feats, image_feats, _curv, eps=self.lorentz_eps
            )
            _box_image_aperture = L.half_aperture(
                box_image_feats, _curv, eps=self.lorentz_eps
            )

            _cross_text_angle = L.oxy_angle(
                box_text_feats, text_feats, _curv, eps=self.lorentz_eps
            )
            _box_text_aperture = L.half_aperture(
                box_text_feats, _curv, eps=self.lorentz_eps
            )

            # Hyperparameters for apertures
            _global_aperture_thresh = 0.7  # inter-modal
            _local_aperture_thresh = 1.2  # intra-modal

            text_image_entailment_loss = torch.clamp(
                _angle - _global_aperture_thresh * _aperture, min=0
            ).mean()
            box_text_image_entailment_loss = torch.clamp(
                _box_angle - _global_aperture_thresh * _box_aperture, min=0
            ).mean()
            cross_image_entailment_loss = torch.clamp(
                _cross_image_angle - _local_aperture_thresh * _box_image_aperture, min=0
            ).mean()
            cross_text_entailment_loss = torch.clamp(
                _cross_text_angle - _local_aperture_thresh * _box_text_aperture, min=0
            ).mean()

            entailment_loss = 0.5 * (
                text_image_entailment_loss
                + box_text_image_entailment_loss
                + cross_image_entailment_loss
                + cross_text_entailment_loss
            )

            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss

        return {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "text_image_entailment_loss": text_image_entailment_loss,
                "box_text_image_entailment_loss": box_text_image_entailment_loss,
                "cross_image_entailment_loss": cross_image_entailment_loss,
                "cross_text_entailment_loss": cross_text_entailment_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": _curv,
            },
        }


class PHyCLIP(MERU):
    """
    Implementation of PHyCLIP model that embeds images and text in multiple hyperbolic spaces,
    each representing a different concept, and combines them using boolean algebra.

    Args:
        visual: ConvNet or ViT image encoder to compute image features.
        textual: Transformer-based encoder to compute text features.
        embed_dim: Size of the visual and textual embedding vectors.
        subspace_dim: Dimension of each concept-specific hyperbolic space.
        curv_init: Initial curvature for each hyperbolic space.
        learn_curv: Whether to learn the curvature parameters during training.
        entail_weight: Weight for the entailment loss component.
        use_boxes: Whether to use box images and texts for training.
        lorentz_eps: Small epsilon value for numerical stability in Lorentz operations.
        product_metric: Product metric to combine distances across concept spaces.
            Options: "l1" (mean of distances) or "l2" (sqrt of mean of squared distances).
        pixel_mean: RGB mean values for image normalization.
        pixel_std: RGB standard deviation values for image normalization.
    """

    def __init__(
        self,
        visual: nn.Module,
        textual: TransformerTextEncoder,
        embed_dim: int,
        subspace_dim: int = 8,
        curv_init: float = 1.0,
        learn_curv: bool = True,
        entail_weight: float = 0.0,
        use_boxes: bool = False,
        lorentz_eps: float = 1e-6,
        product_metric: str = "l1",
        pixel_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        pixel_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__(
            visual,
            textual,
            embed_dim,
            curv_init,
            learn_curv,
            entail_weight,
            use_boxes,
            lorentz_eps,
            pixel_mean,
            pixel_std,
        )

        # Store box usage flag
        self.use_boxes = use_boxes

        # Validate and store product metric
        if product_metric not in ["l1", "l2"]:
            raise ValueError(
                f"product_metric must be 'l1' or 'l2', got '{product_metric}'"
            )
        self.product_metric = product_metric

        # Calculate number of concepts to match total embedding dimension
        self.subspace_dim = subspace_dim
        self.num_subspaces = embed_dim // subspace_dim
        if embed_dim % subspace_dim != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by subspace_dim ({subspace_dim})"
            )

        # Curvature parameters for each concept space
        self.curvs = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(curv_init).log(), requires_grad=learn_curv)
                for _ in range(self.num_subspaces)
            ]
        )

        # Scaling factors for each concept space
        self.visual_concept_alphas = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(subspace_dim**-0.5).log())
                for _ in range(self.num_subspaces)
            ]
        )
        self.textual_concept_alphas = nn.ParameterList(
            [
                nn.Parameter(torch.tensor(subspace_dim**-0.5).log())
                for _ in range(self.num_subspaces)
            ]
        )


    def encode_image(self, images: torch.Tensor, project: bool):
        """
        Encode images into multiple concept-specific hyperbolic spaces.

        Args:
            images: Image batch in BCHW format, with pixel values in [0, 1].
            project: Whether to project features onto hyperbolic spaces.

        Returns:
            Image features of shape (B, embed_dim).
        """
        # Get base features from the encoder
        base_feats = super().encode_image(images, project=False)

        # shape: (B, embed_dim) -> (B, num_subspaces, subspace_dim)
        subspace_feats = base_feats.view(-1, self.num_subspaces, self.subspace_dim)

        if project:
            # Scale and project to hyperbolic space for all concepts at once
            subspace_feats = subspace_feats * torch.stack(
                [alpha.exp() for alpha in self.visual_concept_alphas]
            ).view(1, -1, 1)

            with torch.autocast(self.device.type, dtype=torch.float32):
                # Apply exp_map0_batch to all concept spaces at once
                B, N, D = subspace_feats.shape
                # Use pre-computed curvatures from forward pass if available
                if hasattr(self, "_cached_curvs") and self._cached_curvs is not None:
                    curvs = self._cached_curvs
                else:
                    curvs = torch.stack([curv.exp() for curv in self.curvs])

                # Apply batch exponential map directly without transpose
                subspace_feats = L.exp_map0_batch(
                    subspace_feats, curvs, eps=self.lorentz_eps
                )

        # Reshape back to (B, embed_dim)
        return subspace_feats.view(-1, self.embed_dim)

    def encode_text(self, tokens: torch.Tensor, project: bool):
        """
        Encode text into multiple concept-specific hyperbolic spaces.

        Args:
            tokens: List of tensors containing text tokens.
            project: Whether to project features onto hyperbolic spaces.

        Returns:
            Text features of shape (B, embed_dim).
        """
        # Get base features from the encoder
        base_feats = super().encode_text(tokens, project=False)

        # shape: (B, embed_dim) -> (B, num_subspaces, subspace_dim)
        subspace_feats = base_feats.view(-1, self.num_subspaces, self.subspace_dim)

        if project:
            # Scale and project to hyperbolic space for all concepts at once
            subspace_feats = subspace_feats * torch.stack(
                [alpha.exp() for alpha in self.textual_concept_alphas]
            ).view(1, -1, 1)

            with torch.autocast(self.device.type, dtype=torch.float32):
                # Apply exp_map0_batch to all concept spaces at once
                B, N, D = subspace_feats.shape
                # Use pre-computed curvatures from forward pass if available
                if hasattr(self, "_cached_curvs") and self._cached_curvs is not None:
                    curvs = self._cached_curvs
                else:
                    curvs = torch.stack([curv.exp() for curv in self.curvs])

                # Apply batch exponential map directly without transpose
                subspace_feats = L.exp_map0_batch(
                    subspace_feats, curvs, eps=self.lorentz_eps
                )

        # Reshape back to (B, embed_dim)
        return subspace_feats.view(-1, self.embed_dim)

    def _compute_product_metric_contrastive_loss(
        self,
        image_subspace_feats: torch.Tensor,
        box_image_subspace_feats: Optional[torch.Tensor],
        text_subspace_feats: torch.Tensor,
        box_text_subspace_feats: Optional[torch.Tensor],
        all_image_subspace_feats: torch.Tensor,
        all_text_subspace_feats: torch.Tensor,
        curvs: torch.Tensor,
        targets: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss using product metric to combine distances across concept spaces.

        Args:
            image_subspace_feats: Local image features (B, num_subspaces, subspace_dim)
            text_subspace_feats: Local text features (B, num_subspaces, subspace_dim)
            all_image_subspace_feats: All image features (B*world_size, num_subspaces, subspace_dim)
            all_text_subspace_feats: All text features (B*world_size, num_subspaces, subspace_dim)
            curvs: Curvatures for each concept space (num_subspaces,)
            targets: Target indices for matching pairs (batch_size,)
            scale: Temperature scaling factor
            box_image_subspace_feats: Box image features (B, num_subspaces, subspace_dim) [optional]
            box_text_subspace_feats: Box text features (B, num_subspaces, subspace_dim) [optional]

        Returns:
            Contrastive loss using the configured product metric:
            - L1: mean of distances across concept spaces
            - L2: sqrt of mean of squared distances across concept spaces
        """
        # Compute distances for all concept spaces at once
        image_distances = L.pairwise_dist_batch(
            image_subspace_feats,
            all_text_subspace_feats,
            curvs,
            eps=self.lorentz_eps,
        )

        text_distances = L.pairwise_dist_batch(
            text_subspace_feats,
            all_image_subspace_feats,
            curvs,
            eps=self.lorentz_eps,
        )

        # Apply product metric based on configured metric type
        if self.product_metric == "l1":
            # L1 product metric: mean of distances across concept spaces
            combined_image_distances = torch.mean(image_distances, dim=0)
            combined_text_distances = torch.mean(text_distances, dim=0)
        elif self.product_metric == "l2":
            # L2 product metric: sqrt of mean of squared distances across concept spaces
            combined_image_distances = torch.sqrt(torch.mean(image_distances**2, dim=0))
            combined_text_distances = torch.sqrt(torch.mean(text_distances**2, dim=0))
        else:
            raise ValueError(f"Unsupported product_metric: {self.product_metric}")

        # Convert distances to logits (negative distances)
        image_logits = -combined_image_distances
        text_logits = -combined_text_distances

        # Compute cross entropy loss using combined distances
        contrastive_loss = 0.5 * (
            nn.functional.cross_entropy(scale * image_logits, targets)
            + nn.functional.cross_entropy(scale * text_logits, targets)
        )

        # If boxes are used, add box contrastive losses
        if (
            self.use_boxes
            and box_image_subspace_feats is not None
            and box_text_subspace_feats is not None
        ):
            box_image_distances = L.pairwise_dist_batch(
                box_image_subspace_feats,
                all_text_subspace_feats,
                curvs,
                eps=self.lorentz_eps,
            )

            box_text_distances = L.pairwise_dist_batch(
                box_text_subspace_feats,
                all_image_subspace_feats,
                curvs,
                eps=self.lorentz_eps,
            )

            # Apply the same product metric for box features
            if self.product_metric == "l1":
                combined_box_image_distances = torch.mean(box_image_distances, dim=0)
                combined_box_text_distances = torch.mean(box_text_distances, dim=0)
            elif self.product_metric == "l2":
                combined_box_image_distances = torch.sqrt(
                    torch.mean(box_image_distances**2, dim=0)
                )
                combined_box_text_distances = torch.sqrt(
                    torch.mean(box_text_distances**2, dim=0)
                )

            box_image_logits = -combined_box_image_distances
            box_text_logits = -combined_box_text_distances

            box_contrastive_loss = 0.5 * (
                nn.functional.cross_entropy(scale * box_image_logits, targets)
                + nn.functional.cross_entropy(scale * box_text_logits, targets)
            )

            # Average the original and box contrastive losses
            contrastive_loss = 0.5 * (contrastive_loss + box_contrastive_loss)

        return contrastive_loss

    def _compute_entailment_loss_single(
        self,
        superior_subspace_feats: torch.Tensor,
        subordinate_subspace_feats: torch.Tensor,
        curvs: torch.Tensor,
        aperture_thresh: float = 0.7,
    ) -> torch.Tensor:
        """
        Compute entailment loss using uniform averaging.

        Args:
            superior_subspace_feats: Superior concept features (B, num_subspaces, subspace_dim)
            subordinate_subspace_feats: Subordinate concept features (B, num_subspaces, subspace_dim)
            curvs: Curvatures for each concept space (num_subspaces,)
            aperture_thresh: Threshold for aperture scaling

        Returns:
            Entailment loss uniformly averaged across all concept spaces
        """
        # Compute angles and apertures for all concept spaces
        angles = L.oxy_angle_batch(
            superior_subspace_feats,  # (batch, concept, dim)
            subordinate_subspace_feats,  # (batch, concept, dim)
            curvs,  # (num_subspaces,)
            eps=self.lorentz_eps,
        ).transpose(0, 1)  # (concept, batch) -> (batch, concept)

        apertures = L.half_aperture_batch(
            superior_subspace_feats,  # (batch, concept, dim)
            curvs,  # (num_subspaces,)
            eps=self.lorentz_eps,
        ).transpose(0, 1)  # (concept, batch) -> (batch, concept)

        # Compute entailment loss for each concept space
        entailment_losses = torch.clamp(
            angles - aperture_thresh * apertures, min=0
        )  # (batch, concept)

        # Use uniform averaging (traditional mean)
        return torch.mean(entailment_losses)

    def _compute_entailment_loss(
        self,
        text_subspace_feats: torch.Tensor,
        box_text_subspace_feats: Optional[torch.Tensor],
        image_subspace_feats: torch.Tensor,
        box_image_subspace_feats: Optional[torch.Tensor],
        curvs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute entailment loss for all concept spaces using mean-based approach.

        Args:
            text_subspace_feats: Text features of shape (B, num_subspaces, subspace_dim)
            box_text_subspace_feats: Box text features (B, num_subspaces, subspace_dim)
            image_subspace_feats: Image features of shape (B, num_subspaces, subspace_dim)
            box_image_subspace_feats: Box image features (B, num_subspaces, subspace_dim)
            curvs: Curvatures for each concept space (num_subspaces,)

        Returns:
            Entailment loss components using mean averaging
        """
        # Initialize all loss components to zero
        total_entailment_loss = torch.tensor(0.0, device=self.device)
        text_image_entailment_loss = torch.tensor(0.0, device=self.device)
        box_text_image_entailment_loss = torch.tensor(0.0, device=self.device)
        cross_image_entailment_loss = torch.tensor(0.0, device=self.device)
        cross_text_entailment_loss = torch.tensor(0.0, device=self.device)

        # Hyperparameters for apertures (from HyCoCLIP)
        _global_aperture_thresh = 0.7  # inter-modal
        _local_aperture_thresh = 1.2  # intra-modal

        # Compute text -> image entailment loss using mean
        # Text is the superior concept (more general), image is subordinate (more specific)
        text_image_entailment_loss = self._compute_entailment_loss_single(
            superior_subspace_feats=text_subspace_feats,
            subordinate_subspace_feats=image_subspace_feats,
            curvs=curvs,
            aperture_thresh=_global_aperture_thresh,
        )

        total_entailment_loss = text_image_entailment_loss

        # If boxes are used, add additional entailment losses using mean
        if (
            self.use_boxes
            and box_text_subspace_feats is not None
            and box_image_subspace_feats is not None
        ):
            # Box text -> box image entailment
            box_text_image_entailment_loss = self._compute_entailment_loss_single(
                superior_subspace_feats=box_text_subspace_feats,
                subordinate_subspace_feats=box_image_subspace_feats,
                curvs=curvs,
                aperture_thresh=_global_aperture_thresh,
            )

            # Cross-modal entailment: box_image -> image
            # Box image is superior (more general region), image is subordinate (specific region)
            cross_image_entailment_loss = self._compute_entailment_loss_single(
                superior_subspace_feats=box_image_subspace_feats,
                subordinate_subspace_feats=image_subspace_feats,
                curvs=curvs,
                aperture_thresh=_local_aperture_thresh,
            )

            # Cross-modal entailment: box_text -> text
            # Box text is superior (more general description), text is subordinate (specific description)
            cross_text_entailment_loss = self._compute_entailment_loss_single(
                superior_subspace_feats=box_text_subspace_feats,
                subordinate_subspace_feats=text_subspace_feats,
                curvs=curvs,
                aperture_thresh=_local_aperture_thresh,
            )

            # Combine all entailment losses
            total_entailment_loss = 0.5 * (
                text_image_entailment_loss
                + box_text_image_entailment_loss
                + cross_image_entailment_loss
                + cross_text_entailment_loss
            )

        return (
            total_entailment_loss,
            text_image_entailment_loss,
            box_text_image_entailment_loss,
            cross_image_entailment_loss,
            cross_text_entailment_loss,
        )

    def forward(
        self,
        images: torch.Tensor,
        box_images: torch.Tensor,
        tokens: torch.Tensor,
        box_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Forward pass computing losses for each concept space.

        Args:
            images: Image batch in BCHW format.
            tokens: List of tensors containing text tokens.
            box_images: Box image batch in BCHW format [optional].
            box_tokens: List of tensors containing box text tokens [optional].

        Returns:
            Dictionary containing loss and logging information.
        """
        # Validate box inputs
        if self.use_boxes:
            if box_images is None or box_tokens is None:
                raise ValueError(
                    "box_images and box_tokens must be provided when use_boxes=True"
                )

        # Clamp parameters to ensure stability
        for curv in self.curvs:
            curv.data = torch.clamp(
                curv.data,
                self._curv_minmax["min"].to(curv.data.device),
                self._curv_minmax["max"].to(curv.data.device),
            )

        for alpha in self.visual_concept_alphas:
            alpha.data = torch.clamp(alpha.data, max=0.0)
        for alpha in self.textual_concept_alphas:
            alpha.data = torch.clamp(alpha.data, max=0.0)

        # Clamp temperature such that logits are not scaled more than 100x.
        # ln(100) = ~4.6052
        self.logit_scale.data = torch.clamp(self.logit_scale.data, max=4.6052)
        _scale = self.logit_scale.exp()

        # Compute curvatures once for all concept spaces
        curvs = torch.stack([curv.exp() for curv in self.curvs])

        # Cache curvatures for use in encode methods
        self._cached_curvs = curvs

        # Get concept-specific features
        image_feats = self.encode_image(images, project=True)
        text_feats = self.encode_text(tokens, project=True)

        # Reshape to concept format for loss computation
        image_subspace_feats = image_feats.view(
            -1, self.num_subspaces, self.subspace_dim
        )
        text_subspace_feats = text_feats.view(-1, self.num_subspaces, self.subspace_dim)

        # Process box features if boxes are used
        box_image_subspace_feats = None
        box_text_subspace_feats = None
        if self.use_boxes:
            box_image_feats = self.encode_image(box_images, project=True)
            box_text_feats = self.encode_text(box_tokens, project=True)

            box_image_subspace_feats = box_image_feats.view(
                -1, self.num_subspaces, self.subspace_dim
            )
            box_text_subspace_feats = box_text_feats.view(
                -1, self.num_subspaces, self.subspace_dim
            )

        # Gather features from all GPUs
        all_image_feats = dist.gather_across_processes(image_feats)
        all_text_feats = dist.gather_across_processes(text_feats)

        # Concatenate features from all GPUs
        all_image_feats = torch.cat(all_image_feats, dim=0)
        all_text_feats = torch.cat(all_text_feats, dim=0)

        # Reshape gathered features to concept format
        all_image_subspace_feats = all_image_feats.view(
            -1, self.num_subspaces, self.subspace_dim
        )
        all_text_subspace_feats = all_text_feats.view(
            -1, self.num_subspaces, self.subspace_dim
        )

        # Compute losses for each concept space
        batch_size = image_subspace_feats.shape[0]
        targets = torch.arange(batch_size, device=self.device)
        targets = targets + batch_size * self._rank

        with torch.autocast(self.device.type, dtype=torch.float32):
            # Compute contrastive loss using product metric
            contrastive_loss = self._compute_product_metric_contrastive_loss(
                image_subspace_feats=image_subspace_feats,
                box_image_subspace_feats=box_image_subspace_feats,
                text_subspace_feats=text_subspace_feats,
                box_text_subspace_feats=box_text_subspace_feats,
                all_image_subspace_feats=all_image_subspace_feats,
                all_text_subspace_feats=all_text_subspace_feats,
                curvs=curvs,
                targets=targets,
                scale=_scale,
            )

            # Compute entailment loss for all concept spaces
            (
                entailment_loss,
                text_image_entailment_loss,
                box_text_image_entailment_loss,
                cross_image_entailment_loss,
                cross_text_entailment_loss,
            ) = self._compute_entailment_loss(
                text_subspace_feats=text_subspace_feats,
                box_text_subspace_feats=box_text_subspace_feats,
                image_subspace_feats=image_subspace_feats,
                box_image_subspace_feats=box_image_subspace_feats,
                curvs=curvs,
            )



            loss = contrastive_loss
            if self.entail_weight > 0:
                loss = loss + self.entail_weight * entailment_loss


        # Clear cached curvatures after forward pass
        self._cached_curvs = None

        output_dict = {
            "loss": loss,
            "logging": {
                "contrastive_loss": contrastive_loss,
                "text_image_entailment_loss": text_image_entailment_loss,
                "box_text_image_entailment_loss": box_text_image_entailment_loss,
                "cross_image_entailment_loss": cross_image_entailment_loss,
                "cross_text_entailment_loss": cross_text_entailment_loss,
                "entailment_loss": entailment_loss,
                "logit_scale": _scale,
                "curv": [curv.exp().item() for curv in self.curvs],
            },
        }

        return output_dict
