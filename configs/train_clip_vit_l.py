# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

from phyclip.config import LazyCall as L
from phyclip.encoders.image_encoders import build_timm_vit
from phyclip.encoders.text_encoders import TransformerTextEncoder
from phyclip.models import CLIPBaseline

from .train_phyclip_vit_l import dataset, optim, train  # noqa: F401

model = L(CLIPBaseline)(
    visual=L(build_timm_vit)(
        arch="vit_large_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512", vocab_size=49408, context_length=77
    ),
    embed_dim=512,
    use_boxes=False,  # Set to True to use box data for training data augmentation
)

optim.optimizer.params.exclude_params = ["logit_scale"]
