# ---------------------------------------
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ---------------------------------------

from .train_clip_vit_l import dataset, model, optim, train  # noqa: F401

model.visual.arch = "vit_base_patch16_224"
