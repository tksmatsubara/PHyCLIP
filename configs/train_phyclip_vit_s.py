from .train_phyclip_vit_l import dataset, model, optim, train  # noqa: F401

model.visual.arch = "vit_small_mocov3_patch16_224"
