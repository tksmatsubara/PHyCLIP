from .train_phyclip_vit_l import dataset, model, optim, train  # noqa: F401

model.visual.arch = "vit_base_patch16_224"
