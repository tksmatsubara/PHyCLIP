import torch
from torch.nn import functional as F

from phyclip import lorentz as L
from phyclip.models import MERU, CLIPBaseline, HyCoCLIP, PHyCLIP


def process_text_features(
    model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Process text features based on model type.

    Args:
        model: The model instance
        text_features: Raw text features from the model

    Returns:
        Processed text features ready for similarity computation
    """
    if isinstance(model, PHyCLIP):
        # map class features to each concept's hyperboloid
        class_feats = text_features.mean(dim=0).view(
            -1, model.num_subspaces, model.subspace_dim
        )

        class_feats = class_feats * torch.stack(
            [alpha.exp() for alpha in model.textual_concept_alphas]
        ).view(1, -1, 1)

        curvs = torch.stack([curv.exp() for curv in model.curvs])
        class_feats = L.exp_map0_batch(class_feats, curvs).view(model.num_subspaces, -1)
        return class_feats
    elif isinstance(model, (HyCoCLIP, MERU)):
        # Ensemble in the tangent space, then project to Hyperboloid.
        class_feats = text_features.mean(dim=0)
        class_feats = class_feats * model.textual_alpha.exp()
        class_feats = L.exp_map0(class_feats, model.curv.exp())
        return class_feats
    else:
        # Ensemble prompt features: normalize -> average -> normalize.
        class_feats = F.normalize(text_features, dim=-1)
        class_feats = class_feats.mean(dim=0)
        class_feats = F.normalize(class_feats, dim=-1)
        return class_feats


def compute_similarity_scores(
    model: PHyCLIP | HyCoCLIP | MERU | CLIPBaseline,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """
    Compute similarity scores between image and text features based on model type.

    Args:
        model: The model instance
        image_features: Image features
        text_features: Processed text features (classifier)

    Returns:
        Similarity scores
    """

    if isinstance(model, PHyCLIP):
        curvs = torch.stack([curv.exp() for curv in model.curvs])
        image_features = image_features.view(
            -1, model.num_subspaces, model.subspace_dim
        )
        text_features = text_features.view(-1, model.num_subspaces, model.subspace_dim)
        scores = L.pairwise_inner_batch(image_features, text_features, curvs).sum(dim=0)
        return scores
    elif isinstance(model, (HyCoCLIP, MERU)):
        scores = L.pairwise_inner(image_features, text_features, model.curv.exp())
        return scores
    else:
        scores = image_features @ text_features.T
        return scores
