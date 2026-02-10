"""
Interpolate between points using a trained HyCoCLIP, MERU or CLIP model,
and a pool of text and images (and their encoded representations).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from enum import Enum
from io import BytesIO

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from phyclip import lorentz as L
from phyclip.config import LazyConfig, LazyFactory
from phyclip.models import MERU, CLIPBaseline, HyCoCLIP, PHyCLIP
from phyclip.tokenizer import Tokenizer
from phyclip.utils.checkpointing import CheckpointManager
from phyclip.utils.evaluation import compute_similarity_scores

parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA(
    "--checkpoint-path",
    help="Path to checkpoint of a trained HyCoCLIP/MERU/CLIP model.",
)
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--image-path", help="Path to an image (.jpg) for perfoming traversal.")
_AA(
    "--target-image-path",
    help="Path to an image (.jpg) for perfoming traversal to this target image.",
)
_AA("--steps", type=int, default=50, help="Number of traversal steps.")
_AA(
    "--download-data",
    action="store_true",
    help="Download the data for the Flickr dataset.",
)
_AA("--data-path", help="Path to download the data for the Flickr dataset.")
_AA(
    "--feats-path",
    default="./datasets/flickr_grounded/flickr_feats.pt",
    help="Path to the features for the Flickr dataset.",
)
_AA(
    "--image-to-image-traversal",
    action="store_true",
    help="Do interpolated traversal from image to image.",
)


def interpolate(model, feats: torch.Tensor, root_feat: torch.Tensor, steps: int):
    """
    Interpolate between given feature vector and `[ROOT]` depending on model type.
    """

    # Linear interpolation between root and image features. For HyCoCLIP and MERU,
    # this happens in the tangent space of the origin.
    if isinstance(model, PHyCLIP):
        curvs = torch.stack([curv.exp() for curv in model.curvs])
        feats = feats.view(-1, model.num_subspaces, model.subspace_dim)
        feats = L.log_map0_batch(feats, curvs)
    elif isinstance(model, (HyCoCLIP, MERU)):
        feats = L.log_map0(feats, model.curv.exp())

    # interpolate between root and image features
    if isinstance(model, PHyCLIP):
        # Reshape root_feat to match concept spaces
        root_feat_concepts = root_feat.view(model.num_subspaces, model.subspace_dim)

        # interpolate between each concept space
        interp_feats_list = []
        for i in range(model.num_subspaces):
            interp_feats_i = [
                torch.lerp(root_feat_concepts[i], feats[:, i], weight.item())
                for weight in torch.linspace(0.0, 1.0, steps=steps)
            ]
            interp_feats_i = torch.stack(interp_feats_i)  # (steps, subspace_dim)
            interp_feats_list.append(interp_feats_i)

        # Stack along concept dimension: (steps, num_subspaces, subspace_dim)
        interp_feats = torch.stack(interp_feats_list, dim=1).squeeze(2)
    else:
        interp_feats = [
            torch.lerp(root_feat, feats, weight.item())
            for weight in torch.linspace(0.0, 1.0, steps=steps)
        ]
        interp_feats = torch.stack(interp_feats)  # (steps, subspace_dim)

    # Lift on the Hyperboloid (for PHyCLIP, HyCoCLIP and MERU), or L2 normalize (for CLIP).
    if isinstance(model, PHyCLIP):
        # Reshape for exp_map0_batch: (steps, num_subspaces, subspace_dim) -> (steps, num_subspaces, subspace_dim)
        # exp_map0_batch expects (B, N, D) where B=steps, N=num_subspaces, D=subspace_dim
        interp_feats = L.exp_map0_batch(interp_feats, curvs)
        # Reshape back to (steps, embed_dim)
        interp_feats = interp_feats.view(steps, -1)
    elif isinstance(model, (HyCoCLIP, MERU)):
        interp_feats = L.exp_map0(interp_feats, model.curv.exp())
    else:
        interp_feats = torch.nn.functional.normalize(interp_feats, dim=-1)

    # Reverse the traversal order: (image first, root last)
    return interp_feats.flip(0)


def calc_scores(
    model, image_feats: torch.Tensor, all_feats: torch.Tensor, has_root: bool
):
    """
    Calculate similarity scores between the input image and dataset features depending
    on model type.

    Args:
        has_root: Flag to indicate whether the last text embedding (at dim=0)
            is the `[ROOT]` embedding.
    """

    all_scores = []

    if isinstance(model, (PHyCLIP, HyCoCLIP, MERU)):
        for feats_batch in all_feats.split(65536):
            scores = compute_similarity_scores(model, image_feats, feats_batch)
            all_scores.append(scores)

        all_scores = torch.cat(all_scores, dim=1)
        return all_scores
    else:
        # model is not needed here.
        return image_feats @ all_feats.T


_INTER_STR_TO_CV2 = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "bilinear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
    "lanczos4": cv2.INTER_LANCZOS4,
}


def inter_str_to_cv2(inter_str):
    inter_str = inter_str.lower()
    if inter_str not in _INTER_STR_TO_CV2:
        raise ValueError(f"Invalid option for interpolation: {inter_str}")
    return _INTER_STR_TO_CV2[inter_str]


class ResizeMode(Enum):
    no = 0  # pylint: disable=invalid-name
    keep_ratio = 1  # pylint: disable=invalid-name
    center_crop = 2  # pylint: disable=invalid-name
    border = 3  # pylint: disable=invalid-name
    keep_ratio_largest = 4  # pylint: disable=invalid-name


class Resizer:
    def __init__(
        self,
        image_size,
        resize_mode,
        resize_only_if_bigger,
        upscale_interpolation="lanczos",
        downscale_interpolation="area",
        encode_quality=95,
        skip_reencode=False,
        min_image_size=0,
        max_image_area=float("inf"),
        max_aspect_ratio=float("inf"),
    ):
        self.image_size = image_size

        if isinstance(resize_mode, str):
            if resize_mode not in ResizeMode.__members__:  # pylint: disable=unsupported-membership-test
                raise ValueError(f"Invalid option for resize_mode: {resize_mode}")
            resize_mode = ResizeMode[resize_mode]

        self.resize_mode = resize_mode
        self.min_image_size = min_image_size
        self.max_image_area = max_image_area
        self.max_aspect_ratio = max_aspect_ratio
        self.resize_only_if_bigger = resize_only_if_bigger
        cv2_img_quality = int(cv2.IMWRITE_JPEG_QUALITY)
        self.encode_params = [cv2_img_quality, encode_quality]
        self.what_ext = "jpeg"
        self.skip_reencode = skip_reencode

        self.upscale_interpolation = inter_str_to_cv2(upscale_interpolation)
        self.downscale_interpolation = inter_str_to_cv2(downscale_interpolation)

    def __call__(self, img):
        cv2.setNumThreads(1)
        if img is None:
            raise ValueError("Image decoding error")
        if len(img.shape) == 3 and img.shape[-1] == 4:
            # alpha matting with white background
            alpha = img[:, :, 3, np.newaxis]
            img = alpha / 255 * img[..., :3] + 255 - alpha
            img = np.rint(img.clip(min=0, max=255)).astype(np.uint8)

        original_height, original_width = img.shape[:2]
        # check if image is too small
        if min(original_height, original_width) < self.min_image_size:
            return None, None, None, None, None, "image too small"
        if original_height * original_width > self.max_image_area:
            return None, None, None, None, None, "image area too large"
        # check if wrong aspect ratio
        if (
            max(original_height, original_width) / min(original_height, original_width)
            > self.max_aspect_ratio
        ):
            return None, None, None, None, None, "aspect ratio too large"

        # resizing in following conditions
        if self.resize_mode in (ResizeMode.keep_ratio, ResizeMode.center_crop):
            downscale = min(original_width, original_height) > self.image_size
            if not self.resize_only_if_bigger or downscale:
                interpolation = (
                    self.downscale_interpolation
                    if downscale
                    else self.upscale_interpolation
                )
                img = A.smallest_max_size(
                    img, self.image_size, interpolation=interpolation
                )
                if self.resize_mode == ResizeMode.center_crop:
                    img = A.center_crop(img, self.image_size, self.image_size)
        elif self.resize_mode in (ResizeMode.border, ResizeMode.keep_ratio_largest):
            downscale = max(original_width, original_height) > self.image_size
            if not self.resize_only_if_bigger or downscale:
                interpolation = (
                    self.downscale_interpolation
                    if downscale
                    else self.upscale_interpolation
                )
                img = A.longest_max_size(
                    img, self.image_size, interpolation=interpolation
                )
                if self.resize_mode == ResizeMode.border:
                    img = A.pad(
                        img,
                        self.image_size,
                        self.image_size,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=[255, 255, 255],
                    )

        height, width = img.shape[:2]
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img)

        return img, width, height, original_width, original_height, None


@torch.inference_mode()
def get_data_feats(
    device, resizer: Resizer, tsv_path: str, model: HyCoCLIP | MERU | CLIPBaseline
) -> tuple[list[str], torch.Tensor]:
    tokenizer = Tokenizer()
    image_transform = T.Compose(
        [T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()]
    )

    crop_dim_cutoff = 32 * 32
    total_shards = (
        17  # Number of shards in the dataset. 17 for Flickr grounded dataset.
    )
    min_batch_size = 512
    previous_img_file_name = ""
    item_list = []
    representation_list = []

    images_to_encode = []
    texts_to_encode = []
    image_items = []
    text_items = []

    for shard_num in range(total_shards):
        sample_train_shard = pd.read_csv(
            f"{tsv_path}/train-{shard_num:04d}.tsv",
            sep="\t",
            header=None,
            names=["data_id", "data"],
        )
        samples_in_shard = len(sample_train_shard["data"])

        for idx in tqdm(
            range(samples_in_shard),
            desc=f"Processing shard {shard_num + 1}/{total_shards}",
        ):
            sample_data = sample_train_shard["data"][idx]
            sample_data = json.loads(sample_data)
            img_file_name = sample_data["file_name"]

            if img_file_name != previous_img_file_name:
                previous_img_file_name = img_file_name
                image_bytes = base64.b64decode(
                    bytes(sample_data["image"], encoding="raw_unicode_escape")
                )
                image = Image.open(BytesIO(image_bytes))
                image_tensor, _, _, _, _, error = resizer(np.array(image))
                image_tensor = image_transform(image_tensor)
                images_to_encode.append(image_tensor)
                image_items.append(img_file_name)

                bboxes_of_image = []
                texts_of_image = []

            img_caption = sample_data["caption"]
            texts_to_encode.append(img_caption)
            text_items.append(img_caption)

            for n, annotation in enumerate(sample_data["annos"]):
                bbox_dims = annotation["bbox"]
                if str(bbox_dims) not in bboxes_of_image:
                    bboxes_of_image.append(str(bbox_dims))
                    left = bbox_dims[0]
                    top = bbox_dims[1]
                    right = left + bbox_dims[2]
                    bottom = top + bbox_dims[3]

                    if (right - left) * (bottom - top) >= crop_dim_cutoff:
                        entity_image = image.crop((left, top, right, bottom))
                        entity_image, _, _, _, _, error = resizer(
                            np.array(entity_image)
                        )
                        entity_image = image_transform(entity_image)
                        images_to_encode.append(entity_image)
                        image_items.append(f"{img_file_name}_{str(bbox_dims)}")

                tokens_positive = annotation["tokens_positive"][0]
                entity_text = img_caption[tokens_positive[0] : tokens_positive[1]]
                if entity_text not in texts_of_image:
                    texts_of_image.append(entity_text)
                    texts_to_encode.append(entity_text)
                    text_items.append(entity_text)

            if len(images_to_encode) >= min_batch_size:
                representation_list.append(
                    model.encode_image(
                        torch.stack(images_to_encode).to(device), project=True
                    )
                )
                item_list.extend(image_items)
                images_to_encode = []
                image_items = []

            if len(texts_to_encode) >= min_batch_size:
                text_tokens = tokenizer(texts_to_encode)
                representation_list.append(model.encode_text(text_tokens, project=True))
                item_list.extend(text_items)
                texts_to_encode = []
                text_items = []

    representation_list.append(
        model.encode_image(torch.stack(images_to_encode).to(device), project=True)
    )
    item_list.extend(image_items)
    text_tokens = tokenizer(texts_to_encode)
    representation_list.append(model.encode_text(text_tokens, project=True))
    item_list.extend(text_items)

    item_feats = torch.cat(representation_list, dim=0)

    return item_list, item_feats


@torch.inference_mode()
def main(_A: argparse.Namespace):
    resizer = Resizer(
        image_size=224,
        resize_mode="border",
        resize_only_if_bigger=False,
        upscale_interpolation="lanczos",
        downscale_interpolation="area",
        encode_quality=95,
        skip_reencode=False,
        min_image_size=0,
        max_image_area=float("inf"),
        max_aspect_ratio=float("inf"),
    )

    # Get the current device (this will be `cuda:0` here by default) or use CPU.
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create the model using training config and load pre-trained weights.
    _C_TRAIN = LazyConfig.load(_A.train_config)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()

    CheckpointManager(model=model).load(_A.checkpoint_path)

    if isinstance(model, (PHyCLIP, HyCoCLIP, MERU)):
        root_feat = torch.zeros(_C_TRAIN.model.embed_dim, device=device)
    else:
        # CLIP model checkpoint should have the 'root' embedding.
        root_feat = torch.load(_A.checkpoint_path)["root"].to(device)

    if not os.path.exists(_A.feats_path):
        if _A.download_data:
            snapshot_download(
                repo_id="gligen/flickr_tsv",
                repo_type="dataset",
                local_dir=_A.data_path,
                local_dir_use_symlinks=False,
            )

        item_list, item_feats = get_data_feats(device, resizer, _A.data_path, model)
        torch.save((item_list, item_feats), _A.feats_path)
    else:
        item_list, item_feats = torch.load(_A.feats_path)

    # Add [ROOT] to the pool of text feats.
    item_list.append("[ROOT]")
    item_feats = torch.cat([item_feats, root_feat[None, ...]])

    print(f"Total items in item_list: {len(item_list)}")
    print(f"Size of item_feats: {item_feats.size()}")

    # ------------------------------------------------------------------------
    print(f"\nPerforming image to root traversals with source: {_A.image_path}...")
    # ------------------------------------------------------------------------
    image = Image.open(_A.image_path).convert("RGB")

    image_transform = T.Compose(
        [T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()]
    )
    image, _, _, _, _, error = resizer(np.array(image))
    image = image_transform(image).to(device)
    image_feats = model.encode_image(image[None, ...], project=True)[0]

    interp_feats = interpolate(model, image_feats, root_feat, _A.steps)
    nn1_scores = calc_scores(model, interp_feats, item_feats, has_root=True)

    nn1_scores, _nn1_idxs = nn1_scores.max(dim=-1)
    nn1_texts = [item_list[_idx.item()] for _idx in _nn1_idxs]

    # De-duplicate retrieved texts (multiple points may have same NN) and print.
    print("Texts retrieved from [IMAGE] -> [ROOT] traversal:")
    unique_nn1_texts = []
    for _text in nn1_texts:
        if _text not in unique_nn1_texts:
            unique_nn1_texts.append(_text)
            print(f"  - {_text}")

    if _A.image_to_image_traversal:
        # ------------------------------------------------------------------------
        print(
            f"\nPerforming image to image traversals with source: {_A.image_path} and target: {_A.target_image_path}..."
        )
        # ------------------------------------------------------------------------
        image = Image.open(_A.image_path).convert("RGB")
        target_image = Image.open(_A.target_image_path).convert("RGB")

        image_transform = T.Compose(
            [
                T.Resize(224, T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
            ]
        )
        image, _, _, _, _, error = resizer(np.array(image))
        image = image_transform(image).to(device)
        image_feats = model.encode_image(image[None, ...], project=True)[0]

        target_image, _, _, _, _, error = resizer(np.array(target_image))
        target_image = image_transform(target_image).to(device)
        target_image_feats = model.encode_image(target_image[None, ...], project=True)[
            0
        ]

        interp_feats = interpolate(model, image_feats, target_image_feats, _A.steps)
        nn1_scores = calc_scores(model, interp_feats, item_feats, has_root=True)

        nn1_scores, _nn1_idxs = nn1_scores.max(dim=-1)
        nn1_texts = [item_list[_idx.item()] for _idx in _nn1_idxs]

        # De-duplicate retrieved texts (multiple points may have same NN) and print.
        print("Texts retrieved from [SOURCE IMAGE] -> [TARGET IMAGE] traversal:")
        unique_nn1_texts = []
        for _text in nn1_texts:
            if _text not in unique_nn1_texts:
                unique_nn1_texts.append(_text)
                print(f"  - {_text}")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
