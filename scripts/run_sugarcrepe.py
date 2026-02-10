import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from phyclip.config import LazyConfig, LazyFactory
from phyclip.tokenizer import Tokenizer
from phyclip.utils.checkpointing import CheckpointManager

script_dir = Path(__file__).parent
repo_root = script_dir.parent
coco_image_root = repo_root / "datasets" / "eval" / "coco" / "val2017"
data_root = repo_root / "phyclip" / "evaluation" / "sugar_crepe" / "data"


# Retrieval function based on sugarcrepe implementation using PHyCLIP's similarity score
def text_retrieval(pos_text, neg_text, image, model, tokenizer, transform, device):
    pos_tokens = tokenizer([pos_text])
    pos_text_embedding = model.encode_text(pos_tokens, project=True)

    neg_tokens = tokenizer([neg_text])
    neg_text_embedding = model.encode_text(neg_tokens, project=True)

    image_tensor = transform(image).unsqueeze(dim=0).to(device)
    image_embedding = model.encode_image(image_tensor, project=True)

    from phyclip.utils.evaluation import compute_similarity_scores

    pos_score = compute_similarity_scores(model, image_embedding, pos_text_embedding)[
        0, 0
    ]

    neg_score = compute_similarity_scores(model, image_embedding, neg_text_embedding)[
        0, 0
    ]

    return 1 if pos_score.item() > neg_score.item() else 0


def evaluate(image_root, dataset, model, tokenizer, transform, device):
    metrics = {}
    for c, data_dict in dataset.items():
        print(f"Evaluating {c}...")
        correct_cnt = 0
        for i, data in tqdm(data_dict.items(), desc=f"evaluating {c}"):
            image_path = os.path.join(image_root, data["filename"])
            image = Image.open(image_path)
            correct = text_retrieval(
                data["caption"],
                data["negative_caption"],
                image,
                model,
                tokenizer,
                transform,
                device,
            )
            correct_cnt += correct
        count = len(data_dict)
        metrics[c] = correct_cnt / count
        print(f"  {c}: {metrics[c]:.4f} ({correct_cnt}/{count})")

    total_correct = sum(metrics[cat] * len(dataset[cat]) for cat in metrics.keys())
    total_count = sum(len(dataset[cat]) for cat in dataset.keys())
    metrics["overall"] = total_correct / total_count
    print(f"Overall accuracy: {metrics['overall']:.4f}")

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SugarCrepe with PHyCLIP using direct SugarCrepe import."
    )
    parser.add_argument(
        "--train-config", required=True, help="Path to PHyCLIP train config (py/yaml)."
    )
    parser.add_argument(
        "--checkpoint-path", required=True, help="Path to PHyCLIP checkpoint."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cfg = LazyConfig.load(args.train_config)
    model = LazyFactory.build_model(train_cfg, device).eval().to(device)
    CheckpointManager(model=model).load(args.checkpoint_path)
    tokenizer = Tokenizer()

    transform = transforms.Compose(
        [
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
        ]
    )

    checkpoint_path = Path(args.checkpoint_path)
    save_dir = checkpoint_path.parent / "sugarcrepe_results"

    data_dict = {
        "add_obj": f"{data_root}/add_obj.json",
        "add_att": f"{data_root}/add_att.json",
        "replace_obj": f"{data_root}/replace_obj.json",
        "replace_att": f"{data_root}/replace_att.json",
        "replace_rel": f"{data_root}/replace_rel.json",
        "swap_obj": f"{data_root}/swap_obj.json",
        "swap_att": f"{data_root}/swap_att.json",
    }

    dataset = {}
    for c, data_path in data_dict.items():
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                dataset[c] = json.load(f)

    if not dataset:
        raise FileNotFoundError(f"No SugarCrepe data found in {data_root}")

    if not coco_image_root.exists():
        raise RuntimeError(
            f"COCO image root not found: {coco_image_root}. Please ensure COCO val2017 images are available."
        )

    metrics = evaluate(
        str(coco_image_root), dataset, model, tokenizer, transform, device
    )
    scores = None

    save_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(scores, torch.Tensor):
        torch.save(scores.detach().cpu(), save_dir / "scores.pt")
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
