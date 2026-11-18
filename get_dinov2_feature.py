#!/usr/bin/env python3
"""Extract DINOv2 (ViT-L/14) features for the OCT visual memory bank.

The script walks an image directory (all subdirectories are scanned) and
writes a feature pickle in the layout consumed by ``infer.py``:

    (filenames, features)

- ``filenames``: image file names exactly as found on disk
- ``features``: ``float32`` numpy vectors, one per file name

The OCT2017 4-class layout works out of the box: file names start with the
class name (``CNV-...jpeg``, ``DME-...jpeg``, ``DRUSEN-...jpeg``,
``NORMAL-...jpeg``) and ``infer.py`` recovers the label from that prefix.

Every image is resized to 224x224 with bicubic interpolation and normalized
with the ImageNet statistics, matching the torch.hub DINOv2 preprocessing.

The default output file is ``data/dinov2_vitl14_features_<split>_<mode>.pkl``.

Examples
--------
    python get_dinov2_feature.py --split test --mode last_layer ^
        --images "path\\to\\split"

    python get_dinov2_feature.py --split train --mode multi_layer ^
        --images "path\\to\\split"
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Multi-layer mode concatenates the CLS token of layers 12, 18 and 24.
# The torch.hub model exposes blocks 0-23, so layer L lives at index L - 1.
HUB_MULTI_LAYER_BLOCKS = (11, 17, 23)
TRANSFORMERS_MULTI_LAYER_LAYERS = (12, 18, 24)


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 ViT-L/14 features into the pkl format used by infer.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split name; it only selects the default output file name.",
    )
    parser.add_argument(
        "--images",
        required=True,
        help="Image directory of one split. Every image below it is processed recursively.",
    )
    parser.add_argument(
        "--mode",
        choices=["last_layer", "multi_layer"],
        default="multi_layer",
        help="last_layer: final CLS feature. multi_layer: CLS of layers 12/18/24, concatenated.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output pickle path. Defaults to data/dinov2_vitl14_features_<split>_<mode>.pkl.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "hub", "transformers"],
        default="auto",
        help="Model source: torch.hub (facebookresearch/dinov2) or transformers (facebook/dinov2-large).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda:0 or cpu. Defaults to cuda:0 when CUDA is available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N images (sorted) of the directory, useful for smoke tests.",
    )
    return parser


def load_model(backend):
    """Load DINOv2 ViT-L/14 and report which backend produced it."""
    if backend in ("auto", "hub"):
        try:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
            print("Loaded DINOv2 ViT-L/14 through torch.hub.")
            return model, "hub"
        except Exception as exc:  # noqa: BLE001 - fall back only when allowed
            if backend == "hub":
                raise
            print(f"torch.hub loading failed: {exc}")
            model = load_cached_hub_repo()
            if model is not None:
                return model, "hub"
            print("Falling back to transformers (facebook/dinov2-large) ...")
    from transformers import AutoModel

    try:
        model = AutoModel.from_pretrained("facebook/dinov2-large")
    except Exception as exc:  # noqa: BLE001 - surface an actionable message
        raise SystemExit(
            "Could not load DINOv2 through transformers either "
            f"({type(exc).__name__}: {exc}).\n"
            "Try --backend hub with a populated torch.hub cache, upgrade transformers "
            "(`pip install -U transformers`), or download facebook/dinov2-large once."
        ) from exc
    print("Loaded DINOv2 ViT-L/14 through transformers.")
    return model, "transformers"


def load_cached_hub_repo():
    """Load DINOv2 from a torch.hub clone that is already on disk (offline mode)."""
    cached_repo = os.path.join(torch.hub.get_dir(), "facebookresearch_dinov2_main")
    if not os.path.isdir(cached_repo):
        return None
    try:
        model = torch.hub.load(cached_repo, "dinov2_vitl14", source="local")
        print(f"Loaded DINOv2 ViT-L/14 from the cached torch.hub clone at {cached_repo}.")
        return model
    except Exception as exc:  # noqa: BLE001 - report and let the caller decide
        print(f"Loading the cached torch.hub clone failed: {exc}")
        return None


def get_transform():
    """Image preprocessing for the torch.hub DINOv2 model."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def final_norm_layer(model):
    """Return the transformer's final normalization layer when the model has one."""
    for attribute in ("norm", "layernorm"):
        layer = getattr(model, attribute, None)
        if layer is not None:
            return layer
    return None


def collect_images(images_dir, limit=None):
    """Return the sorted image paths below ``images_dir``."""
    paths = []
    for root, _, filenames in os.walk(images_dir):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, filename))
    paths.sort()
    if limit is not None:
        paths = paths[:limit]
    return paths


def feature_vector(image, model, backend, transform, mode):
    """Turn an image into a ``float32`` feature vector."""
    tensor = transform(image).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.inference_mode():
        if backend == "hub":
            vector = hub_feature_vector(model, tensor, mode)
        else:
            vector = transformers_feature_vector(model, tensor, mode)

    return vector.astype(np.float32)


def hub_feature_vector(model, tensor, mode):
    if mode == "last_layer":
        # The torch.hub DINOv2 models already return the normalized CLS feature.
        return model(tensor)[0].detach().cpu().numpy()

    tokens = model.prepare_tokens_with_masks(tensor)
    collected = []
    for block_index, block in enumerate(model.blocks):
        tokens = block(tokens)
        if block_index in HUB_MULTI_LAYER_BLOCKS:
            collected.append(model.norm(tokens[:, 0, :]))
    return torch.cat(collected, dim=-1)[0].detach().cpu().numpy()


def transformers_feature_vector(model, tensor, mode):
    outputs = model(pixel_values=tensor, output_hidden_states=True)

    if mode == "last_layer":
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is not None:
            return pooled[0].detach().cpu().numpy()
        return outputs.hidden_states[-1][0, 0].detach().cpu().numpy()

    norm_layer = final_norm_layer(model)
    collected = []
    for layer_index in TRANSFORMERS_MULTI_LAYER_LAYERS:
        cls_token = outputs.hidden_states[layer_index][:, 0, :]
        if norm_layer is not None:
            cls_token = norm_layer(cls_token)
        collected.append(cls_token)
    return torch.cat(collected, dim=-1)[0].detach().cpu().numpy()


def extract_all_images(paths, model, backend, transform, mode):
    """Turn image paths into the (filenames, features) pair."""
    filenames = []
    features = []
    processed = 0

    for image_path in paths:
        filename = os.path.basename(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
            features.append(feature_vector(image, model, backend, transform, mode))
            filenames.append(filename)
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} images.")
        except Exception as exc:  # noqa: BLE001 - skip a single unreadable image
            print(f"Failed: {image_path}: {exc}")

    print(f"Finished: {processed} images, {len(features)} feature vectors.")
    return filenames, features


def save_features(filenames, features, output_path):
    directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump([filenames, features], handle)
    print(f"Features saved to: {os.path.abspath(output_path)}")


def print_summary(filenames, features):
    if not features:
        return
    matrix = np.asarray(features, dtype=np.float32)
    print("=" * 60)
    print(f"Feature vectors:  {matrix.shape[0]}")
    print(f"Feature shape:    {matrix.shape[1]}")
    print(f"Feature dtype:    {matrix.dtype}")
    print(
        "Mean / Std / Min / Max: "
        f"{matrix.mean():.4f} / {matrix.std():.4f} / {matrix.min():.4f} / {matrix.max():.4f}"
    )
    print("Sample file names:")
    for name in filenames[:5]:
        print(f"  - {name}")
    print("=" * 60)


def main(argv=None):
    args = build_argparser().parse_args(argv)

    images_dir = os.path.expanduser(os.path.expandvars(args.images))
    if not os.path.isdir(images_dir):
        raise SystemExit(f"Image directory not found: {images_dir}")

    if args.output:
        output_path = os.path.expanduser(os.path.expandvars(args.output))
    else:
        output_path = os.path.join(
            DATA_DIR, f"dinov2_vitl14_features_{args.split}_{args.mode}.pkl"
        )

    paths = collect_images(images_dir, limit=args.limit)
    if not paths:
        raise SystemExit(f"No images found below: {images_dir}")

    model, backend = load_model(args.backend)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    model.eval()

    print(f"Backend: {backend}")
    print(f"Feature mode: {args.mode}")
    print(f"Images: {images_dir} ({len(paths)} files)")

    filenames, features = extract_all_images(paths, model, backend, get_transform(), args.mode)
    if not features:
        raise SystemExit("No features were extracted; please check --images.")

    print_summary(filenames, features)
    save_features(filenames, features, output_path)


if __name__ == "__main__":
    main()
