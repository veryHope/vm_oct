#!/usr/bin/env python3
"""Extract DINOv3 (ViT-L/16) features for the OCT visual memory bank.

The script walks an image directory (all subdirectories are scanned) and
writes a feature pickle in the layout consumed by ``infer.py``:

    (filenames, features)

- ``filenames``: image file names exactly as found on disk
- ``features``: ``float32`` numpy vectors, one per file name

The OCT2017 4-class layout works out of the box: file names start with the
class name (``CNV-...jpeg``, ``DME-...jpeg``, ``DRUSEN-...jpeg``,
``NORMAL-...jpeg``) and ``infer.py`` recovers the label from that prefix.

Images are preprocessed by the DINOv3 image processor and loaded through
``transformers``, which must be recent enough to know the ``dinov3``
architecture. Weights can come from a local directory or a Hugging Face id.

The default output file is ``data/dinov3_features_<split>_<mode>.pkl``.

Examples
--------
    python get_dinov3_feature.py --split test --mode multi_layer ^
        --model "./models/dinov3-vitl16-pretrain-lvd1689m" ^
        --images "path\\to\\split"

    python get_dinov3_feature.py --split train --mode last_layer ^
        --model "facebook/dinov3-vitl16-pretrain-lvd1689m" ^
        --images "path\\to\\split"
"""

import argparse
import os
import pickle

import numpy as np
import torch
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Multi-layer mode concatenates the [CLS] token of layers 12, 18 and 24.
MULTI_LAYER_LAYERS = (12, 18, 24)


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Extract DINOv3 ViT-L/16 features into the pkl format used by infer.py.",
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
        help="last_layer: final [CLS] feature. multi_layer: [CLS] of layers 12/18/24, concatenated.",
    )
    parser.add_argument(
        "--model",
        default="./models/dinov3-vitl16-pretrain-lvd1689m",
        help="Local model directory or Hugging Face model id for the DINOv3 ViT-L/16 weights.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output pickle path. Defaults to data/dinov3_features_<split>_<mode>.pkl.",
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


def resolve_model_id(model_path):
    """Accept a local directory, a Hugging Face id, or a bare model name."""
    if os.path.isdir(model_path):
        return model_path
    local_path = os.path.join(SCRIPT_DIR, model_path)
    if os.path.isdir(local_path):
        return local_path
    name = os.path.basename(model_path.strip("/\\"))
    if name and not model_path.startswith("facebook/"):
        return f"facebook/{name}"
    return model_path


def load_model(model_id):
    """Load the DINOv3 processor and model."""
    try:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_id)
        try:
            model = AutoModel.from_pretrained(model_id, device_map="auto")
        except (ImportError, ValueError) as exc:
            print(f"device_map='auto' is unavailable ({exc}); loading on a single device instead.")
            model = AutoModel.from_pretrained(model_id)
    except Exception as exc:  # noqa: BLE001 - surface an actionable message
        raise SystemExit(
            f"Could not load DINOv3 from {model_id!r} ({type(exc).__name__}: {exc}).\n"
            "The installed transformers may be too old for DINOv3 "
            "(`pip install -U transformers`), or --model may point to a missing directory."
        ) from exc
    model.eval()
    return processor, model


def final_norm_layer(model):
    """Return the transformer's final normalization layer when the model has one."""
    for attribute in ("norm", "layernorm"):
        layer = getattr(model, attribute, None)
        if layer is not None:
            return layer
    vision_model = getattr(model, "vision_model", None)
    if vision_model is not None:
        for attribute in ("norm", "layernorm"):
            layer = getattr(vision_model, attribute, None)
            if layer is not None:
                return layer
    return None


def extract_cls_with_final_norm(hidden_state, norm_layer):
    cls_token = hidden_state[:, 0, :]
    if norm_layer is not None:
        cls_token = norm_layer(cls_token)
    return cls_token[0].detach().cpu().numpy()


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


def feature_vector(image, processor, model, mode):
    """Turn an image into a ``float32`` feature vector."""
    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states
    norm_layer = final_norm_layer(model)

    if mode == "last_layer":
        return extract_cls_with_final_norm(hidden_states[-1], norm_layer).astype(np.float32)

    if len(hidden_states) <= max(MULTI_LAYER_LAYERS):
        raise ValueError(
            f"Model returned {len(hidden_states)} hidden states; "
            f"cannot extract layers {MULTI_LAYER_LAYERS}."
        )
    parts = [
        extract_cls_with_final_norm(hidden_states[index], norm_layer)
        for index in MULTI_LAYER_LAYERS
    ]
    return np.concatenate(parts).astype(np.float32)


def extract_all_images(paths, processor, model, mode):
    """Turn image paths into the (filenames, features) pair."""
    filenames = []
    features = []
    processed = 0

    for image_path in paths:
        filename = os.path.basename(image_path)
        try:
            image = Image.open(image_path).convert("RGB")
            features.append(feature_vector(image, processor, model, mode))
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
            DATA_DIR, f"dinov3_features_{args.split}_{args.mode}.pkl"
        )

    paths = collect_images(images_dir, limit=args.limit)
    if not paths:
        raise SystemExit(f"No images found below: {images_dir}")

    model_id = resolve_model_id(args.model)
    processor, model = load_model(model_id)
    if args.device:
        model = model.to(torch.device(args.device))
    elif torch.cuda.is_available() and next(model.parameters()).device.type == "cpu":
        model = model.to("cuda:0")

    print(f"Model: {model_id}")
    print(f"Feature mode: {args.mode}")
    print(f"Images: {images_dir} ({len(paths)} files)")

    filenames, features = extract_all_images(paths, processor, model, args.mode)
    if not features:
        raise SystemExit("No features were extracted; please check --images.")

    print_summary(filenames, features)
    save_features(filenames, features, output_path)


if __name__ == "__main__":
    main()
