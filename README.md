## A flexible visual memory framework for OCT-based retinal disease classification using vision foundation models

This repository contains the public code for our OCT classification framework.
The method uses frozen vision foundation models, an expandable visual memory bank, LDA-based feature transformation, and nearest-neighbor retrieval with class-balanced voting for OCT-based retinal disease classification.

On the public 110K dataset, DINOv2 multi-layer features with LDA achieved 96.1% accuracy.
When two new disease classes from a public 6K dataset were added, the framework maintained strong performance with only 0.3% forgetting on the original classes, supporting a flexible and clinically adaptable classification pipeline.

The repository ships:

- `infer.py` runs the KNN inference over pre-extracted feature files.
- `get_dinov2_feature.py` and `get_dinov3_feature.py` generate those feature files with DINOv2 ViT-L/14 and DINOv3 ViT-L/16.

## Features

The bullets follow the pipeline order: foundation models first, then feature extraction, then retrieval.

- **Frozen foundation models**: DINOv2 ViT-L/14 and DINOv3 ViT-L/16 features from a single forward pass, with no fine-tuning and no training loop.
- **Feature extraction included**: `get_dinov2_feature.py` and `get_dinov3_feature.py` turn an image folder into the `data/` pickles, on CPU or GPU.
- **Expandable visual memory bank**: new classes are added by appending feature `pkl` files, so earlier classes stay intact across dataset updates.
- **LDA feature transformation**: optional supervised projection before retrieval (`--enable-lda`).
- **Class-balanced KNN voting**: plurality, distance, softmax and rank strategies, weighted so frequent classes do not dominate the vote.

## Generating Feature Files

The `data/` directory is populated by the two feature-extraction scripts. Each script walks an image directory (all subdirectories are scanned) and writes one pickle per split and feature mode. The OCT2017 4-class layout works out of the box: file names start with the class name (`CNV-...jpeg`, `DME-...jpeg`, `DRUSEN-...jpeg`, `NORMAL-...jpeg`), which is how `infer.py` recovers the ground-truth class.

```bash
# data/dinov2_vitl14_features_test_multi_layer.pkl
python get_dinov2_feature.py ^
  --split test ^
  --mode multi_layer ^
  --images "path\to\split"

# data/dinov3_features_test_multi_layer.pkl
python get_dinov3_feature.py ^
  --split test ^
  --mode multi_layer ^
  --model "./models/dinov3-vitl16-pretrain-lvd1689m" ^
  --images "path\to\split"
```

`--images` is the directory that holds the images of one split (any depth below it is scanned).

- `--mode last_layer` stores the final `CLS` feature; `--mode multi_layer` concatenates the `CLS` features of layers 12, 18 and 24 (3x the feature dimension).
- Images are resized to 224x224 with bicubic interpolation and normalized with the ImageNet statistics.
- File names are stored exactly as found on disk, so `infer.py` can parse the class from the name prefix (`--labels` controls the label list).
- `--limit N` processes only the first N images, which is handy for a quick sanity check.
- DINOv2 loads through `torch.hub` and falls back to `transformers` (`--backend transformers` forces the fallback). DINOv3 loads through `transformers` and needs a recent release.
- The output file is named `data/<model>_features_<split>_<mode>.pkl`, matching the paths used by `infer.py`.

## Feature File Format

Each `pkl` file should contain a tuple:

```python
(filenames, features)
```

- `filenames`: list of file names
- `features`: `numpy.ndarray` or a list of feature vectors

## Installation

```bash
pip install -r requirements.txt
```

The requirements include PyTorch and torchvision for feature extraction. DINOv2 loads through `torch.hub` and DINOv3 needs a recent `transformers` release.

## Usage

```bash
python infer.py ^
  --train "data\\dinov2_vitl14_features_train_multi_layer.pkl" ^
  --test "data\\dinov2_vitl14_features_test_multi_layer.pkl" ^
  --labels CNV DME DRUSEN NORMAL ^
  --k 20 ^
  --enable-lda ^
  --voting rank ^
  --output predictions.json
```

If test file names contain ground-truth class labels, the script will also report accuracy.
