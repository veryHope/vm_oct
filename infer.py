import argparse
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import normalize


DEFAULT_LABELS = ["CNV", "DME", "DRUSEN", "NORMAL"]


def ensure_2d_features(features):
    features = np.asarray(features)
    if features.ndim == 3 and features.shape[1] == 1:
        features = np.squeeze(features, axis=1)
    return features


def extract_label_from_filename(filename, labels):
    upper_name = os.path.basename(str(filename)).upper()
    for label in labels:
        if label.upper() in upper_name:
            return label
    return "UNKNOWN"


def load_features(paths):
    all_names = []
    all_features = []

    for path in paths:
        with open(path, "rb") as file:
            names, features = pickle.load(file)

        if isinstance(features, np.ndarray):
            feature_rows = list(features)
        else:
            feature_rows = list(features)

        all_names.extend(names)
        all_features.extend(feature_rows)

    if not all_names:
        raise ValueError("No features were loaded.")

    return all_names, ensure_2d_features(np.vstack(all_features))


def calculate_voting_weights(similarities, method="rank", tau=0.07, alpha=2.0):
    sims = np.asarray(similarities, dtype=np.float64)

    if method == "plurality":
        return np.ones_like(sims, dtype=np.float64)

    if method == "distance":
        distances = 1.0 - np.clip(sims, -1.0, 1.0)
        return np.exp(-distances)

    if method == "softmax":
        scaled = sims / tau
        exp_sims = np.exp(scaled - np.max(scaled))
        return exp_sims / np.sum(exp_sims)

    if method == "rank":
        ranks = np.arange(1, len(sims) + 1, dtype=np.float64)
        return 1.0 / (alpha + ranks)

    raise ValueError(f"Unsupported voting method: {method}")


def apply_feature_engineering(train_features, test_features, train_labels, enable_lda):
    train_features = normalize(train_features, norm="l2")
    test_features = normalize(test_features, norm="l2")

    if not enable_lda:
        return train_features, test_features

    unique_labels = sorted(set(train_labels))
    if "UNKNOWN" in unique_labels:
        raise ValueError("LDA requires class labels to be parsable from training file names.")
    if len(unique_labels) < 2:
        raise ValueError("LDA requires at least two classes.")

    lda = LinearDiscriminantAnalysis()
    train_features = lda.fit_transform(train_features, train_labels)
    test_features = lda.transform(test_features)

    train_features = normalize(train_features, norm="l2")
    test_features = normalize(test_features, norm="l2")
    return train_features, test_features


def build_class_weights(train_labels):
    counts = Counter(train_labels)
    total = len(train_labels)
    class_count = len(counts)
    return {
        label: total / (class_count * count)
        for label, count in counts.items()
    }


def find_topk_neighbors(train_features, test_features, k):
    similarities = np.matmul(test_features, train_features.T)
    k = max(1, min(int(k), train_features.shape[0]))

    if k == train_features.shape[0]:
        topk_indices = np.argsort(-similarities, axis=1)
    else:
        partition = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
        row_ids = np.arange(similarities.shape[0])[:, None]
        part_sims = similarities[row_ids, partition]
        order = np.argsort(-part_sims, axis=1)
        topk_indices = partition[row_ids, order]

    row_ids = np.arange(similarities.shape[0])[:, None]
    topk_similarities = similarities[row_ids, topk_indices]
    return topk_indices, topk_similarities


def predict_labels(
    train_names,
    test_names,
    train_labels,
    topk_indices,
    topk_similarities,
    class_weights,
    voting_method,
):
    predictions = []

    for sample_index, test_name in enumerate(test_names):
        neighbor_indices = topk_indices[sample_index]
        neighbor_names = [train_names[index] for index in neighbor_indices]
        neighbor_labels = [train_labels[index] for index in neighbor_indices]
        neighbor_sims = topk_similarities[sample_index].astype(float).tolist()

        voting_weights = calculate_voting_weights(neighbor_sims, method=voting_method)
        score_map = defaultdict(float)
        for label, weight in zip(neighbor_labels, voting_weights):
            score_map[label] += class_weights.get(label, 1.0) * float(weight)

        predicted_label = max(score_map.items(), key=lambda item: item[1])[0]
        predictions.append(
            {
                "test_name": test_name,
                "predicted_label": predicted_label,
                "neighbors": neighbor_names,
                "neighbor_labels": neighbor_labels,
                "similarities": neighbor_sims,
                "scores": {label: round(score, 6) for label, score in score_map.items()},
            }
        )

    return predictions


def run_inference(
    train_paths,
    test_paths,
    labels=None,
    k=20,
    enable_lda=False,
    voting_method="rank",
):
    labels = labels or DEFAULT_LABELS

    train_names, train_features = load_features(train_paths)
    test_names, test_features = load_features(test_paths)

    train_labels = [extract_label_from_filename(name, labels) for name in train_names]
    true_test_labels = [extract_label_from_filename(name, labels) for name in test_names]

    train_features, test_features = apply_feature_engineering(
        train_features=train_features,
        test_features=test_features,
        train_labels=train_labels,
        enable_lda=enable_lda,
    )

    class_weights = build_class_weights(train_labels)
    topk_indices, topk_similarities = find_topk_neighbors(train_features, test_features, k=k)
    predictions = predict_labels(
        train_names=train_names,
        test_names=test_names,
        train_labels=train_labels,
        topk_indices=topk_indices,
        topk_similarities=topk_similarities,
        class_weights=class_weights,
        voting_method=voting_method,
    )

    predicted_labels = [item["predicted_label"] for item in predictions]
    has_ground_truth = all(label != "UNKNOWN" for label in true_test_labels)

    result = {
        "train_sample_count": len(train_names),
        "test_sample_count": len(test_names),
        "k": max(1, min(int(k), len(train_names))),
        "enable_lda": bool(enable_lda),
        "voting_method": voting_method,
        "labels": labels,
        "predictions": predictions,
    }

    if has_ground_truth:
        accuracy = float(np.mean(np.asarray(true_test_labels) == np.asarray(predicted_labels)))
        result["accuracy"] = accuracy
        result["true_labels"] = true_test_labels

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="KNN inference script for pre-extracted feature vectors.")
    parser.add_argument("--train", nargs="+", required=True, help="Path to one or more training feature pkl files.")
    parser.add_argument("--test", nargs="+", required=True, help="Path to one or more test feature pkl files.")
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS, help="Label list used to parse class names from file names.")
    parser.add_argument("--k", type=int, default=20, help="Number of nearest neighbors for KNN.")
    parser.add_argument("--enable-lda", action="store_true", help="Enable LDA before retrieval.")
    parser.add_argument(
        "--voting",
        choices=["plurality", "distance", "softmax", "rank"],
        default="rank",
        help="Voting strategy for KNN aggregation.",
    )
    parser.add_argument("--output", default="predictions.json", help="Output path for inference results.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_inference(
        train_paths=args.train,
        test_paths=args.test,
        labels=args.labels,
        k=args.k,
        enable_lda=args.enable_lda,
        voting_method=args.voting,
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"Inference completed. Results saved to: {os.path.abspath(args.output)}")
    if "accuracy" in result:
        print(f"Accuracy: {result['accuracy']:.4f}")


if __name__ == "__main__":
    main()
