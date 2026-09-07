#!/usr/bin/env python3
"""
VisionQC model training pipeline.

Builds a fully local, reproducible training set from clean source images
bundled with scikit-image (no network access, no external dataset
download), synthesizes controlled degradations for every target class,
extracts the 16 engineered OpenCV/NumPy features for every sample, and
trains a RandomForestClassifier to predict the image-quality class.

Usage
-----
    python -m backend.training.train_model
    # or, from the backend/ directory:
    python training/train_model.py

Outputs (written to <repo_root>/models/):
    quality_model.joblib      - the trained sklearn Pipeline (scaler + RF)
    model_metadata.json       - feature names, classes, params, versions
    metrics.json              - accuracy/precision/recall/F1/AUC, per-class
    confusion_matrix.png      - confusion matrix heatmap on the held-out set
    feature_importance.png    - Random Forest feature importances
    samples/                  - a few representative images per class

Data-leakage safeguard
-----------------------
Every degraded sample is tagged with the id of the *source* image it was
derived from. The train/test split is performed on those source-image
groups (via sklearn's GroupShuffleSplit), so no two augmented versions of
the same original photo can ever appear on both sides of the split.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skimage.data as skdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

# Allow running this file directly (``python training/train_model.py``) as
# well as as a module (``python -m backend.training.train_model``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.image_features import FEATURE_NAMES, extract_features  # noqa: E402
from training.degradations import CLASSES, generate_variant  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
SAMPLES_DIR = MODELS_DIR / "samples"

RANDOM_SEED = 42
VARIANTS_PER_CLASS = 6          # degraded samples generated per source image per class
TEST_GROUP_FRACTION = 0.30      # ~30% of *source images* reserved for testing
MODEL_VERSION = "1.0.0"

# Source images bundled locally with scikit-image (no network fetch needed).
# A mix of grayscale and color, portrait/scene/texture content for diversity.
SOURCE_IMAGE_NAMES = [
    "astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry",
    "hubble_deep_field", "colorwheel", "cat", "camera", "coins",
    "retina", "page", "text", "clock", "brick", "grass",
    "gravel", "moon", "cell",
]
# NOTE: "checkerboard" was deliberately excluded. It is a perfectly
# periodic synthetic grid whose spatial period aliases with the 8x8 JPEG
# block grid used by the ``blockiness`` feature, producing pathological
# outlier values (~20x every other image) that do not reflect real-world
# photographs and measurably distorted class statistics during
# development. All remaining sources are natural photographs, scientific
# imagery, or textures without exact periodic structure.

# A range of resize targets (rather than one fixed size) so the "clean"
# class statistics span a realistic range of resolutions/scales instead
# of overfitting to one fixed working size.
RESIZE_TARGETS = [320, 384, 448, 512, 576]


def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    """Normalize a scikit-image sample array to an RGB uint8 image."""
    if img.dtype == bool:
        img = (img.astype(np.uint8)) * 255
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:  # RGBA (e.g. the skimage logo)
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def load_source_images() -> list[tuple[str, np.ndarray]]:
    """Load every bundled scikit-image sample used as a "clean" source."""
    sources = []
    for idx, name in enumerate(SOURCE_IMAGE_NAMES):
        try:
            raw = getattr(skdata, name)()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  [warn] could not load skimage sample '{name}': {exc}")
            continue
        img = _to_rgb_u8(raw)
        target = RESIZE_TARGETS[idx % len(RESIZE_TARGETS)]
        interp = cv2.INTER_AREA if max(img.shape[:2]) > target else cv2.INTER_CUBIC
        img = cv2.resize(img, (target, target), interpolation=interp)
        sources.append((name, img))
    return sources


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Generate the synthetic degraded dataset and extract features.

    Returns
    -------
    X : (n_samples, n_features) float64 array
    y : (n_samples,) str array of class labels
    groups : (n_samples,) str array of source-image ids (for grouped split)
    sample_records : list of dicts with metadata about each sample, used to
        save a handful of representative images to disk afterwards.
    """
    sources = load_source_images()
    print(f"Loaded {len(sources)} local source images: "
          f"{[n for n, _ in sources]}")

    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    group_rows: list[str] = []
    sample_records: list[dict] = []

    for src_idx, (src_name, src_img) in enumerate(sources):
        for cls_idx, cls in enumerate(CLASSES):
            for variant_idx in range(VARIANTS_PER_CLASS):
                seed = (RANDOM_SEED * 100_003
                        + src_idx * 1_009
                        + cls_idx * 101
                        + variant_idx)
                rng = np.random.default_rng(seed)
                degraded_rgb = generate_variant(cls, src_img, rng)
                degraded_bgr = cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR)
                feats = extract_features(degraded_bgr)

                X_rows.append(feats.to_vector())
                y_rows.append(cls)
                group_rows.append(src_name)
                sample_records.append({
                    "source": src_name,
                    "class": cls,
                    "variant": variant_idx,
                    "image_bgr": degraded_bgr,
                })

    X = np.vstack(X_rows)
    y = np.array(y_rows)
    groups = np.array(group_rows)
    print(f"Built dataset: {X.shape[0]} samples x {X.shape[1]} features, "
          f"{len(set(groups))} source groups, {len(set(y))} classes.")
    return X, y, groups, sample_records


def grouped_split(X, y, groups):
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_GROUP_FRACTION,
                                  random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    assert not overlap, f"Data leakage detected! Overlapping source groups: {overlap}"

    print(f"Grouped split -> train: {len(train_idx)} samples "
          f"({len(train_groups)} source images), "
          f"test: {len(test_idx)} samples ({len(test_groups)} source images).")
    print(f"  Train source images: {sorted(train_groups)}")
    print(f"  Test source images:  {sorted(test_groups)}")
    return train_idx, test_idx


def train_and_evaluate(X, y, groups):
    train_idx, test_idx = grouped_split(X, y, groups)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )),
    ])

    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_seconds = time.time() - t0
    print(f"Trained RandomForestClassifier in {train_seconds:.2f}s")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    classes_ = pipeline.named_steps["rf"].classes_

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # One-vs-rest ROC-AUC (macro). Requires >= 2 classes present in y_test.
    try:
        y_test_bin = label_binarize(y_test, classes=classes_)
        roc_auc_macro = roc_auc_score(y_test_bin, y_proba, average="macro",
                                       multi_class="ovr")
    except ValueError as exc:
        print(f"  [warn] could not compute ROC-AUC: {exc}")
        roc_auc_macro = None

    cm = confusion_matrix(y_test, y_pred, labels=classes_)

    metrics = {
        "model_version": MODEL_VERSION,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "random_seed": RANDOM_SEED,
        "n_train_samples": int(len(train_idx)),
        "n_test_samples": int(len(test_idx)),
        "n_train_source_images": int(len(set(groups[train_idx]))),
        "n_test_source_images": int(len(set(groups[test_idx]))),
        "training_seconds": round(train_seconds, 3),
        "classes": list(classes_),
        "accuracy": float(accuracy),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "roc_auc_macro_ovr": float(roc_auc_macro) if roc_auc_macro is not None else None,
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1_score": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in classes_ if cls in report
        },
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(classes_),
    }

    print("\n=== Evaluation summary (held-out, grouped split) ===")
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Precision (macro):   {precision_macro:.4f}")
    print(f"Recall (macro):      {recall_macro:.4f}")
    print(f"F1-score (macro):    {f1_macro:.4f}")
    if roc_auc_macro is not None:
        print(f"ROC-AUC (macro,ovr): {roc_auc_macro:.4f}")
    print("\nPer-class:")
    for cls in classes_:
        pc = metrics["per_class"].get(cls)
        if pc:
            print(f"  {cls:14s} precision={pc['precision']:.3f} "
                  f"recall={pc['recall']:.3f} f1={pc['f1_score']:.3f} "
                  f"support={pc['support']}")

    return pipeline, metrics, classes_, cm, test_idx


def save_confusion_matrix(cm, labels, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
    ax.set_title("VisionQC - Confusion Matrix (held-out test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix -> {out_path}")


def save_feature_importance(pipeline: Pipeline, out_path: Path):
    importances = pipeline.named_steps["rf"].feature_importances_
    order = np.argsort(importances)[::-1]
    names = [FEATURE_NAMES[i] for i in order]
    vals = importances[order]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(names)), vals[::-1], color="#3b82f6")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Random Forest feature importance (Gini)")
    ax.set_title("VisionQC - Feature Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance chart -> {out_path}")

    return {name: float(val) for name, val in zip(names, vals)}


def save_sample_gallery(sample_records: list[dict], test_source_names: set[str]):
    """Save one representative *test-set* image per class to models/samples/."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    saved_classes = set()
    for rec in sample_records:
        if rec["class"] in saved_classes:
            continue
        if rec["source"] not in test_source_names:
            continue
        out_path = SAMPLES_DIR / f"{rec['class']}.jpg"
        cv2.imwrite(str(out_path), rec["image_bgr"])
        saved_classes.add(rec["class"])
    print(f"Saved {len(saved_classes)} representative sample images -> {SAMPLES_DIR}")


def main():
    print("=" * 70)
    print("VisionQC - local reproducible model training")
    print("=" * 70)
    np.random.seed(RANDOM_SEED)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups, sample_records = build_dataset()
    pipeline, metrics, classes_, cm, test_idx = train_and_evaluate(X, y, groups)

    model_path = MODELS_DIR / "quality_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved trained model -> {model_path}")

    metrics_path = MODELS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics -> {metrics_path}")

    save_confusion_matrix(cm, classes_, MODELS_DIR / "confusion_matrix.png")
    importances = save_feature_importance(pipeline, MODELS_DIR / "feature_importance.png")

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "sklearn.ensemble.RandomForestClassifier",
        "pipeline_steps": ["StandardScaler", "RandomForestClassifier"],
        "feature_names": FEATURE_NAMES,
        "classes": list(classes_),
        "trained_at": metrics["trained_at"],
        "random_seed": RANDOM_SEED,
        "hyperparameters": {
            "n_estimators": 400,
            "max_depth": 18,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
        },
        "training_data": {
            "source": "scikit-image bundled sample images "
                       "(skimage.data), synthetically degraded",
            "n_source_images": len(set(groups)),
            "n_total_samples": int(X.shape[0]),
            "variants_per_class_per_source": VARIANTS_PER_CLASS,
            "split_strategy": "GroupShuffleSplit by source image "
                               f"(~{int((1 - TEST_GROUP_FRACTION) * 100)}/"
                               f"{int(TEST_GROUP_FRACTION * 100)} train/test)",
        },
        "feature_importance": importances,
    }
    metadata_path = MODELS_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved model metadata -> {metadata_path}")

    test_source_names = set(groups[test_idx])
    save_sample_gallery(sample_records, test_source_names)

    print("\nDone. Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
