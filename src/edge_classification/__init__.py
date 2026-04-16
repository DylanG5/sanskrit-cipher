"""Edge classification training utilities."""

from .dataset import (
    EDGE_LABEL_KEYS,
    PIECE_TYPE_NAMES,
    EdgeClassificationDataset,
    EdgeClassificationSample,
    build_edge_image_transform,
    classify_piece_type,
    load_edge_classification_samples,
    render_fragment_image,
    split_samples_by_collection,
    write_split_manifest,
)
from .model import EdgeMultiLabelClassifier, load_edge_classifier
from .training import (
    compute_classification_metrics,
    compute_edge_pos_weight,
    evaluate_model,
    predict_dataset,
    train_one_epoch,
    tune_thresholds,
)

__all__ = [
    "EDGE_LABEL_KEYS",
    "PIECE_TYPE_NAMES",
    "EdgeClassificationDataset",
    "EdgeClassificationSample",
    "EdgeMultiLabelClassifier",
    "build_edge_image_transform",
    "classify_piece_type",
    "compute_classification_metrics",
    "compute_edge_pos_weight",
    "evaluate_model",
    "load_edge_classifier",
    "load_edge_classification_samples",
    "predict_dataset",
    "render_fragment_image",
    "split_samples_by_collection",
    "train_one_epoch",
    "tune_thresholds",
    "write_split_manifest",
]
