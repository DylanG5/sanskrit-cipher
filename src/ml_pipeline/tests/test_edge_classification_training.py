"""Tests for the edge classification training utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from edge_classification.dataset import (
    EDGE_LABEL_KEYS,
    EdgeClassificationDataset,
    EdgeClassificationSample,
    classify_piece_type,
    split_samples_by_collection,
)
from edge_classification.training import compute_classification_metrics, tune_thresholds


def _make_sample(tmp_path: Path, collection: str, name: str, labels: dict[str, bool]) -> EdgeClassificationSample:
    image_path = tmp_path / collection / name
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((64, 64, 3), 240, dtype=np.uint8)
    image[12:52, 10:54] = np.array([180, 120, 90], dtype=np.uint8)
    Image.fromarray(image).save(image_path)

    segmentation_coords = json.dumps(
        {
            "contours": [
                [[10, 12], [54, 12], [54, 52], [10, 52]],
            ]
        }
    )
    return EdgeClassificationSample(
        relative_path=f"{collection}/{name}",
        absolute_path=image_path,
        collection=collection,
        labels=labels,
        piece_type=classify_piece_type(labels),
        segmentation_coords=segmentation_coords,
        pixels_per_unit=120.0,
        scale_unit="cm",
    )


def test_classify_piece_type_variants():
    assert classify_piece_type({key: False for key in EDGE_LABEL_KEYS}) == "interior"
    assert classify_piece_type(
        {
            "has_top_edge": True,
            "has_bottom_edge": False,
            "has_left_edge": False,
            "has_right_edge": False,
        }
    ) == "edge"
    assert classify_piece_type(
        {
            "has_top_edge": True,
            "has_bottom_edge": False,
            "has_left_edge": True,
            "has_right_edge": False,
        }
    ) == "corner"
    assert classify_piece_type(
        {
            "has_top_edge": True,
            "has_bottom_edge": True,
            "has_left_edge": False,
            "has_right_edge": False,
        }
    ) == "edge"


def test_split_samples_by_collection_keeps_collections_disjoint(tmp_path: Path):
    samples = [
        _make_sample(tmp_path, "A", "a1.jpg", {key: False for key in EDGE_LABEL_KEYS}),
        _make_sample(tmp_path, "A", "a2.jpg", {key: False for key in EDGE_LABEL_KEYS}),
        _make_sample(tmp_path, "B", "b1.jpg", {key: True for key in EDGE_LABEL_KEYS}),
        _make_sample(tmp_path, "C", "c1.jpg", {key: False for key in EDGE_LABEL_KEYS}),
        _make_sample(tmp_path, "D", "d1.jpg", {key: False for key in EDGE_LABEL_KEYS}),
    ]
    splits = split_samples_by_collection(samples, seed=7)
    all_paths = [sample.relative_path for split in splits.values() for sample in split]

    assert sorted(all_paths) == sorted(sample.relative_path for sample in samples)

    seen_collections = {}
    for split_name, split_samples in splits.items():
        for sample in split_samples:
            if sample.collection in seen_collections:
                assert seen_collections[sample.collection] == split_name
            else:
                seen_collections[sample.collection] = split_name


def test_dataset_returns_tensor_and_targets(tmp_path: Path):
    labels = {
        "has_top_edge": True,
        "has_bottom_edge": False,
        "has_left_edge": True,
        "has_right_edge": False,
    }
    sample = _make_sample(tmp_path, "BLL10", "frag.jpg", labels)
    dataset = EdgeClassificationDataset([sample], img_size=96, augment=False)
    item = dataset[0]

    assert tuple(item["image"].shape) == (3, 96, 96)
    assert item["edge_targets"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert int(item["piece_target"].item()) == 2


def test_threshold_tuning_and_metrics():
    edge_probs = np.array(
        [
            [0.9, 0.2, 0.8, 0.1],
            [0.3, 0.7, 0.2, 0.9],
            [0.8, 0.8, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    edge_targets = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ],
        dtype=np.float32,
    )

    thresholds = tune_thresholds(edge_probs, edge_targets, candidate_thresholds=[0.4, 0.5, 0.6])
    metrics = compute_classification_metrics(edge_probs, edge_targets, thresholds=thresholds)

    assert len(thresholds) == 4
    assert metrics["exact_match_accuracy"] == 1.0
    assert metrics["side_metrics"]["has_top_edge"]["accuracy"] == 1.0
