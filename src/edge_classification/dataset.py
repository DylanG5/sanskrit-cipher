"""Dataset loading and preprocessing for edge classification."""

from __future__ import annotations

import json
import math
import random
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


EDGE_LABEL_KEYS = [
    "has_top_edge",
    "has_bottom_edge",
    "has_left_edge",
    "has_right_edge",
]

PIECE_TYPE_NAMES = ["interior", "edge", "corner"]
PIECE_TYPE_TO_INDEX = {name: index for index, name in enumerate(PIECE_TYPE_NAMES)}


@dataclass(frozen=True)
class EdgeClassificationSample:
    relative_path: str
    absolute_path: Path
    collection: str
    labels: Dict[str, bool]
    piece_type: str
    segmentation_coords: str
    pixels_per_unit: Optional[float] = None
    scale_unit: Optional[str] = None

    def to_manifest_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["absolute_path"] = str(self.absolute_path)
        return payload


def classify_piece_type(labels: Dict[str, bool]) -> str:
    border_edges = {
        key.replace("has_", "").replace("_edge", "") + "_edge"
        for key, value in labels.items()
        if value
    }
    if not border_edges:
        return "interior"
    if len(border_edges) == 1:
        return "edge"

    adjacent_pairs = {
        frozenset(("top_edge", "left_edge")),
        frozenset(("top_edge", "right_edge")),
        frozenset(("bottom_edge", "left_edge")),
        frozenset(("bottom_edge", "right_edge")),
    }
    if any(pair.issubset(border_edges) for pair in adjacent_pairs):
        return "corner"
    return "edge"


def _normalize_relative_path(relative_path: str) -> str:
    return str(relative_path).replace("\\", "/")


def load_edge_classification_samples(
    annotations_path: Path,
    data_root: Path,
    fragments_db: Path,
    require_complete_labels: bool = True,
    require_scale: bool = False,
) -> Tuple[List[EdgeClassificationSample], Dict[str, int]]:
    payload = json.loads(annotations_path.read_text())
    annotations = payload.get("annotations", {})

    conn = sqlite3.connect(fragments_db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT image_path, segmentation_coords, pixels_per_unit, scale_unit, scale_detection_status
            FROM fragments
            """
        ).fetchall()
    finally:
        conn.close()

    db_rows = {
        _normalize_relative_path(str(row["image_path"])): row
        for row in rows
    }

    samples: List[EdgeClassificationSample] = []
    stats = {
        "reviewed_non_skipped": 0,
        "complete_labels": 0,
        "matched_db": 0,
        "with_segmentation": 0,
        "with_scale": 0,
        "usable_samples": 0,
    }

    for relative_path, record in annotations.items():
        normalized_path = _normalize_relative_path(relative_path)
        if normalized_path == "uploads/example.png":
            continue
        if not record.get("reviewed", False) or record.get("skipped", False):
            continue

        stats["reviewed_non_skipped"] += 1

        labels = {key: record.get(key) for key in EDGE_LABEL_KEYS}
        if require_complete_labels and any(value is None for value in labels.values()):
            continue
        if all(value is not None for value in labels.values()):
            stats["complete_labels"] += 1

        row = db_rows.get(normalized_path)
        if row is None:
            continue
        stats["matched_db"] += 1

        segmentation_coords = row["segmentation_coords"]
        if not segmentation_coords:
            continue
        stats["with_segmentation"] += 1

        pixels_per_unit = row["pixels_per_unit"]
        scale_status = row["scale_detection_status"]
        if require_scale and not (pixels_per_unit is not None and scale_status == "success"):
            continue
        if pixels_per_unit is not None and scale_status == "success":
            stats["with_scale"] += 1

        absolute_path = (data_root / normalized_path).resolve()
        if not absolute_path.exists():
            continue

        final_labels = {key: bool(value) for key, value in labels.items() if value is not None}
        if len(final_labels) != len(EDGE_LABEL_KEYS):
            continue

        piece_type = classify_piece_type(final_labels)
        samples.append(
            EdgeClassificationSample(
                relative_path=normalized_path,
                absolute_path=absolute_path,
                collection=normalized_path.split("/", 1)[0],
                labels=final_labels,
                piece_type=piece_type,
                segmentation_coords=str(segmentation_coords),
                pixels_per_unit=float(pixels_per_unit) if pixels_per_unit is not None else None,
                scale_unit=row["scale_unit"],
            )
        )

    samples.sort(key=lambda sample: sample.relative_path)
    stats["usable_samples"] = len(samples)
    return samples, stats


def split_samples_by_collection(
    samples: Sequence[EdgeClassificationSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[EdgeClassificationSample]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1.0")

    grouped: Dict[str, List[EdgeClassificationSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.collection, []).append(sample)

    collections = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(collections)

    split_names = ["train", "val", "test"]
    target_counts = {
        "train": len(samples) * train_ratio,
        "val": len(samples) * val_ratio,
        "test": len(samples) * test_ratio,
    }
    target_positive_counts = {
        split_name: {
            key: sum(int(sample.labels[key]) for sample in samples) * ratio
            for key in EDGE_LABEL_KEYS
        }
        for split_name, ratio in (
            ("train", train_ratio),
            ("val", val_ratio),
            ("test", test_ratio),
        )
    }
    target_collection_counts = {
        "train": len(collections) * train_ratio,
        "val": len(collections) * val_ratio,
        "test": len(collections) * test_ratio,
    }
    split_counts = {name: 0 for name in split_names}
    split_collection_counts = {name: 0 for name in split_names}
    split_positive_counts = {
        name: {key: 0 for key in EDGE_LABEL_KEYS}
        for name in split_names
    }
    split_samples = {name: [] for name in split_names}

    for _, group_samples in collections:
        group_size = len(group_samples)
        group_positive_counts = {
            key: sum(int(sample.labels[key]) for sample in group_samples)
            for key in EDGE_LABEL_KEYS
        }

        def score(split_name: str) -> Tuple[float, float, float, float, float, float]:
            target = max(target_counts[split_name], 1.0)
            target_collections = max(target_collection_counts[split_name], 1.0)
            projected = split_counts[split_name] + group_size
            projected_collections = split_collection_counts[split_name] + 1
            overflow = max(0.0, projected - target_counts[split_name])
            sample_fill_ratio = projected / target
            collection_fill_ratio = projected_collections / target_collections
            positive_balance_error = 0.0
            for key in EDGE_LABEL_KEYS:
                target_positive = max(target_positive_counts[split_name][key], 1.0)
                projected_positive = split_positive_counts[split_name][key] + group_positive_counts[key]
                positive_balance_error += abs((projected_positive / target_positive) - 1.0)
            return (
                overflow / target,
                positive_balance_error / len(EDGE_LABEL_KEYS),
                abs(collection_fill_ratio - 1.0),
                abs(sample_fill_ratio - 1.0),
                split_collection_counts[split_name],
                split_counts[split_name],
            )

        chosen_split = min(split_names, key=score)
        split_samples[chosen_split].extend(group_samples)
        split_counts[chosen_split] += group_size
        split_collection_counts[chosen_split] += 1
        for key in EDGE_LABEL_KEYS:
            split_positive_counts[chosen_split][key] += group_positive_counts[key]

    for name in split_names:
        split_samples[name].sort(key=lambda sample: sample.relative_path)

    return split_samples


def write_split_manifest(path: Path, samples: Iterable[EdgeClassificationSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [sample.to_manifest_dict() for sample in samples]
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_segmentation_contours(segmentation_coords: str) -> List[np.ndarray]:
    payload = json.loads(segmentation_coords)
    contours = payload.get("contours", [])
    parsed: List[np.ndarray] = []

    for contour in contours:
        array = np.asarray(contour, dtype=np.int32)
        if array.ndim == 3 and array.shape[1:] == (1, 2):
            array = array.reshape(-1, 2)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
            continue
        parsed.append(array)

    if not parsed:
        raise ValueError("No valid contours in segmentation data")

    return parsed


def render_sample_image(
    sample: EdgeClassificationSample,
    crop_pad: int = 6,
    target_pixels_per_unit: Optional[float] = 100.0,
    max_scaled_longest_side: int = 1024,
    input_mode: str = "masked_rgb",
) -> np.ndarray:
    image = cv2.imread(str(sample.absolute_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {sample.absolute_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    contours = parse_segmentation_contours(sample.segmentation_coords)
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [contour.astype(np.int32) for contour in contours], 255)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Segmentation mask is empty")

    pad = max(0, int(crop_pad))
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1] - 1, int(xs.max()) + pad)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0] - 1, int(ys.max()) + pad)

    crop_rgb = rgb[y0:y1 + 1, x0:x1 + 1].copy()
    crop_mask = mask[y0:y1 + 1, x0:x1 + 1].copy()

    if (
        target_pixels_per_unit is not None
        and sample.pixels_per_unit is not None
        and sample.pixels_per_unit > 0
    ):
        crop_rgb, crop_mask = _scale_crop(
            crop_rgb,
            crop_mask,
            scale_factor=float(target_pixels_per_unit) / float(sample.pixels_per_unit),
            max_scaled_longest_side=max_scaled_longest_side,
        )

    if input_mode == "masked_rgb":
        output = np.full_like(crop_rgb, 255)
        output[crop_mask > 0] = crop_rgb[crop_mask > 0]
        return output
    if input_mode == "rgb":
        return crop_rgb
    if input_mode == "mask_rgb":
        return np.repeat(crop_mask[:, :, None], 3, axis=2)
    if input_mode == "masked_gray_rgb":
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        output = np.repeat(gray[:, :, None], 3, axis=2)
        output[crop_mask == 0] = 255
        return output

    raise ValueError(f"Unsupported input_mode: {input_mode}")


def _scale_crop(
    crop_rgb: np.ndarray,
    crop_mask: np.ndarray,
    scale_factor: float,
    max_scaled_longest_side: int,
) -> Tuple[np.ndarray, np.ndarray]:
    scale_factor = max(0.05, float(scale_factor))
    out_h = max(3, int(round(crop_rgb.shape[0] * scale_factor)))
    out_w = max(3, int(round(crop_rgb.shape[1] * scale_factor)))

    longest = max(out_h, out_w)
    if max_scaled_longest_side > 0 and longest > max_scaled_longest_side:
        cap = max_scaled_longest_side / float(longest)
        out_h = max(3, int(round(out_h * cap)))
        out_w = max(3, int(round(out_w * cap)))

    scaled_rgb = cv2.resize(crop_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(crop_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    scaled_mask = ((scaled_mask > 0).astype(np.uint8)) * 255
    return scaled_rgb, scaled_mask


class EdgeClassificationDataset(Dataset):
    """PyTorch dataset for edge classification."""

    def __init__(
        self,
        samples: Sequence[EdgeClassificationSample],
        img_size: int = 224,
        augment: bool = False,
        target_pixels_per_unit: Optional[float] = 100.0,
        input_mode: str = "masked_rgb",
        crop_pad: int = 6,
        max_scaled_longest_side: int = 1024,
    ) -> None:
        self.samples = list(samples)
        self.img_size = int(img_size)
        self.augment = bool(augment)
        self.target_pixels_per_unit = target_pixels_per_unit
        self.input_mode = input_mode
        self.crop_pad = int(crop_pad)
        self.max_scaled_longest_side = int(max_scaled_longest_side)
        self.transform = self._build_transforms()

    def _build_transforms(self) -> T.Compose:
        transforms: List[object] = [T.ToPILImage()]

        if self.augment:
            transforms.extend(
                [
                    T.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.1),
                    T.RandomAffine(
                        degrees=0,
                        translate=(0.04, 0.04),
                        scale=(0.96, 1.04),
                        fill=255,
                    ),
                ]
            )

        transforms.extend(
            [
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        return T.Compose(transforms)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        image = render_sample_image(
            sample,
            crop_pad=self.crop_pad,
            target_pixels_per_unit=self.target_pixels_per_unit,
            max_scaled_longest_side=self.max_scaled_longest_side,
            input_mode=self.input_mode,
        )
        tensor = self.transform(image)
        edge_targets = torch.tensor(
            [float(sample.labels[key]) for key in EDGE_LABEL_KEYS],
            dtype=torch.float32,
        )
        piece_target = torch.tensor(
            PIECE_TYPE_TO_INDEX[sample.piece_type],
            dtype=torch.long,
        )
        return {
            "image": tensor,
            "edge_targets": edge_targets,
            "piece_target": piece_target,
            "relative_path": sample.relative_path,
            "collection": sample.collection,
        }
