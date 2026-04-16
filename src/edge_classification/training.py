"""Training utilities for edge classification."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .dataset import EDGE_LABEL_KEYS, EdgeClassificationSample


def compute_edge_pos_weight(samples: Sequence[EdgeClassificationSample]) -> torch.Tensor:
    labels = np.asarray(
        [[int(sample.labels[key]) for key in EDGE_LABEL_KEYS] for sample in samples],
        dtype=np.float32,
    )
    positives = labels.sum(axis=0)
    negatives = len(labels) - positives
    pos_weight = np.where(positives > 0, negatives / np.maximum(positives, 1.0), 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, object]],
    optimizer: torch.optim.Optimizer,
    edge_criterion: nn.Module,
    device: torch.device,
    piece_criterion: Optional[nn.Module] = None,
    piece_loss_weight: float = 0.2,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_edge_loss = 0.0
    total_piece_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        edge_targets = batch["edge_targets"].to(device)
        piece_targets = batch["piece_target"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        edge_loss = edge_criterion(outputs["edge_logits"], edge_targets)
        loss = edge_loss

        piece_loss_value = torch.tensor(0.0, device=device)
        if piece_criterion is not None and "piece_logits" in outputs:
            piece_loss_value = piece_criterion(outputs["piece_logits"], piece_targets)
            loss = loss + (piece_loss_weight * piece_loss_value)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = images.size(0)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_edge_loss += float(edge_loss.item()) * batch_size
        total_piece_loss += float(piece_loss_value.item()) * batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "edge_loss": total_edge_loss / max(total_examples, 1),
        "piece_loss": total_piece_loss / max(total_examples, 1),
    }


def predict_dataset(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, object]],
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    edge_probs: List[np.ndarray] = []
    edge_targets: List[np.ndarray] = []
    piece_logits: List[np.ndarray] = []
    piece_targets: List[np.ndarray] = []
    relative_paths: List[str] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            outputs = model(images)

            edge_prob = torch.sigmoid(outputs["edge_logits"]).cpu().numpy()
            edge_target = batch["edge_targets"].cpu().numpy()
            edge_probs.append(edge_prob)
            edge_targets.append(edge_target)

            if "piece_logits" in outputs:
                piece_logits.append(outputs["piece_logits"].cpu().numpy())
            piece_targets.append(batch["piece_target"].cpu().numpy())
            relative_paths.extend(batch["relative_path"])

    return {
        "edge_probs": np.concatenate(edge_probs, axis=0) if edge_probs else np.empty((0, 4)),
        "edge_targets": np.concatenate(edge_targets, axis=0) if edge_targets else np.empty((0, 4)),
        "piece_logits": np.concatenate(piece_logits, axis=0) if piece_logits else None,
        "piece_targets": np.concatenate(piece_targets, axis=0) if piece_targets else np.empty((0,)),
        "relative_paths": relative_paths,
    }


def tune_thresholds(
    edge_probs: np.ndarray,
    edge_targets: np.ndarray,
    candidate_thresholds: Optional[Sequence[float]] = None,
) -> List[float]:
    if edge_probs.size == 0:
        return [0.5] * len(EDGE_LABEL_KEYS)

    if candidate_thresholds is None:
        candidate_thresholds = [round(value, 2) for value in np.arange(0.2, 0.81, 0.05)]

    tuned: List[float] = []
    for label_index in range(edge_probs.shape[1]):
        best_threshold = 0.5
        best_score = -1.0
        probs = edge_probs[:, label_index]
        targets = edge_targets[:, label_index].astype(np.int32)

        for threshold in candidate_thresholds:
            preds = (probs >= threshold).astype(np.int32)
            tp = int(np.sum((preds == 1) & (targets == 1)))
            fp = int(np.sum((preds == 1) & (targets == 0)))
            fn = int(np.sum((preds == 0) & (targets == 1)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
            if f1 > best_score:
                best_score = f1
                best_threshold = float(threshold)

        tuned.append(best_threshold)

    return tuned


def compute_classification_metrics(
    edge_probs: np.ndarray,
    edge_targets: np.ndarray,
    thresholds: Optional[Sequence[float]] = None,
    piece_logits: Optional[np.ndarray] = None,
    piece_targets: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    thresholds = np.asarray(thresholds if thresholds is not None else [0.5] * len(EDGE_LABEL_KEYS), dtype=np.float32)
    preds = (edge_probs >= thresholds.reshape(1, -1)).astype(np.int32)
    targets = edge_targets.astype(np.int32)

    side_metrics: Dict[str, Dict[str, float]] = {}
    for index, key in enumerate(EDGE_LABEL_KEYS):
        tp = int(np.sum((preds[:, index] == 1) & (targets[:, index] == 1)))
        tn = int(np.sum((preds[:, index] == 0) & (targets[:, index] == 0)))
        fp = int(np.sum((preds[:, index] == 1) & (targets[:, index] == 0)))
        fn = int(np.sum((preds[:, index] == 0) & (targets[:, index] == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
        accuracy = (tp + tn) / max(len(preds), 1)
        side_metrics[key] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    exact_match = float(np.mean(np.all(preds == targets, axis=1))) if len(preds) else 0.0
    micro_accuracy = float(np.mean(preds == targets)) if preds.size else 0.0

    metrics: Dict[str, object] = {
        "thresholds": [float(value) for value in thresholds.tolist()],
        "exact_match_accuracy": exact_match,
        "micro_accuracy": micro_accuracy,
        "side_metrics": side_metrics,
    }

    if piece_logits is not None and piece_targets is not None and len(piece_targets):
        piece_preds = np.argmax(piece_logits, axis=1)
        metrics["piece_accuracy"] = float(np.mean(piece_preds == piece_targets))

    return metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable[Dict[str, object]],
    device: torch.device,
    thresholds: Optional[Sequence[float]] = None,
    tune_on_dataset: bool = False,
) -> Dict[str, object]:
    predictions = predict_dataset(model, dataloader, device)
    active_thresholds = list(thresholds) if thresholds is not None else None
    if tune_on_dataset:
        active_thresholds = tune_thresholds(
            predictions["edge_probs"],
            predictions["edge_targets"],
        )

    metrics = compute_classification_metrics(
        predictions["edge_probs"],
        predictions["edge_targets"],
        thresholds=active_thresholds,
        piece_logits=predictions["piece_logits"],
        piece_targets=predictions["piece_targets"],
    )
    metrics["predictions"] = predictions
    return metrics
