"""Train a multi-label edge classifier from manual fragment annotations."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


SCRIPT_ROOT = Path(__file__).resolve()
SRC_ROOT = SCRIPT_ROOT.parents[2]
REPO_ROOT = SCRIPT_ROOT.parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_classification import (  # noqa: E402
    EDGE_LABEL_KEYS,
    PIECE_TYPE_NAMES,
    EdgeClassificationDataset,
    EdgeMultiLabelClassifier,
    compute_edge_pos_weight,
    evaluate_model,
    load_edge_classification_samples,
    split_samples_by_collection,
    train_one_epoch,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-label fragment edge classifier.")
    parser.add_argument("--annotations", type=Path, default=None, help="Path to edge-annotations.json")
    parser.add_argument("--data-root", type=Path, default=None, help="Path to fragment image root")
    parser.add_argument("--fragments-db", type=Path, default=None, help="Path to fragments.db")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SRC_ROOT / "models" / "edge-classification",
        help="Directory for weights, manifests, and training reports",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional stable run directory name under --output-dir",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--piece-loss-weight", type=float, default=0.2)
    parser.add_argument("--freeze-backbone-fraction", type=float, default=0.7)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--target-pixels-per-unit", type=float, default=100.0)
    parser.add_argument("--crop-pad", type=int, default=6)
    parser.add_argument("--max-scaled-longest-side", type=int, default=1024)
    parser.add_argument(
        "--input-mode",
        choices=["masked_rgb", "rgb", "mask_rgb", "masked_gray_rgb"],
        default="masked_rgb",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-piece-head", action="store_true")
    parser.add_argument("--require-scale", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_supported_images(data_root: Path) -> int:
    count = 0
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}:
            count += 1
    return count


def count_reviewed_annotations(annotations_path: Path) -> int:
    if not annotations_path.exists():
        return -1
    payload = json.loads(annotations_path.read_text())
    annotations = payload.get("annotations", {})
    return sum(1 for record in annotations.values() if record.get("reviewed") and not record.get("skipped"))


def resolve_default_inputs(
    annotations: Optional[Path],
    data_root: Optional[Path],
    fragments_db: Optional[Path],
) -> Tuple[Path, Path, Path]:
    candidate_repo_roots = [REPO_ROOT]
    repo_parent = REPO_ROOT.parent
    if repo_parent.exists():
        for entry in repo_parent.iterdir():
            if entry.is_dir():
                candidate_repo_roots.append(entry)

    def candidate_paths(repo_root: Path) -> Tuple[Path, Path, Path]:
        data_candidate = repo_root / "src" / "web" / "web-canvas" / "data"
        ann_candidate = data_candidate / "edge-annotations.json"
        db_candidate = repo_root / "src" / "web" / "web-canvas" / "electron" / "resources" / "database" / "fragments.db"
        return ann_candidate, data_candidate, db_candidate

    if annotations is None:
        best_annotations = None
        best_score = -1
        for repo_root in candidate_repo_roots:
            ann_candidate, _, _ = candidate_paths(repo_root)
            score = count_reviewed_annotations(ann_candidate)
            if score > best_score:
                best_score = score
                best_annotations = ann_candidate
        annotations = best_annotations

    if data_root is None:
        best_data_root = None
        best_score = -1
        for repo_root in candidate_repo_roots:
            _, data_candidate, _ = candidate_paths(repo_root)
            if not data_candidate.exists():
                continue
            score = count_supported_images(data_candidate)
            if score > best_score:
                best_score = score
                best_data_root = data_candidate
        data_root = best_data_root

    if fragments_db is None:
        for repo_root in candidate_repo_roots:
            _, _, db_candidate = candidate_paths(repo_root)
            if db_candidate.exists():
                fragments_db = db_candidate
                if data_root is not None and repo_root in data_root.parents:
                    break

    if annotations is None or not annotations.exists():
        raise FileNotFoundError("Could not resolve an annotations file. Pass --annotations explicitly.")
    if data_root is None or not data_root.exists():
        raise FileNotFoundError("Could not resolve a populated image root. Pass --data-root explicitly.")
    if fragments_db is None or not fragments_db.exists():
        raise FileNotFoundError("Could not resolve fragments.db. Pass --fragments-db explicitly.")

    return annotations.resolve(), data_root.resolve(), fragments_db.resolve()


def build_dataloaders(args: argparse.Namespace, split_samples: Dict[str, List]) -> Dict[str, DataLoader]:
    loaders: Dict[str, DataLoader] = {}
    for split_name, samples in split_samples.items():
        dataset = EdgeClassificationDataset(
            samples,
            img_size=args.img_size,
            augment=(split_name == "train"),
            target_pixels_per_unit=args.target_pixels_per_unit,
            input_mode=args.input_mode,
            crop_pad=args.crop_pad,
            max_scaled_longest_side=args.max_scaled_longest_side,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def summarize_split(split_name: str, samples: List) -> Dict[str, object]:
    side_positives = {
        key: sum(int(sample.labels[key]) for sample in samples)
        for key in EDGE_LABEL_KEYS
    }
    piece_counts = {
        name: sum(1 for sample in samples if sample.piece_type == name)
        for name in PIECE_TYPE_NAMES
    }
    return {
        "split": split_name,
        "count": len(samples),
        "collections": len({sample.collection for sample in samples}),
        "side_positives": side_positives,
        "piece_counts": piece_counts,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    annotations_path, data_root, fragments_db = resolve_default_inputs(
        args.annotations,
        args.data_root,
        args.fragments_db,
    )
    device = choose_device(args.device)

    samples, sample_stats = load_edge_classification_samples(
        annotations_path=annotations_path,
        data_root=data_root,
        fragments_db=fragments_db,
        require_complete_labels=True,
        require_scale=args.require_scale,
    )
    if args.limit_samples is not None:
        rng = random.Random(args.seed)
        samples = list(samples)
        rng.shuffle(samples)
        samples = samples[: args.limit_samples]
        samples.sort(key=lambda sample: sample.relative_path)

    if len(samples) < 12:
        raise ValueError(f"Not enough usable samples to train: {len(samples)}")

    split_samples = split_samples_by_collection(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    for split_name, split_list in split_samples.items():
        if not split_list:
            raise ValueError(
                f"Split '{split_name}' is empty after grouped splitting. "
                "Use more samples or adjust the split ratios."
            )

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir_name = args.run_name if args.run_name else f"run_{timestamp}"
    run_dir = args.output_dir / run_dir_name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    manifest_dir = run_dir / "manifests"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_list in split_samples.items():
        write_split_manifest(manifest_dir / f"{split_name}.json", split_list)

    run_summary = {
        "annotations_path": str(annotations_path),
        "data_root": str(data_root),
        "fragments_db": str(fragments_db),
        "device": str(device),
        "sample_stats": sample_stats,
        "splits": {name: summarize_split(name, split_list) for name, split_list in split_samples.items()},
        "config": {
            "img_size": args.img_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "piece_loss_weight": args.piece_loss_weight,
            "freeze_backbone_fraction": args.freeze_backbone_fraction,
            "dropout_rate": args.dropout_rate,
            "target_pixels_per_unit": args.target_pixels_per_unit,
            "crop_pad": args.crop_pad,
            "max_scaled_longest_side": args.max_scaled_longest_side,
            "input_mode": args.input_mode,
            "require_scale": args.require_scale,
            "pretrained": not args.no_pretrained,
            "aux_piece_head": not args.no_piece_head,
            "seed": args.seed,
        },
    }

    if args.dry_run:
        (run_dir / "dry_run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
        print(json.dumps(run_summary, indent=2))
        print(f"Wrote manifests and dry-run summary to {run_dir}")
        return

    dataloaders = build_dataloaders(args, split_samples)

    model = EdgeMultiLabelClassifier(
        pretrained=not args.no_pretrained,
        aux_piece_head=not args.no_piece_head,
        dropout_rate=args.dropout_rate,
        freeze_backbone_fraction=args.freeze_backbone_fraction,
    ).to(device)

    edge_pos_weight = compute_edge_pos_weight(split_samples["train"]).to(device)
    edge_criterion = nn.BCEWithLogitsLoss(pos_weight=edge_pos_weight)
    piece_criterion = None if args.no_piece_head else nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_state = None
    best_metrics = None
    best_score = -1.0
    bad_epochs = 0
    history: List[Dict[str, object]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            optimizer=optimizer,
            edge_criterion=edge_criterion,
            piece_criterion=piece_criterion,
            piece_loss_weight=args.piece_loss_weight,
            device=device,
        )

        val_metrics = evaluate_model(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            thresholds=None,
            tune_on_dataset=True,
        )

        scheduler.step(float(val_metrics["exact_match_accuracy"]))
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": {
                "exact_match_accuracy": val_metrics["exact_match_accuracy"],
                "micro_accuracy": val_metrics["micro_accuracy"],
                "piece_accuracy": val_metrics.get("piece_accuracy"),
                "thresholds": val_metrics["thresholds"],
            },
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)

        score = float(val_metrics["exact_match_accuracy"])
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_exact={score:.4f} "
            f"val_micro={float(val_metrics['micro_accuracy']):.4f}"
        )

        if score > best_score:
            best_score = score
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = val_metrics
            torch.save(model.state_dict(), weights_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    torch.save(model.state_dict(), weights_dir / "last.pt")

    if best_state is None or best_metrics is None:
        raise RuntimeError("Training finished without a best checkpoint")

    model.load_state_dict(best_state)
    test_metrics = evaluate_model(
        model=model,
        dataloader=dataloaders["test"],
        device=device,
        thresholds=best_metrics["thresholds"],
        tune_on_dataset=False,
    )

    metadata = {
        **run_summary,
        "best_val_exact_match_accuracy": best_metrics["exact_match_accuracy"],
        "best_thresholds": best_metrics["thresholds"],
        "history": history,
        "test": {
            "exact_match_accuracy": test_metrics["exact_match_accuracy"],
            "micro_accuracy": test_metrics["micro_accuracy"],
            "piece_accuracy": test_metrics.get("piece_accuracy"),
            "side_metrics": test_metrics["side_metrics"],
        },
        "label_keys": EDGE_LABEL_KEYS,
        "piece_type_names": PIECE_TYPE_NAMES,
    }

    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["test"], indent=2))
    print(f"Saved training run to {run_dir}")


if __name__ == "__main__":
    main()
