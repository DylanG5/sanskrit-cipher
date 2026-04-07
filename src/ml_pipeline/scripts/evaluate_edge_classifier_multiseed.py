"""Run grouped multi-seed training/evaluation for the edge classifier."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_ROOT = Path(__file__).resolve()
SRC_ROOT = SCRIPT_ROOT.parents[2]
REPO_ROOT = SCRIPT_ROOT.parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from edge_classification import (  # noqa: E402
    EDGE_LABEL_KEYS,
    EdgeClassificationDataset,
    EdgeClassificationSample,
    evaluate_model,
    load_edge_classifier,
)
from ml_pipeline.processors.edge_detection_processor import EdgeExtractor  # noqa: E402


HEURISTIC_METHODS = ("hull_segments", "oriented_runs", "oriented_runs_scaled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped multi-seed benchmark for the edge classifier.")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seed list")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SRC_ROOT / "models" / "edge-classification-multiseed",
        help="Directory that will contain one training run per seed",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=REPO_ROOT / "reports" / "edge_classifier_multiseed_eval.json",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=REPO_ROOT / "reports" / "edge_classifier_multiseed_eval.csv",
    )
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--fragments-db", type=Path, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--piece-loss-weight", type=float, default=0.2)
    parser.add_argument("--freeze-backbone-fraction", type=float, default=0.7)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=0)
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
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-piece-head", action="store_true")
    parser.add_argument("--require-scale", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing run dirs when present")
    return parser.parse_args()


def parse_seed_list(raw: str) -> List[int]:
    seeds = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        seeds.append(int(value))
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def run_training_for_seed(args: argparse.Namespace, seed: int) -> Path:
    run_name = f"seed_{seed}"
    run_dir = args.output_dir / run_name
    if run_dir.exists() and args.skip_existing:
        return run_dir

    cmd = [
        "python3",
        "-u",
        str(SCRIPT_ROOT.parent / "train_edge_classifier.py"),
        "--output-dir",
        str(args.output_dir),
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--img-size",
        str(args.img_size),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--piece-loss-weight",
        str(args.piece_loss_weight),
        "--freeze-backbone-fraction",
        str(args.freeze_backbone_fraction),
        "--dropout-rate",
        str(args.dropout_rate),
        "--num-workers",
        str(args.num_workers),
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--test-ratio",
        str(args.test_ratio),
        "--target-pixels-per-unit",
        str(args.target_pixels_per_unit),
        "--crop-pad",
        str(args.crop_pad),
        "--max-scaled-longest-side",
        str(args.max_scaled_longest_side),
        "--input-mode",
        args.input_mode,
        "--device",
        args.device,
        "--patience",
        str(args.patience),
    ]

    if args.annotations is not None:
        cmd.extend(["--annotations", str(args.annotations)])
    if args.data_root is not None:
        cmd.extend(["--data-root", str(args.data_root)])
    if args.fragments_db is not None:
        cmd.extend(["--fragments-db", str(args.fragments_db)])
    if args.no_pretrained:
        cmd.append("--no-pretrained")
    if args.no_piece_head:
        cmd.append("--no-piece-head")
    if args.require_scale:
        cmd.append("--require-scale")

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed for seed {seed} with exit code {result.returncode}")
    return run_dir


def evaluate_heuristics_on_manifest(manifest_path: Path, target_pixels_per_unit: float) -> Dict[str, Any]:
    items = json.loads(manifest_path.read_text())
    extractor = EdgeExtractor()
    results: Dict[str, Any] = {}

    for method in HEURISTIC_METHODS:
        side_metrics = {
            key: {"labeled": 0, "correct": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
            for key in EDGE_LABEL_KEYS
        }
        exact_match_count = 0
        failures = []
        total = 0

        for item in items:
            image = cv2.imread(item["absolute_path"])
            if image is None:
                failures.append({"relative_path": item["relative_path"], "error": "read_fail"})
                continue

            payload = json.loads(item["segmentation_coords"])
            contours = payload.get("contours", [])
            contour_list = []
            for contour_like in contours:
                arr = np.asarray(contour_like, dtype=np.int32)
                if arr.ndim == 3 and arr.shape[1:] == (1, 2):
                    arr = arr.reshape(-1, 2)
                if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
                    continue
                contour_list.append(arr.reshape(-1, 1, 2))

            if not contour_list:
                failures.append({"relative_path": item["relative_path"], "error": "invalid_contours"})
                continue

            contour = max(contour_list, key=cv2.contourArea)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, contour_list, 255)

            try:
                if method == "hull_segments":
                    edge_data = extractor.classify_edges_hull_segments(contour=contour, mask=mask)
                elif method == "oriented_runs":
                    edge_data = extractor.classify_edges_oriented_runs(contour=contour, mask=mask)
                else:
                    edge_data = extractor.classify_edges_oriented_runs_scaled(
                        contour=contour,
                        mask=mask,
                        pixels_per_unit=item.get("pixels_per_unit"),
                        target_pixels_per_unit=target_pixels_per_unit,
                    )
            except Exception as exc:
                failures.append({"relative_path": item["relative_path"], "error": str(exc)})
                continue

            pred = {
                "has_top_edge": "top_edge" in edge_data.get("border_edges", []),
                "has_bottom_edge": "bottom_edge" in edge_data.get("border_edges", []),
                "has_left_edge": "left_edge" in edge_data.get("border_edges", []),
                "has_right_edge": "right_edge" in edge_data.get("border_edges", []),
            }

            all_correct = True
            for key in EDGE_LABEL_KEYS:
                gt = bool(item["labels"][key])
                pr = bool(pred[key])
                metric = side_metrics[key]
                metric["labeled"] += 1
                if gt == pr:
                    metric["correct"] += 1
                else:
                    all_correct = False
                if gt and pr:
                    metric["tp"] += 1
                elif (not gt) and (not pr):
                    metric["tn"] += 1
                elif (not gt) and pr:
                    metric["fp"] += 1
                else:
                    metric["fn"] += 1

            total += 1
            if all_correct:
                exact_match_count += 1

        for key, metric in side_metrics.items():
            labeled = max(metric["labeled"], 1)
            precision = metric["tp"] / (metric["tp"] + metric["fp"]) if (metric["tp"] + metric["fp"]) else 0.0
            recall = metric["tp"] / (metric["tp"] + metric["fn"]) if (metric["tp"] + metric["fn"]) else 0.0
            f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
            metric["accuracy"] = metric["correct"] / labeled
            metric["precision"] = precision
            metric["recall"] = recall
            metric["f1"] = f1

        results[method] = {
            "exact_match_count": exact_match_count,
            "exact_match_accuracy": (exact_match_count / total) if total else 0.0,
            "test_samples_scored": total,
            "failures": failures,
            "side_metrics": side_metrics,
        }

    return results


def evaluate_saved_model_on_manifest(run_dir: Path, device_name: str) -> Dict[str, Any]:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    manifest_items = json.loads((run_dir / "manifests" / "test.json").read_text())
    model_config = metadata.get("config", {})

    samples = [
        EdgeClassificationSample(
            relative_path=item["relative_path"],
            absolute_path=Path(item["absolute_path"]),
            collection=item["collection"],
            labels=item["labels"],
            piece_type=item["piece_type"],
            segmentation_coords=item["segmentation_coords"],
            pixels_per_unit=item.get("pixels_per_unit"),
            scale_unit=item.get("scale_unit"),
        )
        for item in manifest_items
    ]
    dataset = EdgeClassificationDataset(
        samples,
        img_size=int(model_config.get("img_size", 224)),
        augment=False,
        target_pixels_per_unit=model_config.get("target_pixels_per_unit", 100.0),
        input_mode=model_config.get("input_mode", "masked_rgb"),
        crop_pad=int(model_config.get("crop_pad", 6)),
        max_scaled_longest_side=int(model_config.get("max_scaled_longest_side", 1024)),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(model_config.get("batch_size", 16)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device_name == "cuda"),
    )

    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    model = load_edge_classifier(
        model_path=str(run_dir / "weights" / "best.pt"),
        device=str(device),
        pretrained=False,
        aux_piece_head=bool(model_config.get("aux_piece_head", True)),
        dropout_rate=float(model_config.get("dropout_rate", 0.3)),
        freeze_backbone_fraction=0.0,
    )
    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        thresholds=metadata.get("best_thresholds"),
        tune_on_dataset=False,
    )
    return {
        "exact_match_accuracy": metrics["exact_match_accuracy"],
        "micro_accuracy": metrics["micro_accuracy"],
        "piece_accuracy": metrics.get("piece_accuracy"),
        "side_metrics": metrics["side_metrics"],
        "test_samples_scored": len(samples),
    }


def aggregate_metrics(seed_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    method_names = ["model", *HEURISTIC_METHODS]
    aggregate: Dict[str, Any] = {}

    for method_name in method_names:
        exact_values = []
        micro_values = []
        piece_values = []
        side_values = {key: [] for key in EDGE_LABEL_KEYS}

        for result in seed_results:
            metrics = result[method_name]
            exact_values.append(float(metrics["exact_match_accuracy"]))
            if "micro_accuracy" in metrics:
                micro_values.append(float(metrics["micro_accuracy"]))
            if metrics.get("piece_accuracy") is not None:
                piece_values.append(float(metrics["piece_accuracy"]))
            for key in EDGE_LABEL_KEYS:
                side_values[key].append(float(metrics["side_metrics"][key]["accuracy"]))

        aggregate[method_name] = {
            "exact_match_accuracy_mean": statistics.mean(exact_values),
            "exact_match_accuracy_std": statistics.pstdev(exact_values) if len(exact_values) > 1 else 0.0,
            "seeds": len(exact_values),
            "side_accuracy_mean": {
                key: statistics.mean(values) for key, values in side_values.items()
            },
            "side_accuracy_std": {
                key: statistics.pstdev(values) if len(values) > 1 else 0.0
                for key, values in side_values.items()
            },
        }
        if micro_values:
            aggregate[method_name]["micro_accuracy_mean"] = statistics.mean(micro_values)
            aggregate[method_name]["micro_accuracy_std"] = statistics.pstdev(micro_values) if len(micro_values) > 1 else 0.0
        if piece_values:
            aggregate[method_name]["piece_accuracy_mean"] = statistics.mean(piece_values)
            aggregate[method_name]["piece_accuracy_std"] = statistics.pstdev(piece_values) if len(piece_values) > 1 else 0.0

    return aggregate


def write_csv(path: Path, seed_results: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "run_dir",
        "method",
        "exact_match_accuracy",
        "micro_accuracy",
        "piece_accuracy",
        "test_samples_scored",
        "top_accuracy",
        "bottom_accuracy",
        "left_accuracy",
        "right_accuracy",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in seed_results:
            for method_name in ["model", *HEURISTIC_METHODS]:
                metrics = result[method_name]
                writer.writerow(
                    {
                        "seed": result["seed"],
                        "run_dir": result["run_dir"],
                        "method": method_name,
                        "exact_match_accuracy": metrics["exact_match_accuracy"],
                        "micro_accuracy": metrics.get("micro_accuracy"),
                        "piece_accuracy": metrics.get("piece_accuracy"),
                        "test_samples_scored": metrics.get("test_samples_scored"),
                        "top_accuracy": metrics["side_metrics"]["has_top_edge"]["accuracy"],
                        "bottom_accuracy": metrics["side_metrics"]["has_bottom_edge"]["accuracy"],
                        "left_accuracy": metrics["side_metrics"]["has_left_edge"]["accuracy"],
                        "right_accuracy": metrics["side_metrics"]["has_right_edge"]["accuracy"],
                    }
                )


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seed_results: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        run_dir = run_training_for_seed(args, seed)
        model_metrics = evaluate_saved_model_on_manifest(run_dir, args.device)
        heuristic_metrics = evaluate_heuristics_on_manifest(
            run_dir / "manifests" / "test.json",
            target_pixels_per_unit=args.target_pixels_per_unit,
        )
        seed_results.append(
            {
                "seed": seed,
                "run_dir": str(run_dir),
                "model": model_metrics,
                **heuristic_metrics,
            }
        )

        print(
            "model exact={:.3f} hull={:.3f} oriented={:.3f} oriented_scaled={:.3f}".format(
                seed_results[-1]["model"]["exact_match_accuracy"],
                seed_results[-1]["hull_segments"]["exact_match_accuracy"],
                seed_results[-1]["oriented_runs"]["exact_match_accuracy"],
                seed_results[-1]["oriented_runs_scaled"]["exact_match_accuracy"],
            )
        )

    aggregate = aggregate_metrics(seed_results)
    report = {
        "seeds": seeds,
        "config": {
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "target_pixels_per_unit": args.target_pixels_per_unit,
            "input_mode": args.input_mode,
            "require_scale": args.require_scale,
            "pretrained": not args.no_pretrained,
            "aux_piece_head": not args.no_piece_head,
        },
        "per_seed": seed_results,
        "aggregate": aggregate,
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.report_csv, seed_results)

    print("\nAggregate exact-match means:")
    for method_name in ["model", *HEURISTIC_METHODS]:
        metrics = aggregate[method_name]
        print(
            f"{method_name}: mean={metrics['exact_match_accuracy_mean']:.3f} std={metrics['exact_match_accuracy_std']:.3f}"
        )
    print(f"Wrote {args.report_json}")
    print(f"Wrote {args.report_csv}")


if __name__ == "__main__":
    main()
