"""
Evaluate edge detection methods against manual edge annotations.

This script:
1. Loads reviewed annotations from edge-annotations.json
2. Runs the segmentation model to obtain fragment masks
3. Runs one or more edge-detection methods on the resulting mask/contour
4. Reports per-side accuracy and exact-match accuracy
5. Writes a JSON summary and detailed CSV for inspection
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


SCRIPT_ROOT = Path(__file__).resolve()
SRC_ROOT = SCRIPT_ROOT.parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ml_pipeline.processors.edge_detection_processor import EdgeExtractor  # noqa: E402


EDGE_KEYS = [
    "has_top_edge",
    "has_bottom_edge",
    "has_left_edge",
    "has_right_edge",
]
METHODS = ("hull_segments", "bbox_runs", "oriented_runs", "oriented_runs_scaled")


@dataclass
class AnnotationSample:
    relative_path: str
    absolute_path: Path
    labels: Dict[str, Optional[bool]]
    reviewed: bool
    skipped: bool
    notes: str
    scale_unit: Optional[str] = None
    pixels_per_unit: Optional[float] = None


def parse_args() -> argparse.Namespace:
    default_annotations = SRC_ROOT / "web" / "web-canvas" / "data" / "edge-annotations.json"
    default_data_root = SRC_ROOT / "web" / "web-canvas" / "data"
    default_seg_model = SRC_ROOT / "models" / "segmentation" / "best.pt"
    default_scale_db = SRC_ROOT / "web" / "web-canvas" / "electron" / "resources" / "database" / "fragments.db"
    default_report_json = SRC_ROOT.parent / "reports" / "edge_detection_annotation_eval.json"
    default_report_csv = SRC_ROOT.parent / "reports" / "edge_detection_annotation_eval_details.csv"

    parser = argparse.ArgumentParser(description="Evaluate edge detection against manual annotations.")
    parser.add_argument("--annotations", type=Path, default=default_annotations)
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--seg-model", type=Path, default=default_seg_model)
    parser.add_argument("--scale-db", type=Path, default=default_scale_db)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-scale", action="store_true")
    parser.add_argument("--target-pixels-per-unit", type=float, default=100.0)
    parser.add_argument(
        "--methods",
        default="hull_segments,bbox_runs,oriented_runs,oriented_runs_scaled",
        help="Comma-separated list of methods to evaluate",
    )
    parser.add_argument("--report-json", type=Path, default=default_report_json)
    parser.add_argument("--report-csv", type=Path, default=default_report_csv)
    return parser.parse_args()


def load_scale_rows(scale_db_path: Path) -> Dict[str, Dict[str, Any]]:
    import sqlite3

    if not scale_db_path.exists():
        return {}

    conn = sqlite3.connect(scale_db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT image_path, scale_unit, pixels_per_unit, scale_detection_status FROM fragments"
        ).fetchall()
    finally:
        conn.close()

    scale_rows: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        scale_rows[str(row["image_path"]).replace("\\", "/")] = {
            "scale_unit": row["scale_unit"],
            "pixels_per_unit": row["pixels_per_unit"],
            "scale_detection_status": row["scale_detection_status"],
        }
    return scale_rows


def load_samples(
    annotations_path: Path,
    data_root: Path,
    scale_rows: Optional[Dict[str, Dict[str, Any]]] = None,
    require_scale: bool = False,
) -> List[AnnotationSample]:
    payload = json.loads(annotations_path.read_text())
    annotations = payload.get("annotations", {})
    samples: List[AnnotationSample] = []
    scale_rows = scale_rows or {}

    for relative_path, record in annotations.items():
        if not record.get("reviewed", False):
            continue
        if record.get("skipped", False):
            continue

        labels: Dict[str, Optional[bool]] = {
            key: record.get(key)
            for key in EDGE_KEYS
        }
        scale_row = scale_rows.get(relative_path, {})
        pixels_per_unit = scale_row.get("pixels_per_unit")
        scale_status = scale_row.get("scale_detection_status")
        if require_scale and not (pixels_per_unit is not None and scale_status == "success"):
            continue
        samples.append(
            AnnotationSample(
                relative_path=relative_path,
                absolute_path=data_root / relative_path,
                labels=labels,
                reviewed=bool(record.get("reviewed")),
                skipped=bool(record.get("skipped")),
                notes=str(record.get("notes", "")),
                scale_unit=scale_row.get("scale_unit"),
                pixels_per_unit=float(pixels_per_unit) if pixels_per_unit is not None else None,
            )
        )

    samples.sort(key=lambda sample: sample.relative_path)
    return samples


def summarize_side_counts(samples: Iterable[AnnotationSample]) -> Dict[str, int]:
    counts = {key: 0 for key in EDGE_KEYS}
    for sample in samples:
        for key, value in sample.labels.items():
            if value is not None:
                counts[key] += 1
    return counts


def run_segmentation(
    model: YOLO,
    image_path: Path,
    conf: float,
    iou: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    result = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        verbose=False,
        device=device,
    )[0]
    if result.masks is None or len(result.masks) == 0:
        raise ValueError("No masks detected")

    mask = result.masks.data[0].cpu().numpy()
    mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
    mask_binary = (mask_resized > 0.5).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError("No contour extracted from mask")

    contour = max(contours, key=cv2.contourArea)
    mask_uint8 = mask_binary * 255
    return img, mask_uint8, contour


def derive_prediction(
    extractor: EdgeExtractor,
    method: str,
    contour: np.ndarray,
    mask: np.ndarray,
    pixels_per_unit: Optional[float] = None,
    target_pixels_per_unit: float = 150.0,
) -> Dict[str, Any]:
    if method == "hull_segments":
        return extractor.classify_edges_hull_segments(
            contour=contour,
            mask=mask,
        )
    if method == "bbox_runs":
        return extractor.classify_edges(
            contour=contour,
            mask=mask,
        )
    if method == "oriented_runs":
        return extractor.classify_edges_oriented_runs(
            contour=contour,
            mask=mask,
        )
    if method == "oriented_runs_scaled":
        return extractor.classify_edges_oriented_runs_scaled(
            contour=contour,
            mask=mask,
            pixels_per_unit=pixels_per_unit,
            target_pixels_per_unit=target_pixels_per_unit,
        )
    raise ValueError(f"Unknown method: {method}")


def init_method_summary(method: str) -> Dict[str, Any]:
    return {
        "method": method,
        "images_attempted": 0,
        "segmentation_failures": 0,
        "prediction_failures": 0,
        "images_scored": 0,
        "exact_match": {
            "count": 0,
            "total": 0,
            "accuracy": None,
        },
        "side_metrics": {
            key: {
                "labeled": 0,
                "correct": 0,
                "accuracy": None,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "precision": None,
                "recall": None,
                "f1": None,
            }
            for key in EDGE_KEYS
        },
        "worst_examples": [],
    }


def finalize_method_summary(summary: Dict[str, Any]) -> None:
    exact = summary["exact_match"]
    if exact["total"] > 0:
        exact["accuracy"] = exact["count"] / exact["total"]

    for metrics in summary["side_metrics"].values():
        if metrics["labeled"] > 0:
            metrics["accuracy"] = metrics["correct"] / metrics["labeled"]
        tp = metrics["tp"]
        fp = metrics["fp"]
        fn = metrics["fn"]
        precision = tp / (tp + fp) if tp + fp > 0 else None
        recall = tp / (tp + fn) if tp + fn > 0 else None
        if precision is not None and recall is not None and precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1

    summary["worst_examples"] = sorted(
        summary["worst_examples"],
        key=lambda item: (-item["errors"], item["relative_path"]),
    )[:12]


def main() -> None:
    args = parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods: {', '.join(invalid)}")

    scale_rows = load_scale_rows(args.scale_db)
    samples = load_samples(
        args.annotations,
        args.data_root,
        scale_rows=scale_rows,
        require_scale=args.require_scale,
    )
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No reviewed, non-skipped annotations found.")

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading segmentation model: {args.seg_model}")
    model = YOLO(str(args.seg_model))
    extractor = EdgeExtractor()

    report: Dict[str, Any] = {
        "annotations_path": str(args.annotations),
        "data_root": str(args.data_root),
        "segmentation_model": str(args.seg_model),
        "scale_db": str(args.scale_db),
        "require_scale": args.require_scale,
        "target_pixels_per_unit": args.target_pixels_per_unit,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "sample_count": len(samples),
        "labeled_side_counts": summarize_side_counts(samples),
        "methods": {method: init_method_summary(method) for method in methods},
    }

    detail_rows: List[Dict[str, Any]] = []

    for index, sample in enumerate(samples, start=1):
        if index == 1 or index % 10 == 0 or index == len(samples):
            print(f"[{index}/{len(samples)}] {sample.relative_path}")

        segmentation_error: Optional[str] = None
        contour: Optional[np.ndarray] = None
        mask: Optional[np.ndarray] = None

        try:
            _, mask, contour = run_segmentation(
                model=model,
                image_path=sample.absolute_path,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
            )
        except Exception as exc:  # noqa: BLE001
            segmentation_error = str(exc)

        for method in methods:
            summary = report["methods"][method]
            summary["images_attempted"] += 1

            row: Dict[str, Any] = {
                "method": method,
                "relative_path": sample.relative_path,
                "segmentation_ok": segmentation_error is None,
                "segmentation_error": segmentation_error,
                "gt_top": sample.labels["has_top_edge"],
                "gt_bottom": sample.labels["has_bottom_edge"],
                "gt_left": sample.labels["has_left_edge"],
                "gt_right": sample.labels["has_right_edge"],
                "scale_unit": sample.scale_unit,
                "pixels_per_unit": sample.pixels_per_unit,
            }

            if segmentation_error is not None or contour is None or mask is None:
                summary["segmentation_failures"] += 1
                detail_rows.append(row)
                continue

            try:
                prediction = derive_prediction(
                    extractor,
                    method,
                    contour,
                    mask,
                    pixels_per_unit=sample.pixels_per_unit,
                    target_pixels_per_unit=args.target_pixels_per_unit,
                )
                border_edges = set(prediction.get("border_edges", []))
            except Exception as exc:  # noqa: BLE001
                summary["prediction_failures"] += 1
                row["prediction_error"] = str(exc)
                detail_rows.append(row)
                continue

            summary["images_scored"] += 1

            all_known_correct = True
            known_labels = 0
            errors = 0

            for side_key, side_name in (
                ("has_top_edge", "top_edge"),
                ("has_bottom_edge", "bottom_edge"),
                ("has_left_edge", "left_edge"),
                ("has_right_edge", "right_edge"),
            ):
                gt_value = sample.labels[side_key]
                pred_value = side_name in border_edges
                row[f"pred_{side_key}"] = pred_value

                if gt_value is None:
                    continue

                known_labels += 1
                side_metrics = summary["side_metrics"][side_key]
                side_metrics["labeled"] += 1

                if pred_value == gt_value:
                    side_metrics["correct"] += 1
                else:
                    all_known_correct = False
                    errors += 1

                if gt_value and pred_value:
                    side_metrics["tp"] += 1
                elif gt_value and not pred_value:
                    side_metrics["fn"] += 1
                elif not gt_value and pred_value:
                    side_metrics["fp"] += 1
                else:
                    side_metrics["tn"] += 1

            row["piece_type"] = prediction.get("piece_type")
            row["pred_borders"] = ",".join(sorted(border_edges))
            row["scale_normalization_json"] = (
                json.dumps(prediction.get("scale_normalization", {}), sort_keys=True)
                if prediction.get("scale_normalization") is not None
                else ""
            )
            row["scores_json"] = json.dumps(prediction.get("scores", {}), sort_keys=True)
            row["known_labels"] = known_labels
            row["errors"] = errors

            if known_labels > 0:
                summary["exact_match"]["total"] += 1
                if all_known_correct:
                    summary["exact_match"]["count"] += 1
                else:
                    summary["worst_examples"].append(
                        {
                            "relative_path": sample.relative_path,
                            "errors": errors,
                            "pred_borders": sorted(border_edges),
                            "piece_type": prediction.get("piece_type"),
                        }
                    )

            detail_rows.append(row)

    for method in methods:
        finalize_method_summary(report["methods"][method])

    with args.report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    fieldnames = [
        "method",
        "relative_path",
        "segmentation_ok",
        "segmentation_error",
        "prediction_error",
        "gt_top",
        "gt_bottom",
        "gt_left",
        "gt_right",
        "scale_unit",
        "pixels_per_unit",
        "pred_has_top_edge",
        "pred_has_bottom_edge",
        "pred_has_left_edge",
        "pred_has_right_edge",
        "piece_type",
        "pred_borders",
        "known_labels",
        "errors",
        "scale_normalization_json",
        "scores_json",
    ]
    with args.report_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    print(f"Wrote summary JSON: {args.report_json}")
    print(f"Wrote detail CSV: {args.report_csv}")
    for method in methods:
        summary = report["methods"][method]
        exact = summary["exact_match"]
        print(
            f"{method}: exact_match={exact['count']}/{exact['total']}"
            + (f" ({exact['accuracy']:.3f})" if exact["accuracy"] is not None else "")
            + f", scored={summary['images_scored']}, seg_fail={summary['segmentation_failures']}, pred_fail={summary['prediction_failures']}"
        )


if __name__ == "__main__":
    main()
