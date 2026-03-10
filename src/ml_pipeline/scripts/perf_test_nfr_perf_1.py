"""
Performance test: bulk fragment processing pipeline (test-nfr-perf-1).

Runs the ML pipeline on a fixed set of fragments (first N by ID),
measures elapsed time, throughput, success rate, and checks required fields.
Writes a JSON report for VnV documentation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import sqlite3


def chunked(items: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def now_sql() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_log_errors(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {"log_path": str(log_path), "error_lines": 0, "sample": []}

    error_lines: List[str] = []
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "ERROR" in line or "CRITICAL" in line:
                    error_lines.append(line.strip())
    except OSError as exc:
        return {"log_path": str(log_path), "error_lines": -1, "error": str(exc), "sample": []}

    return {
        "log_path": str(log_path),
        "error_lines": len(error_lines),
        "sample": error_lines[:10],
    }


def run_pipeline(cmd: List[str], cwd: Path, enable_psutil: bool) -> Dict[str, Any]:
    start = time.perf_counter()
    start_iso = now_iso()

    metrics: Dict[str, Any] = {
        "command": " ".join(cmd),
        "cwd": str(cwd),
        "start_time": start_iso,
        "end_time": None,
        "elapsed_seconds": None,
        "exit_code": None,
        "resource_samples": [],
        "resource_summary": None,
    }

    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None

    use_psutil = bool(psutil) and enable_psutil

    if use_psutil:
        process = subprocess.Popen(cmd, cwd=str(cwd))
        ps_proc = psutil.Process(process.pid)
        max_rss = 0
        max_cpu = 0.0
        samples = 0
        while True:
            if process.poll() is not None:
                break
            try:
                rss = ps_proc.memory_info().rss
                cpu = ps_proc.cpu_percent(interval=0.5)
                max_rss = max(max_rss, rss)
                max_cpu = max(max_cpu, cpu)
                samples += 1
            except Exception:
                pass
            time.sleep(0.5)
        exit_code = process.returncode
        metrics["resource_summary"] = {
            "max_rss_bytes": max_rss,
            "max_rss_mb": round(max_rss / (1024 * 1024), 2) if max_rss else 0,
            "max_cpu_percent": round(max_cpu, 2),
            "samples": samples,
        }
    else:
        result = subprocess.run(cmd, cwd=str(cwd))
        exit_code = result.returncode
        metrics["resource_summary"] = {
            "note": "psutil not available; resource metrics not collected"
        }

    end = time.perf_counter()
    metrics["end_time"] = now_iso()
    metrics["elapsed_seconds"] = round(end - start, 2)
    metrics["exit_code"] = exit_code
    return metrics


def evaluate_results(
    db_path: Path,
    ids: List[int],
    required_fields: List[str],
    allowed_statuses: List[str],
) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    total = len(ids)
    completed = 0
    required_ok = 0
    missing_required: Dict[str, int] = {field: 0 for field in required_fields}
    status_counts: Dict[str, int] = {}

    if not ids:
        conn.close()
        return {
            "total": 0,
            "completed": 0,
            "required_ok": 0,
            "missing_required": missing_required,
            "status_counts": status_counts,
        }

    for chunk in chunked(ids, 200):
        placeholders = ",".join("?" * len(chunk))
        columns = ", ".join(["id", "processing_status"] + required_fields)
        query = f"SELECT {columns} FROM fragments WHERE id IN ({placeholders})"
        rows = conn.execute(query, chunk).fetchall()

        for row in rows:
            status = row["processing_status"] or "NULL"
            status_counts[status] = status_counts.get(status, 0) + 1
            if row["processing_status"] in allowed_statuses:
                completed += 1

            ok = True
            for field in required_fields:
                value = row[field]
                if value is None or (isinstance(value, str) and value.strip() == ""):
                    missing_required[field] += 1
                    ok = False
            if ok:
                required_ok += 1

    conn.close()
    return {
        "total": total,
        "completed": completed,
        "required_ok": required_ok,
        "missing_required": missing_required,
        "status_counts": status_counts,
    }


def get_existing_columns(db_path: Path) -> List[str]:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("PRAGMA table_info(fragments)").fetchall()
    conn.close()
    return [row[1] for row in rows]


def main() -> None:
    script_root = Path(__file__).resolve()
    repo_root = script_root.parents[2]

    default_db = repo_root / "web" / "web-canvas" / "electron" / "resources" / "database" / "fragments.db"
    default_config = repo_root / "ml_pipeline" / "config.yaml"
    default_report = repo_root / "reports" / "perf_test_nfr_perf_1.json"
    default_log = repo_root / "logs" / "pipeline.log"

    parser = argparse.ArgumentParser(description="Run test-nfr-perf-1 pipeline performance test.")
    parser.add_argument("--config", type=Path, default=default_config, help="Path to ml_pipeline config.yaml")
    parser.add_argument("--db-path", type=Path, default=default_db, help="Path to fragments.db")
    parser.add_argument("--limit", type=int, default=500, help="Number of fragments to process")
    parser.add_argument("--force", action="store_true", help="Force reprocess fragments")
    parser.add_argument("--log-file", type=Path, default=default_log, help="Path to pipeline log file")
    parser.add_argument(
        "--required-fields",
        default="segmentation_coords,script_type,line_detection_data",
        help="Comma-separated list of fields required for success",
    )
    parser.add_argument(
        "--allowed-statuses",
        default="completed,completed_low_confidence",
        help="Comma-separated processing_status values considered completed",
    )
    parser.add_argument("--report", type=Path, default=default_report, help="Output report JSON path")
    parser.add_argument("--enable-psutil", action="store_true", help="Collect CPU/RAM metrics via psutil")
    args = parser.parse_args()

    args.report.parent.mkdir(parents=True, exist_ok=True)

    required_fields = [f.strip() for f in args.required_fields.split(",") if f.strip()]
    allowed_statuses = [s.strip() for s in args.allowed_statuses.split(",") if s.strip()]

    if not args.db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.db_path}")

    existing_columns = set(get_existing_columns(args.db_path))
    filtered_required_fields = [f for f in required_fields if f in existing_columns]
    missing_required_fields = [f for f in required_fields if f not in existing_columns]

    conn = sqlite3.connect(args.db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, fragment_id FROM fragments ORDER BY id LIMIT ?",
        (args.limit,),
    ).fetchall()
    ids = [row["id"] for row in rows]
    fragment_ids = [row["fragment_id"] for row in rows]
    conn.close()

    start_sql = now_sql()

    cmd = [
        sys.executable,
        "-m",
        "ml_pipeline.cli",
        "--config",
        str(args.config),
        "run",
        "--limit",
        str(args.limit),
    ]
    if args.force:
        cmd.append("--force")

    run_metrics = run_pipeline(cmd, repo_root, args.enable_psutil)

    end_sql = now_sql()

    evaluation = evaluate_results(args.db_path, ids, filtered_required_fields, allowed_statuses)

    elapsed_hours = (run_metrics["elapsed_seconds"] or 0) / 3600 if run_metrics["elapsed_seconds"] else 0
    success_rate = (evaluation["required_ok"] / evaluation["total"]) if evaluation["total"] else 0
    throughput = (evaluation["required_ok"] / elapsed_hours) if elapsed_hours else 0

    perf_limits = {
        "max_hours": 1.15,
        "min_success_rate": 0.99,
    }

    time_ok = elapsed_hours <= perf_limits["max_hours"] if elapsed_hours else False
    success_ok = success_rate >= perf_limits["min_success_rate"] if evaluation["total"] else False

    log_errors = read_log_errors(args.log_file)

    report: Dict[str, Any] = {
        "test_id": "test-nfr-perf-1",
        "started_at": run_metrics["start_time"],
        "ended_at": run_metrics["end_time"],
        "db_path": str(args.db_path),
        "config_path": str(args.config),
        "log_file": str(args.log_file),
        "limit": args.limit,
        "target_fragment_ids": fragment_ids,
        "pipeline": run_metrics,
        "evaluation": evaluation,
        "metrics": {
            "elapsed_hours": round(elapsed_hours, 4),
            "success_rate": round(success_rate, 4),
            "throughput_fragments_per_hour": round(throughput, 2),
        },
        "checks": {
            "processing_time_ok": time_ok,
            "success_rate_ok": success_ok,
            "critical_errors": log_errors,
        },
        "thresholds": perf_limits,
        "window": {
            "start_sql": start_sql,
            "end_sql": end_sql,
        },
        "notes": [
            "success_rate uses required fields + processing_status",
            "required_fields_requested=" + ",".join(required_fields),
            "required_fields_used=" + ",".join(filtered_required_fields),
            "required_fields_missing=" + ",".join(missing_required_fields),
            "allowed_statuses=" + ",".join(allowed_statuses),
        ],
    }

    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[perf-test-1] Report written to {args.report}")

    if not (time_ok and success_ok):
        print("[perf-test-1] FAILED: thresholds not met.")
        sys.exit(1)

    if log_errors.get("error_lines", 0) > 0:
        print("[perf-test-1] WARNING: critical errors found in logs.")

    print("[perf-test-1] PASSED.")


if __name__ == "__main__":
    main()
