"""
Generate a VnV performance report from perf test JSON outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def format_perf1(data: Optional[Dict[str, Any]]) -> str:
    if not data:
        return (
            "Status: Not run (missing perf_test_nfr_perf_1.json)\n"
            "Notes: Run `python ml_pipeline/scripts/perf_test_nfr_perf_1.py` to generate results.\n"
        )

    metrics = data.get("metrics", {})
    checks = data.get("checks", {})
    thresholds = data.get("thresholds", {})
    evaluation = data.get("evaluation", {})

    return (
        f"Status: {'PASS' if checks.get('processing_time_ok') and checks.get('success_rate_ok') else 'FAIL'}\n"
        f"Elapsed hours: {metrics.get('elapsed_hours')}\n"
        f"Success rate: {metrics.get('success_rate')}\n"
        f"Throughput (fragments/hour): {metrics.get('throughput_fragments_per_hour')}\n"
        f"Thresholds: max_hours={thresholds.get('max_hours')}, min_success_rate={thresholds.get('min_success_rate')}\n"
        f"Total fragments: {evaluation.get('total')}\n"
        f"Completed (status OK): {evaluation.get('completed')}\n"
        f"Required fields OK: {evaluation.get('required_ok')}\n"
        f"Log errors: {checks.get('critical_errors', {}).get('error_lines')}\n"
    )


def format_perf2(data: Optional[Dict[str, Any]]) -> str:
    if not data:
        return (
            "Status: Not run (missing perf_test_nfr_perf_2.json)\n"
            "Notes: Run `npm run perf:test2` in web/web-canvas to generate results.\n"
        )

    checks = data.get("checks", {})
    search = data.get("search", {})
    canvas = data.get("canvas", {})
    totals = data.get("totals", {})

    status_ok = all([
        checks.get("mean_search_ok"),
        checks.get("p95_search_ok"),
        checks.get("canvas_500ms_ok"),
        checks.get("error_rate_ok"),
    ])

    return (
        f"Status: {'PASS' if status_ok else 'FAIL'}\n"
        f"Operations: {totals.get('operations')} (errors: {totals.get('errors')}, error_rate: {totals.get('error_rate')})\n"
        f"Search mean ms: {search.get('mean_ms')}\n"
        f"Search p95 ms: {search.get('p95_ms')}\n"
        f"Canvas p95 ms: {canvas.get('p95_ms')}\n"
        f"Canvas under 500ms ratio: {canvas.get('under_500ms_ratio')}\n"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_perf1 = repo_root / "reports" / "perf_test_nfr_perf_1.json"
    default_perf2 = repo_root / "reports" / "perf_test_nfr_perf_2.json"
    default_out = repo_root / "reports" / "vnv_performance_report.md"

    parser = argparse.ArgumentParser(description="Generate VnV performance report.")
    parser.add_argument("--perf1", type=Path, default=default_perf1)
    parser.add_argument("--perf2", type=Path, default=default_perf2)
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()

    perf1 = load_json(args.perf1)
    perf2 = load_json(args.perf2)

    content = (
        "# VnV Performance Report\n\n"
        "## test-nfr-perf-1\n"
        f"{format_perf1(perf1)}\n"
        "## test-nfr-perf-2\n"
        f"{format_perf2(perf2)}\n"
    )

    args.out.write_text(content, encoding="utf-8")
    print(f"[vnv-report] Wrote {args.out}")


if __name__ == "__main__":
    main()
