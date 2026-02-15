"""
CLI orchestrator: extract edges → filter → match → compute positions → write DB.

Usage:
    python -m reconstruction.pipeline \
        --db-path src/web/web-canvas/electron/resources/database/fragments.db \
        --top-k 5 --version "1.0"
"""

import argparse
import json
import sqlite3
import time
from typing import Dict, List

from .db import (
    MatchRow,
    clear_matches,
    ensure_edge_matches_table,
    get_eligible_fragments,
    insert_matches,
)
from .edge_extraction import ScaleAwareDescriptor, ScaleAwareEdgeExtractor
from .position_computer import PositionComputer
from .scale_aware_matcher import ScaleAwareMatcher, build_descriptor_index


def run_pipeline(db_path: str, top_k: int = 5, version: str = "1.0", limit: int = 0) -> None:
    conn = sqlite3.connect(db_path)
    ensure_edge_matches_table(conn)

    print(f"[pipeline] Connected to {db_path}")
    print(f"[pipeline] Parameters: top_k={top_k}, version={version}")

    # 1. Load eligible fragments
    fragments = get_eligible_fragments(conn)
    print(f"[pipeline] Found {len(fragments)} eligible fragments (have scale + segmentation)")
    if not fragments:
        print("[pipeline] No eligible fragments. Exiting.")
        conn.close()
        return

    # 2. Extract descriptors
    extractor = ScaleAwareEdgeExtractor()
    all_descriptors: List[ScaleAwareDescriptor] = []
    frag_descriptor_map: Dict[str, List[ScaleAwareDescriptor]] = {}

    t0 = time.time()
    n_frags = len(fragments)
    for i, frag in enumerate(fragments, 1):
        try:
            descs = extractor.extract_from_segmentation_coords(
                segmentation_coords_json=frag.segmentation_coords,
                image_shape=(frag.image_height, frag.image_width),
                fragment_id=frag.fragment_id,
                pixels_per_unit=frag.pixels_per_unit,
                scale_unit=frag.scale_unit,
                line_count=frag.line_count,
                script_type=frag.script_type,
            )
            all_descriptors.extend(descs)
            frag_descriptor_map[frag.fragment_id] = descs
            if i % 50 == 0 or i == n_frags:
                print(f"  [extract] {i}/{n_frags} fragments ({len(all_descriptors)} edges so far)")
        except Exception as e:
            print(f"  [WARN] Skipping {frag.fragment_id}: {e}")

    t_extract = time.time() - t0
    print(f"[pipeline] Extracted {len(all_descriptors)} tear-edge descriptors in {t_extract:.1f}s")

    if len(all_descriptors) < 2:
        print("[pipeline] Not enough descriptors for matching. Exiting.")
        conn.close()
        return

    # 3. Build index and match
    print("[pipeline] Building position+script+length index...")
    index = build_descriptor_index(all_descriptors)
    index_stats = index.stats()
    print(f"[pipeline] Index buckets: {index_stats}")

    # Filter to only edges that have a script type (required for matching)
    matchable = [d for d in all_descriptors if d.script_type]
    if limit > 0:
        matchable = matchable[:limit]
        print(f"[pipeline] Limited to {len(matchable)} edges (--limit {limit})")
    else:
        print(f"[pipeline] {len(matchable)} edges have script type (matchable)")

    matcher = ScaleAwareMatcher()
    position_computer = PositionComputer()

    all_match_rows: List[MatchRow] = []
    t0 = time.time()

    n_descs = len(matchable)
    for desc_idx, desc in enumerate(matchable, 1):
        if desc_idx % 500 == 0 or desc_idx == n_descs:
            print(f"  [match] {desc_idx}/{n_descs} edges processed ({len(all_match_rows)} matches so far)")
        matches = matcher.find_matches_indexed(desc, index, top_k=top_k)
        for rank_idx, (score, details, cand) in enumerate(matches):
            # Compute placement
            try:
                placement = position_computer.compute_placement(desc, cand)
            except Exception:
                placement = None

            confidence = 1.0 / (1.0 + score)

            all_match_rows.append(MatchRow(
                fragment_a_id=desc.fragment_id,
                edge_a_name=desc.edge_name,
                fragment_b_id=cand.fragment_id,
                edge_b_name=cand.edge_name,
                score=score,
                rank=rank_idx + 1,
                confidence=confidence,
                score_details=json.dumps({k: round(v, 6) for k, v in details.items()}),
                relative_x_cm=placement.relative_x_cm if placement else 0,
                relative_y_cm=placement.relative_y_cm if placement else 0,
                rotation_deg=placement.rotation_deg if placement else 0,
                algorithm_version=version,
            ))

    t_match = time.time() - t0
    print(f"[pipeline] Computed {len(all_match_rows)} matches in {t_match:.1f}s")

    # 4. Write to DB
    clear_matches(conn, version)
    n = insert_matches(conn, all_match_rows)
    print(f"[pipeline] Wrote {n} rows to edge_matches table")

    conn.close()
    print("[pipeline] Done.")


def main():
    parser = argparse.ArgumentParser(description="Run edge matching pipeline")
    parser.add_argument("--db-path", required=True, help="Path to fragments.db")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K matches per edge")
    parser.add_argument("--version", default="1.0", help="Algorithm version tag")
    parser.add_argument("--limit", type=int, default=0, help="Limit edges to match (0 = all)")
    args = parser.parse_args()
    run_pipeline(args.db_path, args.top_k, args.version, args.limit)


if __name__ == "__main__":
    main()
