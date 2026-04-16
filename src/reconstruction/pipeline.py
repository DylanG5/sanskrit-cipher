"""
CLI orchestrator: 3-stage fragment matching pipeline.

Stage 1: Embedding retrieval — narrow 10K fragments to ~50 candidates per fragment
Stage 2: Geometric edge matching + overlap check — verify physical plausibility
Stage 3: Visual join scoring — boundary similarity + text line continuity

Usage:
    # Full 3-stage pipeline (requires embeddings to be pre-computed)
    python -m reconstruction.pipeline \
        --db-path src/web/web-canvas/electron/resources/database/fragments.db \
        --cache-dir src/web/web-canvas/electron/resources/cache \
        --top-k 5 --version "2.0"

    # Legacy mode (geometry-only, no embeddings required)
    python -m reconstruction.pipeline \
        --db-path src/web/web-canvas/electron/resources/database/fragments.db \
        --top-k 5 --version "1.0" --no-embeddings
"""

import argparse
import json
import sqlite3
import time
from typing import Dict, List, Optional

from .db import (
    FragmentRow,
    MatchRow,
    clear_matches,
    ensure_edge_matches_table,
    get_eligible_fragments,
    insert_matches,
)
from .edge_extraction import ScaleAwareDescriptor, ScaleAwareEdgeExtractor
from .overlap_detector import check_overlap
from .position_computer import PlacementResult, PositionComputer
from .scale_aware_matcher import ScaleAwareMatcher, build_descriptor_index


def _build_fragment_lookup(fragments: List[FragmentRow]) -> Dict[str, FragmentRow]:
    """Build a fragment_id → FragmentRow lookup."""
    return {f.fragment_id: f for f in fragments}


def run_pipeline(
    db_path: str,
    top_k: int = 5,
    version: str = "2.0",
    limit: int = 0,
    cache_dir: Optional[str] = None,
    use_embeddings: bool = True,
    embedding_top_k: int = 50,
    use_join_scoring: bool = True,
) -> None:
    conn = sqlite3.connect(db_path)
    ensure_edge_matches_table(conn)

    print(f"[pipeline] Connected to {db_path}")
    print(f"[pipeline] Parameters: top_k={top_k}, version={version}, "
          f"embeddings={'on' if use_embeddings else 'off'}, "
          f"join_scoring={'on' if use_join_scoring else 'off'}")

    # ---------------------------------------------------------------
    # Load eligible fragments
    # ---------------------------------------------------------------
    fragments = get_eligible_fragments(conn)
    frag_lookup = _build_fragment_lookup(fragments)
    print(f"[pipeline] Found {len(fragments)} eligible fragments (have scale + segmentation)")
    if not fragments:
        print("[pipeline] No eligible fragments. Exiting.")
        conn.close()
        return

    # ---------------------------------------------------------------
    # Stage 1: Embedding retrieval (optional)
    # ---------------------------------------------------------------
    embedding_index = None
    embedding_similarities: Dict[str, Dict[str, float]] = {}  # frag_a -> {frag_b: sim}

    if use_embeddings and cache_dir:
        try:
            from .embedding_index import EmbeddingIndex
            print("[pipeline] Stage 1: Building embedding index...")
            t0 = time.time()
            embedding_index = EmbeddingIndex.build(conn, cache_dir)
            print(f"[pipeline] Embedding index built: {embedding_index.size} fragments in {time.time() - t0:.1f}s")

            # Pre-compute candidate lists per fragment
            print(f"[pipeline] Retrieving top-{embedding_top_k} candidates per fragment...")
            for frag in fragments:
                candidates = embedding_index.query(frag.fragment_id, top_k=embedding_top_k)
                if candidates:
                    embedding_similarities[frag.fragment_id] = {
                        c.fragment_id: c.similarity for c in candidates
                    }
            print(f"[pipeline] Stage 1 complete: {len(embedding_similarities)} fragments have candidates")
        except ImportError as e:
            print(f"[pipeline] Stage 1 skipped (missing dependency: {e})")
            use_embeddings = False
        except ValueError as e:
            print(f"[pipeline] Stage 1 skipped ({e})")
            use_embeddings = False

    # ---------------------------------------------------------------
    # Extract edge descriptors
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # Stage 2: Geometric matching + overlap check
    # ---------------------------------------------------------------
    print("[pipeline] Stage 2: Geometric matching with overlap validation...")

    matchable = [d for d in all_descriptors if d.script_type]
    if limit > 0:
        matchable = matchable[:limit]
        print(f"[pipeline] Limited to {len(matchable)} edges (--limit {limit})")
    else:
        print(f"[pipeline] {len(matchable)} edges have script type (matchable)")

    # If embeddings available, restrict to only edges whose fragments have candidates
    embedding_candidate_set: Optional[Dict[str, set]] = None
    if use_embeddings and embedding_similarities:
        embedding_candidate_set = {
            frag_id: set(cands.keys())
            for frag_id, cands in embedding_similarities.items()
        }
        # Only match edges belonging to fragments with embedding candidates
        before = len(matchable)
        matchable = [d for d in matchable if d.fragment_id in embedding_candidate_set]
        print(f"[pipeline] Embedding filter: {before} → {len(matchable)} edges "
              f"(fragments with embedding candidates)")

        # Build a restricted descriptor index containing only candidate fragments
        # For each fragment, collect the union of all its candidates' fragment IDs
        all_candidate_frag_ids: set = set()
        for frag_id in embedding_candidate_set:
            all_candidate_frag_ids.add(frag_id)
            all_candidate_frag_ids.update(embedding_candidate_set[frag_id])

        restricted_descriptors = [
            d for d in all_descriptors
            if d.fragment_id in all_candidate_frag_ids
        ]
        print(f"[pipeline] Building restricted index: {len(restricted_descriptors)} edges "
              f"from {len(all_candidate_frag_ids)} fragments")
        index = build_descriptor_index(restricted_descriptors)
    else:
        print("[pipeline] Building full position+script+length index...")
        index = build_descriptor_index(all_descriptors)

    index_stats = index.stats()
    print(f"[pipeline] Index buckets: {index_stats}")

    matcher = ScaleAwareMatcher()
    position_computer = PositionComputer()

    all_match_rows: List[MatchRow] = []
    overlap_rejected = 0
    embedding_filtered = 0
    t0 = time.time()

    n_descs = len(matchable)
    for desc_idx, desc in enumerate(matchable, 1):
        if desc_idx % 500 == 0 or desc_idx == n_descs:
            print(f"  [match] {desc_idx}/{n_descs} edges processed "
                  f"({len(all_match_rows)} matches, {overlap_rejected} overlap-rejected, "
                  f"{embedding_filtered} embedding-filtered)")

        # Get the set of candidate fragment IDs for this edge's fragment
        candidate_frag_ids = (
            embedding_candidate_set.get(desc.fragment_id)
            if embedding_candidate_set is not None
            else None
        )

        matches = matcher.find_matches_indexed(desc, index, top_k=top_k * 3)  # fetch extra for filtering

        rank_counter = 0
        for score, details, cand in matches:
            # Skip candidates not in the embedding candidate set
            if candidate_frag_ids is not None and cand.fragment_id not in candidate_frag_ids:
                embedding_filtered += 1
                continue

            # Compute placement
            try:
                placement = position_computer.compute_placement(desc, cand)
            except Exception:
                continue

            # Overlap check
            frag_a = frag_lookup.get(desc.fragment_id)
            frag_b = frag_lookup.get(cand.fragment_id)
            if frag_a and frag_b:
                is_valid, overlap_ratio = check_overlap(
                    seg_coords_a=frag_a.segmentation_coords,
                    seg_coords_b=frag_b.segmentation_coords,
                    placement=placement,
                    pixels_per_unit_a=frag_a.pixels_per_unit,
                    pixels_per_unit_b=frag_b.pixels_per_unit,
                    image_width_a=frag_a.image_width,
                    image_height_a=frag_a.image_height,
                )
                if not is_valid:
                    overlap_rejected += 1
                    continue
            else:
                overlap_ratio = 0.0

            rank_counter += 1
            if rank_counter > top_k:
                break

            confidence = 1.0 / (1.0 + score)

            # Add embedding similarity to score details
            emb_sim = 0.0
            if use_embeddings and desc.fragment_id in embedding_similarities:
                emb_sim = embedding_similarities[desc.fragment_id].get(cand.fragment_id, 0.0)

            score_detail = {k: round(v, 6) for k, v in details.items()}
            score_detail["embedding_similarity"] = round(emb_sim, 6)
            score_detail["overlap_ratio"] = round(overlap_ratio, 6)

            all_match_rows.append(MatchRow(
                fragment_a_id=desc.fragment_id,
                edge_a_name=desc.edge_name,
                fragment_b_id=cand.fragment_id,
                edge_b_name=cand.edge_name,
                score=score,
                rank=rank_counter,
                confidence=confidence,
                score_details=json.dumps(score_detail),
                relative_x_cm=placement.relative_x_cm,
                relative_y_cm=placement.relative_y_cm,
                rotation_deg=placement.rotation_deg,
                algorithm_version=version,
            ))

    t_match = time.time() - t0
    print(f"[pipeline] Stage 2 complete: {len(all_match_rows)} matches in {t_match:.1f}s "
          f"({overlap_rejected} rejected by overlap check)")

    # ---------------------------------------------------------------
    # Stage 3: Visual join scoring (optional)
    # ---------------------------------------------------------------
    if use_join_scoring and cache_dir and all_match_rows:
        try:
            from .join_scorer import JoinScorer
            print("[pipeline] Stage 3: Visual join scoring...")
            t0 = time.time()

            scorer = JoinScorer(cache_dir=cache_dir)

            # Load line detection data for all involved fragments
            line_data_cache: Dict[str, Optional[str]] = {}

            def _get_line_data(frag_id: str) -> Optional[str]:
                if frag_id not in line_data_cache:
                    row = conn.execute(
                        "SELECT line_detection_data FROM fragments WHERE fragment_id = ?",
                        (frag_id,),
                    ).fetchone()
                    line_data_cache[frag_id] = row[0] if row else None
                return line_data_cache[frag_id]

            rescored = []
            for match in all_match_rows:
                frag_a = frag_lookup.get(match.fragment_a_id)
                frag_b = frag_lookup.get(match.fragment_b_id)
                if not frag_a or not frag_b:
                    rescored.append(match)
                    continue

                placement = PlacementResult(
                    relative_x_cm=match.relative_x_cm,
                    relative_y_cm=match.relative_y_cm,
                    rotation_deg=match.rotation_deg,
                    alignment_error=0.0,
                )

                existing_details = json.loads(match.score_details)
                emb_sim = existing_details.get("embedding_similarity", 0.0)

                join_score = scorer.score_join(
                    fragment_a_id=match.fragment_a_id,
                    fragment_b_id=match.fragment_b_id,
                    placement=placement,
                    geometric_score=match.score,
                    embedding_similarity=emb_sim,
                    edge_a_position=match.edge_a_name.split("_")[0],  # e.g. "left" from "left_seg2"
                    pixels_per_unit_a=frag_a.pixels_per_unit,
                    pixels_per_unit_b=frag_b.pixels_per_unit,
                    line_detection_data_a=_get_line_data(match.fragment_a_id),
                    line_detection_data_b=_get_line_data(match.fragment_b_id),
                )

                # Update score details with join scoring info
                existing_details.update(join_score.details)

                rescored.append(MatchRow(
                    fragment_a_id=match.fragment_a_id,
                    edge_a_name=match.edge_a_name,
                    fragment_b_id=match.fragment_b_id,
                    edge_b_name=match.edge_b_name,
                    score=join_score.final_score,
                    rank=match.rank,
                    confidence=1.0 / (1.0 + join_score.final_score),
                    score_details=json.dumps({k: round(v, 6) for k, v in existing_details.items()}),
                    relative_x_cm=match.relative_x_cm,
                    relative_y_cm=match.relative_y_cm,
                    rotation_deg=match.rotation_deg,
                    algorithm_version=version,
                ))

            # Re-rank within each query edge group
            from collections import defaultdict
            edge_groups: Dict[str, List[MatchRow]] = defaultdict(list)
            for m in rescored:
                key = f"{m.fragment_a_id}:{m.edge_a_name}"
                edge_groups[key].append(m)

            all_match_rows = []
            for key, group in edge_groups.items():
                group.sort(key=lambda m: m.score)
                for rank_idx, m in enumerate(group):
                    all_match_rows.append(MatchRow(
                        fragment_a_id=m.fragment_a_id,
                        edge_a_name=m.edge_a_name,
                        fragment_b_id=m.fragment_b_id,
                        edge_b_name=m.edge_b_name,
                        score=m.score,
                        rank=rank_idx + 1,
                        confidence=m.confidence,
                        score_details=m.score_details,
                        relative_x_cm=m.relative_x_cm,
                        relative_y_cm=m.relative_y_cm,
                        rotation_deg=m.rotation_deg,
                        algorithm_version=version,
                    ))

            t_join = time.time() - t0
            print(f"[pipeline] Stage 3 complete: rescored {len(all_match_rows)} matches in {t_join:.1f}s")
        except ImportError as e:
            print(f"[pipeline] Stage 3 skipped (missing dependency: {e})")
        except Exception as e:
            print(f"[pipeline] Stage 3 failed, using Stage 2 scores: {e}")

    # ---------------------------------------------------------------
    # Write to DB
    # ---------------------------------------------------------------
    clear_matches(conn, version)
    n = insert_matches(conn, all_match_rows)
    print(f"[pipeline] Wrote {n} rows to edge_matches table")

    conn.close()
    print("[pipeline] Done.")


def main():
    parser = argparse.ArgumentParser(description="Run 3-stage fragment matching pipeline")
    parser.add_argument("--db-path", required=True, help="Path to fragments.db")
    parser.add_argument("--cache-dir", default=None, help="Path to cache directory (for embeddings)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K matches per edge")
    parser.add_argument("--embedding-top-k", type=int, default=50, help="Candidates from embedding retrieval")
    parser.add_argument("--version", default="2.0", help="Algorithm version tag")
    parser.add_argument("--limit", type=int, default=0, help="Limit edges to match (0 = all)")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip Stage 1 (embedding retrieval)")
    parser.add_argument("--no-join-scoring", action="store_true", help="Skip Stage 3 (visual join scoring)")
    args = parser.parse_args()

    run_pipeline(
        db_path=args.db_path,
        top_k=args.top_k,
        version=args.version,
        limit=args.limit,
        cache_dir=args.cache_dir,
        use_embeddings=not args.no_embeddings,
        embedding_top_k=args.embedding_top_k,
        use_join_scoring=not args.no_join_scoring,
    )


if __name__ == "__main__":
    main()
