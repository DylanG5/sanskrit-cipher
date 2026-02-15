"""
Scale-aware edge matching: candidate filtering and multi-criteria scoring.

Replaces EdgeIndex / EdgeMatcher from v2 with scale-aware logic.
"""

import bisect
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import entropy as kl_divergence

from .config import (
    COMPATIBLE_POSITIONS,
    EDGE_LENGTH_TOLERANCE,
    LINE_COUNT_TOLERANCE,
    NUM_RESAMPLE_POINTS,
    WEIGHTS,
)
from .edge_extraction import ScaleAwareDescriptor, _resample_points


# ---------------------------------------------------------------------------
# Candidate filter (fast rejection before expensive scoring)
# ---------------------------------------------------------------------------

class CandidateFilter:
    """Quickly reject incompatible edge pairs."""

    @staticmethod
    def is_compatible(query: ScaleAwareDescriptor, candidate: ScaleAwareDescriptor) -> bool:
        # 1. Not the same fragment (including recto/verso pairs: "xxxA" ↔ "xxxB")
        q_base = query.fragment_id.rstrip('AB')
        c_base = candidate.fragment_id.rstrip('AB')
        if q_base == c_base:
            return False

        # 2. Both must be tear edges
        if query.edge_type != 'tear' or candidate.edge_type != 'tear':
            return False

        # 3. Compatible positions: horizontal↔horizontal, vertical↔vertical
        allowed = COMPATIBLE_POSITIONS.get(query.position_label)
        if allowed is None or candidate.position_label not in allowed:
            return False

        # 4. Same script type required (skip if either is unknown)
        if not query.script_type or not candidate.script_type:
            return False
        if query.script_type != candidate.script_type:
            return False

        # 5. Similar line count (±tolerance)
        if (query.line_count is not None and candidate.line_count is not None):
            if abs(query.line_count - candidate.line_count) > LINE_COUNT_TOLERANCE:
                return False

        # 6. Similar real-world edge length (within tolerance)
        if query.length_cm > 0 and candidate.length_cm > 0:
            ratio = abs(query.length_cm - candidate.length_cm) / max(query.length_cm, candidate.length_cm)
            if ratio > EDGE_LENGTH_TOLERANCE:
                return False

        return True


# ---------------------------------------------------------------------------
# Scale-aware scorer
# ---------------------------------------------------------------------------

class ScaleAwareMatcher:
    """Score edge pairs using normalised + real-world features."""

    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = weights or WEIGHTS

    def score(self, a: ScaleAwareDescriptor, b: ScaleAwareDescriptor) -> Tuple[float, Dict[str, float]]:
        """
        Lower is better.  Returns (composite_score, detail_dict).
        """
        details: Dict[str, float] = {}

        # 1. Normalised-point RMSE (scale-invariant shape)
        pts_a = a.normalized_points
        pts_b = b.normalized_points
        # Flip b for complementary matching
        pts_b_flipped = pts_b[::-1]
        rmse_normal = float(np.sqrt(np.mean((pts_a - pts_b_flipped) ** 2)))
        details['shape_score'] = rmse_normal

        # 2. Fourier descriptor distance
        fd_dist = float(np.linalg.norm(a.fourier_descriptors - b.fourier_descriptors))
        details['fourier_score'] = fd_dist

        # 3. Real-world length penalty
        if a.length_cm > 0 and b.length_cm > 0:
            length_penalty = abs(a.length_cm - b.length_cm) / max(a.length_cm, b.length_cm)
        else:
            length_penalty = 1.0
        details['scale_length_penalty'] = length_penalty

        # 4. Real-world (cm-space) point RMSE
        cm_a = _resample_points(a.points_cm, NUM_RESAMPLE_POINTS)
        cm_b = _resample_points(b.points_cm, NUM_RESAMPLE_POINTS)
        if cm_a is not None and cm_b is not None:
            # Centre both in cm-space
            cm_a_c = cm_a - cm_a.mean(axis=0, keepdims=True)
            cm_b_c = cm_b - cm_b.mean(axis=0, keepdims=True)
            cm_b_c_flipped = cm_b_c[::-1]
            scale_rmse = float(np.sqrt(np.mean((cm_a_c - cm_b_c_flipped) ** 2)))
        else:
            scale_rmse = 1.0
        details['scale_shape_score'] = scale_rmse

        # 5. Curvature histogram JS-divergence
        eps = 1e-10
        p = a.curvature_histogram + eps
        q = b.curvature_histogram + eps
        p = p / p.sum()
        q = q / q.sum()
        m = 0.5 * (p + q)
        js = 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))
        details['histogram_distance'] = float(js)

        # Weighted composite
        composite = sum(self.weights[k] * details[k] for k in self.weights)
        return composite, details

    def find_matches(
        self,
        query: ScaleAwareDescriptor,
        candidates: List[ScaleAwareDescriptor],
        top_k: int = 5,
    ) -> List[Tuple[float, Dict[str, float], ScaleAwareDescriptor]]:
        """
        Filter + score + rank candidates for *query*.
        Returns list of (score, details, candidate) sorted best-first.
        """
        candidate_filter = CandidateFilter()
        scored: List[Tuple[float, Dict[str, float], ScaleAwareDescriptor]] = []
        for cand in candidates:
            if not candidate_filter.is_compatible(query, cand):
                continue
            score, details = self.score(query, cand)
            scored.append((score, details, cand))
        scored.sort(key=lambda x: x[0])
        return scored[:top_k]

    def find_matches_indexed(
        self,
        query: ScaleAwareDescriptor,
        index: 'DescriptorIndex',
        top_k: int = 5,
    ) -> List[Tuple[float, Dict[str, float], ScaleAwareDescriptor]]:
        """
        Uses a pre-built index keyed by (position_group, script_type),
        with binary-search on length_cm to only score edges within ±30%.
        """
        scored: List[Tuple[float, Dict[str, float], ScaleAwareDescriptor]] = []

        allowed_positions = COMPATIBLE_POSITIONS.get(query.position_label, [])
        script = query.script_type or ''
        if not script:
            return []

        q_len = query.length_cm
        if q_len > 0:
            lo = q_len * (1.0 - EDGE_LENGTH_TOLERANCE)
            hi = q_len * (1.0 + EDGE_LENGTH_TOLERANCE)
        else:
            lo, hi = 0.0, float('inf')

        q_base = query.fragment_id.rstrip('AB')

        for pos in allowed_positions:
            key = f"{pos}:{script}"
            bucket = index.get_bucket(key)
            if bucket is None:
                continue

            descs, lengths = bucket

            # Binary search for the length window
            i_lo = bisect.bisect_left(lengths, lo)
            i_hi = bisect.bisect_right(lengths, hi)

            for i in range(i_lo, i_hi):
                cand = descs[i]

                # Inline the remaining cheap checks (skip full is_compatible)
                c_base = cand.fragment_id.rstrip('AB')
                if q_base == c_base:
                    continue
                if (query.line_count is not None and cand.line_count is not None
                        and abs(query.line_count - cand.line_count) > LINE_COUNT_TOLERANCE):
                    continue

                score, details = self.score(query, cand)
                scored.append((score, details, cand))

        scored.sort(key=lambda x: x[0])
        return scored[:top_k]


class DescriptorIndex:
    """
    Pre-built index keyed by "position:script_type".
    Each bucket is sorted by length_cm for binary-search filtering.
    """

    def __init__(self, descriptors: List[ScaleAwareDescriptor]):
        buckets: Dict[str, List[ScaleAwareDescriptor]] = defaultdict(list)
        for desc in descriptors:
            if desc.edge_type != 'tear' or not desc.script_type:
                continue
            key = f"{desc.position_label}:{desc.script_type}"
            buckets[key].append(desc)

        # Sort each bucket by length_cm and pre-extract the lengths array
        self._buckets: Dict[str, Tuple[List[ScaleAwareDescriptor], List[float]]] = {}
        for key, descs in buckets.items():
            descs.sort(key=lambda d: d.length_cm)
            lengths = [d.length_cm for d in descs]
            self._buckets[key] = (descs, lengths)

    def get_bucket(self, key: str):
        return self._buckets.get(key)

    def stats(self) -> Dict[str, int]:
        return {k: len(v[0]) for k, v in self._buckets.items()}


def build_descriptor_index(
    descriptors: List[ScaleAwareDescriptor],
) -> DescriptorIndex:
    """Build an index sorted by length_cm within each (position, script_type) bucket."""
    return DescriptorIndex(descriptors)
