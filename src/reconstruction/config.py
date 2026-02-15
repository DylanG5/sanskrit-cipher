"""Configuration constants for the reconstruction pipeline."""

# Scoring weights for scale-aware matching
WEIGHTS = {
    'shape_score': 0.30,           # Normalized point RMSE (scale-invariant)
    'fourier_score': 0.15,         # Fourier descriptor distance
    'scale_length_penalty': 0.25,  # Real-world length difference
    'scale_shape_score': 0.15,     # Real-world point RMSE, unnormalized
    'histogram_distance': 0.15,    # Curvature histogram JS-divergence
}

# Candidate filter thresholds
LINE_COUNT_TOLERANCE = 2           # +-2 lines
EDGE_LENGTH_TOLERANCE = 0.30       # 30% relative difference allowed
# Position compatibility: horizontal edges match horizontal, vertical match vertical
COMPATIBLE_POSITIONS: dict[str, list[str]] = {
    'left': ['left', 'right'],
    'right': ['left', 'right'],
    'top': ['top', 'bottom'],
    'bottom': ['top', 'bottom'],
}

# Contour segmentation parameters
CURVATURE_WINDOW = 11
CORNER_THRESHOLD = 0.4
MIN_SEGMENT_LENGTH = 50
STRAIGHTNESS_THRESHOLD = 1.10
MIN_CORNER_SPACING_RATIO = 0.03

# Feature extraction
NUM_RESAMPLE_POINTS = 80
NUM_FOURIER_COEFFICIENTS = 16
CURVATURE_HISTOGRAM_BINS = 8

# Alignment / position
SEPARATION_SCALE_FACTOR = 0.5   # cm of gap between aligned edges
MIN_EDGE_LENGTH_CM = 0.3        # Minimum real-world edge length to consider

# Grid scale default (pixels per cm on canvas)
DEFAULT_GRID_SCALE = 25
