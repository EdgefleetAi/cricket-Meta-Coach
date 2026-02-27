import os
import logging
from pathlib import Path


# Base directory (project root)
BASE_DIR = Path(__file__).parent

# Paths
VIDEO_DIR = BASE_DIR / "videos"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

LOG_LEVEL = logging.INFO

# Ensure folders exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# View-specific metrics
SIDE_VIEW_METRICS = [
    "HeadVerticalDrift_deg","HeadAngle_deg", "HeadStabilityIndex","FollowThroughAngle_deg","HeadStillnessAfterImpact_deg",
    "FrontFootStrideLength_cm", "BackFootAnchorStabilityIndex","TrunkLean_deg","SpineAngle_deg",
    "WeightTransferEfficiency_pct", "StepDirectionDeviation_deg",
    "FootContactTime_ms"
]

FRONT_VIEW_METRICS = [
    "HeadAlignmentScore","HeadLateralDeviation_deg",
    "TorsoRotation_deg", "TorsoStabilityIndex", "PelvisAlignmentScore",
    "LateralBend_deg"
]

# Common params
VISIBILITY_THRESH = 0.5
SMOOTH_WINDOW = 5
HANDEDNESS = 'right' 
k = 12.0 # sensitivity (higher = stricter on movement)

# Logging
LOG_LEVEL = 'INFO'