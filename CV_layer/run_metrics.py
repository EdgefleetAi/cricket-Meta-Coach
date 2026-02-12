# run_metrics.py
import os
import sys
import time
import logging
from pathlib import Path
import importlib.util


# Force UTF-8 for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())


# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config (create config.py if not exists)
from config import (
    VIDEO_DIR, OUTPUT_DIR, LOG_DIR,
    SIDE_VIEW_METRICS, FRONT_VIEW_METRICS
)

# Setup logging (production style)
log_file = LOG_DIR / "run_metrics.log"
# Logging setup (after path setup)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "run_metrics.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

# All available metrics (name → function)
METRIC_REGISTRY = {}

def register_metric(name, module_name, function_name):
    """Dynamically import metric function"""
    try:
        spec = importlib.util.spec_from_file_location(
            name, PROJECT_ROOT / "metrics" / f"{module_name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, function_name)
        METRIC_REGISTRY[name] = func
        logger.debug(f"Registered metric: {name}")
    except Exception as e:
        logger.error(f"Failed to register {name}: {e}")

# Register all metrics (add more here)
#register_metric("HeadVerticalDrift_deg", "HeadVerticalDrift", "extract_head_vertical_drift_deg")
#register_metric("HeadAngle_deg", "HeadAngle", "extract_head_angle_deg")
#register_metric("HeadStabilityIndex", "HeadStabilityIndex", "extract_head_stability_index")
register_metric("HeadLateralDeviation_deg", "HeadLateralDeviation", "extract_head_lateral_deviation_deg")
#register_metric("HeadAlignmentScore", "HeadAlignmentScore", "extract_head_alignment_score")
register_metric("LateralBend_deg", "LateralBend", "extract_lateral_bend_deg")
register_metric("TorsoRotation_deg", "TorsoRotation", "extract_torso_rotation_deg")
register_metric("TorsoStabilityIndex", "TorsoStability", "extract_torso_stability_index")
# Add more metrics the same way

def detect_view(video_name: str) -> str:
    name = video_name.lower()
    if any(x in name for x in ['side', 'side_on', 'lateral', 'profile']):
        return 'side'
    elif any(x in name for x in ['front', 'front_on', 'face', 'direct']):
        return 'front'
    return 'unknown'

def run_metrics_for_video(video_path: Path, output_folder: Path):
    """Run relevant metrics for one video"""
    start_time = time.time()
    video_name = video_path.name
    view = detect_view(video_name)

    logger.info(f"{'='*80}")
    logger.info(f"Processing: {video_name}")
    logger.info(f"View detected: {view.upper()}")
    logger.info(f"Output: {output_folder}")
    logger.info(f"{'='*80}")

    output_folder.mkdir(parents=True, exist_ok=True)

    if view == 'side':
        metrics_to_run = SIDE_VIEW_METRICS
    elif view == 'front':
        metrics_to_run = FRONT_VIEW_METRICS
    else:
        logger.warning("View unknown → running all metrics (may be slow/inaccurate)")
        metrics_to_run = list(METRIC_REGISTRY.keys())

    success_count = 0
    fail_count = 0

    for metric_name in metrics_to_run:
        if metric_name not in METRIC_REGISTRY:
            logger.warning(f"Metric {metric_name} not registered – skipping")
            continue

        func = METRIC_REGISTRY[metric_name]
        csv_path = output_folder / f"{metric_name}.csv"

        try:
            logger.info(f"→ Running {metric_name}...")
            result = func(str(video_path), str(output_folder), logger=logger)
            if result is not None:
                logger.info(f"  ✅ {metric_name} completed → {csv_path.name}")
                success_count += 1
            else:
                logger.warning(f"  ⚠️ {metric_name} returned None")
                fail_count += 1
        except Exception as e:
            logger.error(f"  ❌ {metric_name} failed: {str(e)}", exc_info=True)
            fail_count += 1

    elapsed = time.time() - start_time
    logger.info(f"Completed {video_name} in {elapsed:.1f}s | Success: {success_count} | Failed: {fail_count}")

def main():
    start_total = time.time()
    logger.info("=== Starting Batch Metrics Processing ===")

    if not VIDEO_DIR.exists():
        logger.error(f"Videos directory not found: {VIDEO_DIR}")
        return

    video_files = []
    for root, _, files in os.walk(VIDEO_DIR):
        for f in files:
            if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                video_files.append(Path(root) / f)

    if not video_files:
        logger.error("No video files found")
        return

    logger.info(f"Found {len(video_files)} video(s)")

    for video_path in sorted(video_files):
        session_id = video_path.stem
        output_folder = OUTPUT_DIR / session_id
        run_metrics_for_video(video_path, output_folder)

    total_time = time.time() - start_total
    logger.info(f"=== All Done in {total_time:.1f} seconds ===")
    logger.info(f"Total CSVs generated in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()