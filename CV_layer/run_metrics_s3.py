# run_metrics.py (fixed version - view from S3 key, not temp file)

import os
import sys
import time
import logging
from pathlib import Path
import importlib.util
import boto3
import tempfile

# Force UTF-8 for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config
from config import (
    OUTPUT_DIR, LOG_DIR,
    SIDE_VIEW_METRICS, FRONT_VIEW_METRICS
)

# S3 setup
BUCKET = "metacoach-videos-s3-bucket"
s3 = boto3.client('s3')

# Logging setup
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

# Metric registry (your registrations)
METRIC_REGISTRY = {}

def register_metric(name, module_name, function_name):
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

# Register all metrics (same as yours)
register_metric("HeadLateralDeviation_deg", "HeadLateralDeviation", "extract_head_lateral_deviation_deg")
register_metric("HeadVerticalDrift_deg", "HeadVerticalDrift", "extract_head_vertical_drift_deg")
register_metric("HeadAngle_deg", "HeadAngle", "extract_head_angle_deg")
register_metric("HeadStabilityIndex", "HeadStabilityIndex", "extract_head_stability_index")
register_metric("HeadAlignmentScore", "HeadAlignmentScore", "extract_head_alignment_score")
register_metric("LateralBend_deg", "LateralBend", "extract_lateral_bend_deg")
register_metric("TorsoRotation_deg", "TorsoRotation", "extract_torso_rotation_deg")
register_metric("TorsoStabilityIndex", "TorsoStability", "extract_torso_stability_index")
register_metric("PelvisAlignmentScore", "PelvisAlignmentScore", "extract_pelvis_alignment_score")
register_metric("FollowThroughAngle_deg", "FollowThroughAngle", "extract_followthrough_angle_deg")
register_metric("HeadStillnessAfterImpact_deg", "HeadStillnessAfterImpact", "extract_head_stillness_after_impact_deg")
register_metric("WeightTransferEfficiency_pct", "WeightTransferEfficiency", "extract_weight_transfer_efficiency_pct")
register_metric("StepDirectionDeviation_deg", "StepDirectionDeviation", "extract_step_direction_deviation_deg")
register_metric("FrontFootStrideLength_cm", "FrontFootStrideLength", "extract_frontfoot_stride_length_cm")
register_metric("BackFootAnchorStabilityIndex", "BackFootAnchorStability", "extract_backfoot_anchor_stability_index")
register_metric("TrunkLean_deg", "TrunkLean", "extract_trunk_lean_deg")
register_metric("SpineAngle_deg", "SpineAngle", "extract_spine_angle_deg")

def detect_view(video_name: str) -> str:
    name = video_name.lower().replace('.mov', '').replace('.mp4', '').replace('.avi', '').replace('.mkv', '')
    if any(x in name for x in ['side', 'side_on', 'lateral', 'profile']):
        return 'side'
    elif any(x in name for x in ['front', 'front_on', 'face', 'direct']):
        return 'front'
    return 'unknown'

def run_metrics_for_video(video_path: Path, output_folder: Path, original_video_name: str):
    """Run relevant metrics – view from original S3 name"""
    start_time = time.time()
    view = detect_view(original_video_name)  # ← key change: original name use

    logger.info(f"{'='*80}")
    logger.info(f"Processing: {original_video_name} (local: {video_path.name})")
    logger.info(f"View detected: {view.upper()}")
    logger.info(f"Output: {output_folder}")
    logger.info(f"{'='*80}")

    output_folder.mkdir(parents=True, exist_ok=True)

    if view == 'side':
        metrics_to_run = SIDE_VIEW_METRICS
        logger.info(f"Running SIDE view metrics ({len(metrics_to_run)})")
    elif view == 'front':
        metrics_to_run = FRONT_VIEW_METRICS
        logger.info(f"Running FRONT view metrics ({len(metrics_to_run)})")
    else:
        logger.warning("View unknown → skipping this video")
        return

    # Only registered metrics
    metrics_to_run = [m for m in metrics_to_run if m in METRIC_REGISTRY]
    if not metrics_to_run:
        logger.error(f"No valid metrics for view '{view}' – check config.py")
        return

    success_count = 0
    fail_count = 0

    for metric_name in metrics_to_run:
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
    logger.info(f"Completed {original_video_name} in {elapsed:.1f}s | Success: {success_count} | Failed: {fail_count}")

def upload_csvs_to_s3(local_folder: Path, s3_prefix: str):
    for root, _, files in os.walk(local_folder):
        for file in files:
            if file.endswith(".csv"):
                local_path = os.path.join(root, file)
                s3_key = f"{s3_prefix}/{file}"
                try:
                    s3.upload_file(local_path, BUCKET, s3_key)
                    logger.info(f"Uploaded CSV: {s3_key}")
                    os.remove(local_path)
                except Exception as e:
                    logger.error(f"CSV upload failed {file}: {e}")

def main():
    start_total = time.time()
    logger.info("=== Starting Batch Metrics Processing (S3 Test Mode) ===")

    test_s3_keys = [
        "videos/raw/test/front_1.MOV",
        "videos/raw/test/front_2.MOV",
        "videos/raw/test/front_3.MOV",
        "videos/raw/test/side_1.MOV",
        "videos/raw/test/side_2.MOV"
    ]

    for s3_key in test_s3_keys:
        video_name = Path(s3_key).stem  # front_1
        logger.info(f"Processing S3 video: {s3_key}")

        temp_file = tempfile.NamedTemporaryFile(suffix=".MOV", delete=False)
        local_temp_path = temp_file.name
        temp_file.close()

        try:
            logger.info(f"Downloading from S3: {s3_key}")
            s3.download_file(BUCKET, s3_key, local_temp_path)

            output_folder = OUTPUT_DIR / "test" / video_name
            run_metrics_for_video(Path(local_temp_path), output_folder, video_name)  # ← pass original name

            upload_csvs_to_s3(output_folder, f"csvs/test/{video_name}")

        except Exception as e:
            logger.error(f"S3 processing failed for {s3_key}: {e}")
        finally:
            try:
                os.unlink(local_temp_path)
                logger.info(f"Temp file cleaned: {local_temp_path}")
            except PermissionError:
                logger.warning(f"Temp file locked: {local_temp_path} – close any app using it")

    total_time = time.time() - start_total
    logger.info(f"=== S3 Test Processing Done in {total_time:.1f} seconds ===")

if __name__ == "__main__":
    main()