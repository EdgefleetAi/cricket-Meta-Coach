# metrics/weight_transfer_efficiency_pct.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging

mp_pose = mp.solutions.pose

# Utils import
from utils import get_point, detect_handedness

def extract_weight_transfer_efficiency_pct(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=5,
    burn_in_fraction=0.25,
    min_stride_px=50.0,
    handedness_check_frames=30
):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Weight Transfer Efficiency for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    landmarks_list = []  # Store pose_landmarks.landmark (RepeatedCompositeContainer)
    hip_positions = []
    back_foot_x = None
    max_stride = min_stride_px
    valid_count = 0
    vis_fail = 0

    LH_ID = mp_pose.PoseLandmark.LEFT_HIP.value
    RH_ID = mp_pose.PoseLandmark.RIGHT_HIP.value
    LA_ID = mp_pose.PoseLandmark.LEFT_ANKLE.value
    RA_ID = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    burn_in_frames = max(15, int(total_frames * burn_in_fraction))
    logger.info(f"Dynamic burn-in: {burn_in_frames} frames ({burn_in_frames/total_frames*100:.1f}%)")

    handedness = 'right'  # Default safe fallback

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting WeightTransferEfficiency_pct"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                hip_positions.append(None)
                landmarks_list.append(None)
                continue

            lm = res.pose_landmarks.landmark  # landmark container

            lh, vLH = get_point(lm, LH_ID, w, h)
            rh, vRH = get_point(lm, RH_ID, w, h)
            la, vLA = get_point(lm, LA_ID, w, h)
            ra, vRA = get_point(lm, RA_ID, w, h)

            min_vis = min(vLH, vRH, vLA, vRA)
            if min_vis < visibility_thresh:
                vis_fail += 1
                hip_positions.append(None)
                landmarks_list.append(None)
                continue

            mid_hip = (lh + rh) / 2.0
            hip_positions.append(mid_hip)
            landmarks_list.append(res.pose_landmarks)  # Full pose_landmarks object for detect_handedness

            valid_count += 1

            # Handedness detect once
            if handedness == 'right' and valid_count >= handedness_check_frames:  # Only detect if default
                try:
                    handedness = detect_handedness(landmarks_list, w, h, num_frames=handedness_check_frames)
                    logger.info(f"Handedness detected after {valid_count} valid frames: {handedness}")
                except Exception as e:
                    logger.warning(f"Handedness detection failed: {e} - defaulting to 'right'")
                    handedness = 'right'

            # Back foot reference
            if frame_idx >= burn_in_frames and back_foot_x is None:
                if vLA > visibility_thresh and vRA > visibility_thresh:
                    if handedness == 'right':
                        back_foot_x = ra[0]
                    else:
                        back_foot_x = la[0]
                    logger.info(f"Back foot set at frame {frame_idx} ({handedness}): x = {back_foot_x:.2f}")

            if la is not None and ra is not None:
                current_stride = abs(la[0] - ra[0])
                max_stride = max(max_stride, current_stride)

    cap.release()

    if back_foot_x is None:
        logger.warning("No back foot detected – fallback x=0")
        back_foot_x = 0.0

    if max_stride < min_stride_px:
        max_stride = 100.0
        logger.warning(f"Max stride too small – fallback {max_stride}")

    raw_efficiency = [np.nan] * total_frames
    cumulative_transfer = 0.0

    for frame_idx in range(total_frames):
        if hip_positions[frame_idx] is None:
            continue

        hip_x = hip_positions[frame_idx][0]
        transfer_dist = hip_x - back_foot_x
        cumulative_transfer = max(cumulative_transfer, transfer_dist)

        efficiency = (cumulative_transfer / max_stride) * 100 if max_stride > 0 else 0.0
        efficiency = np.clip(efficiency, 0.0, 100.0)
        raw_efficiency[frame_idx] = efficiency

    series = pd.Series(raw_efficiency)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    series = series.ffill().bfill()

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "WeightTransferEfficiency_pct": series.round(2)
    })

    csv_path = os.path.join(session_folder, "WeightTransferEfficiency_pct.csv")

    # Permission fix: try to remove if exists
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
            logger.info(f"Removed existing locked CSV: {csv_path}")
        except Exception as e:
            logger.warning(f"Could not remove existing CSV: {e}")

    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV saved: {csv_path}")
    except PermissionError as e:
        logger.error(f"Permission denied on CSV write: {e}")
        logger.error("Close Excel or any app using the file and rerun")
        return df

    avg_eff = df['WeightTransferEfficiency_pct'].mean(skipna=True)
    max_eff = df['WeightTransferEfficiency_pct'].max(skipna=True)

    logger.info(f"Avg Weight Transfer Efficiency: {avg_eff:.2f}%")
    logger.info(f"Max Weight Transfer Efficiency: {max_eff:.2f}%")

    print(f"✅ Weight Transfer Efficiency CSV saved: {csv_path}")
    print(f"  - Avg Efficiency: {avg_eff:.2f}%")
    print(f"  - Max Efficiency: {max_eff:.2f}%")

    return df