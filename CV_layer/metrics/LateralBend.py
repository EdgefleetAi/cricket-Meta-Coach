# metrics/lateral_bend_deg.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging

mp_pose = mp.solutions.pose

# Utils import
from utils import get_point, angle_between

def extract_lateral_bend_deg(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=5,
    burn_in_frames=15,
    ref_frame_start=15,
    ref_frame_end=30,
    max_bend_cap=45.0
):
    """
    Lateral Bend (Trunk Side Bend) - Production Grade (Finalized):
    - Spine vector: Mid-shoulder to Mid-hip
    - Angle deviation from vertical (0° = straight, higher = side bend)
    - CSV: frame, LateralBend_deg
    - Saves to session_folder + "LateralBend_deg.csv"
    - Best for front-on or semi-front view
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Lateral Bend extraction for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_bend = []
    ref_vertical_angle = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting LateralBend_deg"):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Frame read failed at {frame_idx}")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                logger.debug(f"No landmarks in frame {frame_idx}")
                raw_bend.append(np.nan)
                continue

            lm = res.pose_landmarks.landmark

            ls, vLS = get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            rs, vRS = get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
            lh, vLH = get_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
            rh, vRH = get_point(lm, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)

            if min(vLS, vRS, vLH, vRH) < visibility_thresh:
                logger.debug(f"Low visibility in frame {frame_idx}")
                raw_bend.append(np.nan)
                continue

            mid_shoulder = (ls + rs) / 2.0
            mid_hip = (lh + rh) / 2.0

            # Spine vector (mid-shoulder to mid-hip, downward)
            spine_vec = mid_hip - mid_shoulder

            # Vertical reference: downward y-axis
            vertical_vec = np.array([0, 1.0], dtype=np.float32)

            # Angle between spine and vertical
            bend_deg = angle_between(spine_vec, vertical_vec)

            # Lateral bend as acute deviation (0–90°)
            bend_deg = min(bend_deg, 180 - bend_deg)

            # Burn-in skip
            if frame_idx < burn_in_frames:
                raw_bend.append(0.0)
                continue

            # Set reference bend from stable stance frames
            if frame_idx == ref_frame_end and ref_vertical_angle is None:
                valid_angles = [a for a in raw_bend[ref_frame_start:ref_frame_end+1] if not np.isnan(a)]
                ref_vertical_angle = np.mean(valid_angles) if valid_angles else bend_deg
                logger.info(f"Reference vertical bend set: {ref_vertical_angle:.2f}° (frames {ref_frame_start}-{ref_frame_end})")

            # Deviation from reference (relative lateral bend)
            if ref_vertical_angle is not None:
                relative_bend = abs(bend_deg - ref_vertical_angle)
            else:
                relative_bend = bend_deg

            relative_bend = min(relative_bend, max_bend_cap)

            raw_bend.append(relative_bend)

    cap.release()
    logger.info("Processing completed")

    # Smoothing
    series = pd.Series(raw_bend)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    # NaN fill
    series = series.ffill().bfill()
    logger.info(f"NaN fill applied, remaining NaNs: {series.isna().sum()}")

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "LateralBend_deg": series.round(4)
    })

    csv_path = os.path.join(session_folder, "LateralBend_deg.csv")
    df.to_csv(csv_path, index=False)
    avg_bend = df['LateralBend_deg'].mean()
    logger.info(f"CSV saved: {csv_path} | Avg Lateral Bend: {avg_bend:.2f}°")

    # Phase summary
    stance_avg = df.iloc[burn_in_frames:ref_frame_end]['LateralBend_deg'].mean()
    impact_avg = df.iloc[ref_frame_end:]['LateralBend_deg'].mean()
    logger.info(f"Stance phase avg: {stance_avg:.2f}° | Impact/follow avg: {impact_avg:.2f}°")

    print(f"✅ Lateral Bend CSV saved: {csv_path}")
    return df