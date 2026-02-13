# metrics/spine_angle_deg.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging

mp_pose = mp.solutions.pose

from utils import get_point, angle_between

def extract_spine_angle_deg(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=5,
    burn_in_fraction=0.25,
    max_angle_cap=60.0
):
    """
    Spine Angle (Maintain Spine Angle) - Production Grade:
    - Torso vector: Mid-hip → Mid-shoulder (upward)
    - Angle from vertical upward (0° = perfect upright spine, higher = tilt/lean)
    - Dynamic burn-in skip
    - Visibility carry-forward
    - CSV: frame, SpineAngle_deg (NaN early)
    - Saves to session_folder + "SpineAngle_deg.csv"
    - Best for side-on view
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Spine Angle for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_angle = [np.nan] * total_frames
    last_valid_torso_vec = None

    burn_in_frames = max(15, int(total_frames * burn_in_fraction))
    logger.info(f"Dynamic burn-in: {burn_in_frames} frames ({burn_in_frames/total_frames*100:.1f}%)")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting SpineAngle_deg"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                raw_angle[frame_idx] = np.nan
                continue

            lm = res.pose_landmarks.landmark

            ls, vLS = get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            rs, vRS = get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
            lh, vLH = get_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
            rh, vRH = get_point(lm, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)

            min_vis = min(vLS, vRS, vLH, vRH)
            if min_vis < visibility_thresh:
                if last_valid_torso_vec is not None:
                    torso_vec = last_valid_torso_vec
                else:
                    raw_angle[frame_idx] = np.nan
                    continue

            mid_shoulder = (ls + rs) / 2.0
            mid_hip = (lh + rh) / 2.0

            # Torso vector: mid-hip → mid-shoulder (upward in image)
            torso_vec = mid_shoulder - mid_hip
            last_valid_torso_vec = torso_vec

            # Vertical upward reference (y decreases upward)
            vertical_up = np.array([0.0, -1.0])

            angle = angle_between(torso_vec, vertical_up)
            angle = min(angle, max_angle_cap)

            if frame_idx < burn_in_frames:
                raw_angle[frame_idx] = np.nan
            else:
                raw_angle[frame_idx] = angle

    cap.release()

    series = pd.Series(raw_angle)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    series = series.ffill().bfill()

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "SpineAngle_deg": series.round(4)
    })

    csv_path = os.path.join(session_folder, "SpineAngle_deg.csv")

    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except:
            pass

    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV saved: {csv_path}")
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        return df

    avg_angle = df['SpineAngle_deg'].mean(skipna=True)
    max_angle = df['SpineAngle_deg'].max(skipna=True)

    approx_impact = int(total_frames * 0.6)
    pre = df['SpineAngle_deg'].iloc[:approx_impact].mean(skipna=True)
    post = df['SpineAngle_deg'].iloc[approx_impact:].mean(skipna=True)

    logger.info(f"Avg Spine Angle: {avg_angle:.2f}°")
    logger.info(f"Max Spine Angle: {max_angle:.2f}°")
    logger.info(f"Pre-impact Avg (approx): {pre:.2f}°")
    logger.info(f"Post-impact Avg (approx): {post:.2f}°")

    print(f"✅ Spine Angle CSV saved: {csv_path}")
    print(f"  - Avg Angle: {avg_angle:.2f}° (lower = better maintained spine)")
    print(f"  - Max Angle: {max_angle:.2f}°")

    return df