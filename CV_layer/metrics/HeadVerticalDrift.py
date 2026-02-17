# metrics/head_vertical_drift.py

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

def extract_head_vertical_drift_deg(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=5,
    max_drift_cap=60.0
):
    """
    Head Vertical Drift (2D pixel only) - Production Grade:
    - Head vector: Nose -> MidShoulder
    - Vertical reference: upward (negative y in image coords)
    - Drift angle = deviation from vertical (0° = head straight, higher = forward/back tilt)
    - CSV: frame, HeadVerticalDrift_deg
    - Saves to session_folder + "HeadVerticalDrift_deg.csv"
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Head Vertical Drift extraction for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_drift = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting HeadVerticalDrift_deg"):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Frame read failed at {frame_idx}")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                logger.debug(f"No landmarks in frame {frame_idx}")
                raw_drift.append(np.nan)
                continue

            lm = res.pose_landmarks.landmark

            nose, vN = get_point(lm, mp_pose.PoseLandmark.NOSE.value, w, h)
            ls, vLS = get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            rs, vRS = get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)

            if min(vN, vLS, vRS) < visibility_thresh:
                logger.debug(f"Low visibility in frame {frame_idx}")
                raw_drift.append(np.nan)
                continue

            N = nose
            MS = (ls + rs) / 2.0

            head_vec = N - MS
            vertical_ref = np.array([0, -1])  # Upward in image (negative y)

            included_angle = angle_between(head_vec, vertical_ref)
            drift_deg = min(included_angle, 180.0 - included_angle)  # Acute deviation

            # Outlier cap
            drift_deg = min(drift_deg, max_drift_cap)

            raw_drift.append(drift_deg)

    cap.release()
    logger.info("Processing completed")

    # Smoothing
    series = pd.Series(raw_drift)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    # NaN fill
    series = series.ffill().bfill()
    logger.info(f"NaN fill applied, remaining NaNs: {series.isna().sum()}")

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "HeadVerticalDrift_deg": series.round(4)
    })

    csv_path = os.path.join(session_folder, "HeadVerticalDrift_deg.csv")
    df.to_csv(csv_path, index=False)
    avg_drift = df['HeadVerticalDrift_deg'].mean()
    logger.info(f"CSV saved: {csv_path} | Avg Head Vertical Drift: {avg_drift:.2f}°")

    print(f"✅ Head Vertical Drift CSV saved: {csv_path}")
    return df