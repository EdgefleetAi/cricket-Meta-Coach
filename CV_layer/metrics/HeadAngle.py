# metrics/head_angle.py

import logging
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging

# utils & config import 
from utils import get_point, angle_between, detect_handedness
from config import VISIBILITY_THRESH, SMOOTH_WINDOW

mp_pose = mp.solutions.pose

def extract_head_angle_deg(video_path, session_folder, logger=None):
    """
    Head Angle (2D pixel only) - Production Grade:
    - Head vector: MidEar -> Nose
    - Torso vector: MidHip -> MidShoulder
    - Angle = 180° - included angle → forward tilt deviation (0° = aligned)
    - CSV: only frame, HeadAngle_deg
    - Saves to session_folder + "HeadAngle_deg.csv"
    """
    # Logger setup if not provided
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Head Angle extraction for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_angles = []

    # Optional handedness detect 
    # handedness = detect_handedness(...)  # Uncomment if needed

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting HeadAngle_deg"):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Frame read failed at {frame_idx}")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if not result.pose_landmarks:
                logger.debug(f"No landmarks in frame {frame_idx}")
                raw_angles.append(np.nan)
                continue

            lm = result.pose_landmarks.landmark

            nose, vN = get_point(lm, mp_pose.PoseLandmark.NOSE.value, w, h)
            le, vLE = get_point(lm, mp_pose.PoseLandmark.LEFT_EAR.value, w, h)
            re, vRE = get_point(lm, mp_pose.PoseLandmark.RIGHT_EAR.value, w, h)
            ls, vLS = get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            rs, vRS = get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
            lh, vLH = get_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
            rh, vRH = get_point(lm, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)

            if min(vN, vLE, vRE, vLS, vRS, vLH, vRH) < VISIBILITY_THRESH:
                logger.debug(f"Low visibility in frame {frame_idx}")
                raw_angles.append(np.nan)
                continue

            mid_ear = (le + re) / 2.0
            mid_shoulder = (ls + rs) / 2.0
            mid_hip = (lh + rh) / 2.0

            head_vec = nose - mid_ear
            torso_vec = mid_shoulder - mid_hip

            included_angle = angle_between(head_vec, torso_vec)
            head_angle_deg = 180.0 - included_angle

            # Outlier cap (tracking errors prevent)
            head_angle_deg = min(head_angle_deg, 60.0)

            raw_angles.append(head_angle_deg)

    cap.release()
    logger.info("Processing completed")

    # Smoothing
    series = pd.Series(raw_angles)
    if SMOOTH_WINDOW > 1:
        series = series.rolling(window=SMOOTH_WINDOW, min_periods=1, center=True).mean()

    # NaN fill
    series = series.ffill().bfill()
    logger.info(f"NaN fill applied, remaining NaNs: {series.isna().sum()}")

    # Final CSV
    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "HeadAngle_deg": series.round(4)
    })

    csv_path = os.path.join(session_folder, "HeadAngle_deg.csv")
    df.to_csv(csv_path, index=False)
    avg_angle = df['HeadAngle_deg'].mean()
    logger.info(f"CSV saved: {csv_path} | Avg Head Angle: {avg_angle:.2f}°")

    print(f"✅ Head Angle CSV saved: {csv_path}")
    return df