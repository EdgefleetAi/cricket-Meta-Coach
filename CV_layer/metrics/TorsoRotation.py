# metrics/torso_rotation_deg.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging

mp_pose = mp.solutions.pose

# Utils import (nee project lo correct path ivvu)
from utils import get_point, angle_between

def extract_torso_rotation_deg(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=5,
    burn_in_frames=15,
    ref_frame_start=15,
    ref_frame_end=30,
    max_rotation_cap=90.0
):
    """
    Torso Rotation (Shoulder-Hip Separation) - Production Grade (Finalized):
    - Angle between shoulder line (left-right shoulder) and hip line
    - 0° = no rotation (aligned), higher = more coil (X-factor)
    - Relative to stance reference (neutral rotation subtracted)
    - CSV: frame, TorsoRotation_deg
    - Saves to session_folder + "TorsoRotation_deg.csv"
    - Best for front-on or semi-front view
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Torso Rotation extraction for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_rotation = []
    ref_rotation = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting TorsoRotation_deg"):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Frame read failed at {frame_idx}")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                logger.debug(f"No landmarks in frame {frame_idx}")
                raw_rotation.append(np.nan)
                continue

            lm = res.pose_landmarks.landmark

            ls, vLS = get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            rs, vRS = get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
            lh, vLH = get_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
            rh, vRH = get_point(lm, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)

            if min(vLS, vRS, vLH, vRH) < visibility_thresh:
                logger.debug(f"Low visibility in frame {frame_idx}")
                raw_rotation.append(np.nan)
                continue

            # Shoulder line vector (left to right)
            shoulder_vec = rs - ls

            # Hip line vector (left to right)
            hip_vec = rh - lh

            # Raw rotation angle between shoulder and hip lines
            rotation_deg = angle_between(shoulder_vec, hip_vec)

            # Take acute angle (0–90°)
            rotation_deg = min(rotation_deg, 180 - rotation_deg)

            # Burn-in period
            if frame_idx < burn_in_frames:
                raw_rotation.append(0.0)
                continue

            # Set reference rotation from stable stance frames
            if frame_idx == ref_frame_end and ref_rotation is None:
                valid_rots = [r for r in raw_rotation[ref_frame_start:ref_frame_end+1] if not np.isnan(r)]
                ref_rotation = np.mean(valid_rots) if valid_rots else rotation_deg
                logger.info(f"Reference torso rotation set: {ref_rotation:.2f}° (frames {ref_frame_start}-{ref_frame_end})")

            # Relative rotation (deviation from stance neutral)
            if ref_rotation is not None:
                relative_rotation = abs(rotation_deg - ref_rotation)
            else:
                relative_rotation = rotation_deg

            relative_rotation = min(relative_rotation, max_rotation_cap)

            raw_rotation.append(relative_rotation)

    cap.release()
    logger.info("Processing completed")

    # Smoothing
    series = pd.Series(raw_rotation)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    # NaN fill
    series = series.ffill().bfill()
    logger.info(f"NaN fill applied, remaining NaNs: {series.isna().sum()}")

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "TorsoRotation_deg": series.round(4)
    })

    csv_path = os.path.join(session_folder, "TorsoRotation_deg.csv")
    df.to_csv(csv_path, index=False)
    avg_rot = df['TorsoRotation_deg'].mean()
    logger.info(f"CSV saved: {csv_path} | Avg Torso Rotation: {avg_rot:.2f}°")

    # Phase summary
    stance_avg = df.iloc[burn_in_frames:ref_frame_end]['TorsoRotation_deg'].mean()
    impact_avg = df.iloc[ref_frame_end:]['TorsoRotation_deg'].mean()
    logger.info(f"Stance phase avg: {stance_avg:.2f}° | Impact/follow avg: {impact_avg:.2f}°")

    print(f"✅ Torso Rotation CSV saved: {csv_path}")
    return df