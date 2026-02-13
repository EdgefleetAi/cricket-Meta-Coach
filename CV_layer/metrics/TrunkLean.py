# metrics/trunk_lean_deg.py (Handedness Removed - Final Clean Version)

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging

mp_pose = mp.solutions.pose

from utils import get_point, angle_between

def extract_trunk_lean_deg(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=5,
    burn_in_fraction=0.25,
    max_lean_cap=60.0
):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Trunk Lean for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_lean = [np.nan] * total_frames
    last_valid_trunk_vec = None

    burn_in_frames = max(15, int(total_frames * burn_in_fraction))
    logger.info(f"Dynamic burn-in: {burn_in_frames} frames")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting TrunkLean_deg"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                raw_lean[frame_idx] = np.nan
                continue

            lm = res.pose_landmarks.landmark

            ls, vLS = get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            rs, vRS = get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
            lh, vLH = get_point(lm, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)
            rh, vRH = get_point(lm, mp_pose.PoseLandmark.RIGHT_HIP.value, w, h)

            min_vis = min(vLS, vRS, vLH, vRH)
            if min_vis < visibility_thresh:
                if last_valid_trunk_vec is not None:
                    trunk_vec = last_valid_trunk_vec
                else:
                    raw_lean[frame_idx] = np.nan
                    continue

            mid_shoulder = (ls + rs) / 2.0
            mid_hip = (lh + rh) / 2.0

            trunk_vec = mid_hip - mid_shoulder
            last_valid_trunk_vec = trunk_vec

            vertical_ref = np.array([0.0, 1.0])  # Downward

            lean_deg = angle_between(trunk_vec, vertical_ref)
            lean_deg = min(lean_deg, max_lean_cap)

            if frame_idx < burn_in_frames:
                raw_lean[frame_idx] = np.nan
            else:
                raw_lean[frame_idx] = lean_deg

    cap.release()

    series = pd.Series(raw_lean)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    series = series.ffill().bfill()

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "TrunkLean_deg": series.round(4)
    })

    csv_path = os.path.join(session_folder, "TrunkLean_deg.csv")

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

    avg_lean = df['TrunkLean_deg'].mean(skipna=True)
    max_lean = df['TrunkLean_deg'].max(skipna=True)

    approx_impact = int(total_frames * 0.6)
    pre = df['TrunkLean_deg'].iloc[:approx_impact].mean(skipna=True)
    post = df['TrunkLean_deg'].iloc[approx_impact:].mean(skipna=True)

    logger.info(f"Avg Trunk Lean: {avg_lean:.2f}°")
    logger.info(f"Max Trunk Lean: {max_lean:.2f}°")
    logger.info(f"Pre-impact Avg: {pre:.2f}°")
    logger.info(f"Post-impact Avg: {post:.2f}°")

    print(f"✅ Trunk Lean CSV saved: {csv_path}")
    print(f"  - Avg Lean: {avg_lean:.2f}°")
    print(f"  - Max Lean: {max_lean:.2f}°")

    return df