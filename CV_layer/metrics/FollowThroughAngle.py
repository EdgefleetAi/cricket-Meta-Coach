# metrics/followthrough_angle_deg.py

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

def extract_followthrough_angle_deg(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.35,
    smooth_window=7,
    min_burn_in=20,
    max_burn_fraction=0.35,         # Max 35% skip
    followthrough_fraction=0.45,     # Last 45% valid frames for follow-through
    min_elbow_wrist_dist_px=10.0
):
    """
    Follow-through Angle - Production Grade (Fully Automatic & Robust):
    - Burn-in auto: min 20 to max 35% of video length
    - Follow-through: Last followthrough_fraction of valid angles
    - Visibility fails: carry forward last valid angle
    - CSV: frame, FollowThroughAngle_deg
    - Saves to session_folder + "FollowThroughAngle_deg.csv"
    - Works for short & long videos automatically
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Follow-through Angle for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_angles = []
    valid_count = 0
    vis_fail = 0
    dist_fail = 0
    last_valid_angle = np.nan

    LE_ID = mp_pose.PoseLandmark.LEFT_ELBOW.value
    RE_ID = mp_pose.PoseLandmark.RIGHT_ELBOW.value
    LW_ID = mp_pose.PoseLandmark.LEFT_WRIST.value
    RW_ID = mp_pose.PoseLandmark.RIGHT_WRIST.value

    # Dynamic burn-in
    burn_in_frames = max(min_burn_in, int(total_frames * max_burn_fraction))
    logger.info(f"Dynamic burn-in: {burn_in_frames} frames ({burn_in_frames/total_frames*100:.1f}%)")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting FollowThroughAngle_deg"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                raw_angles.append(np.nan)
                continue

            lm = res.pose_landmarks.landmark

            le, vLE = get_point(lm, LE_ID, w, h)
            re, vRE = get_point(lm, RE_ID, w, h)
            lw, vLW = get_point(lm, LW_ID, w, h)
            rw, vRW = get_point(lm, RW_ID, w, h)

            min_vis = min(vLE, vRE, vLW, vRW)
            if min_vis < visibility_thresh:
                vis_fail += 1
                raw_angles.append(last_valid_angle if not np.isnan(last_valid_angle) else np.nan)
                continue

            mid_elbow = (le + re) / 2.0
            mid_wrist = (lw + rw) / 2.0

            ew_dist = np.linalg.norm(mid_wrist - mid_elbow)
            if ew_dist < min_elbow_wrist_dist_px:
                dist_fail += 1
                raw_angles.append(last_valid_angle if not np.isnan(last_valid_angle) else np.nan)
                continue

            if frame_idx < burn_in_frames:
                raw_angles.append(np.nan)
                continue

            bat_vec = mid_wrist - mid_elbow
            horizontal_vec = np.array([1.0, 0.0])
            angle = angle_between(bat_vec, horizontal_vec)
            follow_angle = np.clip(angle, 0, 180)
            follow_angle = min(follow_angle, 150.0)

            raw_angles.append(follow_angle)
            last_valid_angle = follow_angle
            valid_count += 1

    cap.release()
    logger.info(f"Valid angles: {valid_count}/{total_frames} ({valid_count/total_frames*100:.1f}%)")
    logger.info(f"Visibility fails: {vis_fail}")
    logger.info(f"Distance fails: {dist_fail}")

    # Smoothing
    series = pd.Series(raw_angles)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    series = series.ffill().bfill()

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "FollowThroughAngle_deg": series.round(4)
    })

    csv_path = os.path.join(session_folder, "FollowThroughAngle_deg.csv")
    df.to_csv(csv_path, index=False)

    # Follow-through phase: Last followthrough_fraction of valid angles
    valid_angles = df['FollowThroughAngle_deg'].dropna().values
    if len(valid_angles) > 0:
        start = int(len(valid_angles) * (1 - followthrough_fraction))
        follow_phase = valid_angles[start:]
        follow_avg = np.mean(follow_phase)
        follow_max = np.max(follow_phase)
        used_fraction = len(follow_phase) / len(valid_angles) * 100
        logger.info(f"Follow-through phase used: {used_fraction:.1f}% of valid angles")
    else:
        follow_avg = np.nan
        follow_max = np.nan

    avg_angle = df['FollowThroughAngle_deg'].mean()

    logger.info(f"CSV saved: {csv_path}")
    logger.info(f"Follow-through Avg (last {followthrough_fraction*100:.0f}% valid): {follow_avg:.2f}°")
    logger.info(f"Follow-through Max: {follow_max:.2f}°")
    logger.info(f"Overall Avg: {avg_angle:.2f}°")

    print(f"✅ Follow-through Angle CSV saved: {csv_path}")
    print(f"  - Follow-through Avg (last {followthrough_fraction*100:.0f}%): {follow_avg:.2f}°")
    print(f"  - Follow-through Max: {follow_max:.2f}°")
    print(f"  - Overall Avg: {avg_angle:.2f}°")

    return df