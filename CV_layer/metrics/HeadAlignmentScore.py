# metrics/head_alignment_score.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging

mp_pose = mp.solutions.pose

# Utils import
from utils import get_point

def extract_head_alignment_score(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=5,
    k=0.1  # sensitivity for exp decay (higher = stricter on tilt)
):
    """
    Head Alignment Score (2D pixel only) - Production Grade (Finalized):
    - Combines head roll (eye level tilt) + ear symmetry (left/right y diff)
    - Score 0–1: 1 = perfectly aligned (no tilt, symmetric ears)
    - CSV: frame, HeadAlignmentScore
    - Saves to session_folder + "HeadAlignmentScore.csv"
    - Best for front-on or semi-front view
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Head Alignment Score for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_scores = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting HeadAlignmentScore"):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Frame read failed at {frame_idx}")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                logger.debug(f"No landmarks in frame {frame_idx}")
                raw_scores.append(np.nan)
                continue

            lm = res.pose_landmarks.landmark

            leye, vLE = get_point(lm, mp_pose.PoseLandmark.LEFT_EYE.value, w, h)
            reye, vRE = get_point(lm, mp_pose.PoseLandmark.RIGHT_EYE.value, w, h)
            lear, vLEAR = get_point(lm, mp_pose.PoseLandmark.LEFT_EAR.value, w, h)
            rear, vREAR = get_point(lm, mp_pose.PoseLandmark.RIGHT_EAR.value, w, h)
            ls, vLS = get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            rs, vRS = get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)

            if min(vLE, vRE, vLEAR, vREAR, vLS, vRS) < visibility_thresh:
                logger.debug(f"Low visibility in frame {frame_idx}")
                raw_scores.append(np.nan)
                continue

            # 1. Head roll angle from eyes
            eye_dx = leye[0] - reye[0]
            eye_dy = leye[1] - reye[1]
            if abs(eye_dx) < 1e-6:
                roll_angle = 0.0
            else:
                roll_angle = np.degrees(np.arctan2(eye_dy, eye_dx))
            roll_dev = abs(roll_angle)

            # 2. Ear symmetry (y difference normalized)
            ear_dy = abs(lear[1] - rear[1])
            shoulder_width = abs(ls[0] - rs[0]) + 1e-9
            symmetry_score = 1 - (ear_dy / shoulder_width)
            symmetry_score = np.clip(symmetry_score, 0.0, 1.0)

            # 3. Combined alignment score
            alignment_score = np.exp(-k * roll_dev) * symmetry_score
            alignment_score = np.clip(alignment_score, 0.0, 1.0)

            raw_scores.append(alignment_score)

    cap.release()
    logger.info("Processing completed")

    # Smoothing
    series = pd.Series(raw_scores)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    # NaN fill
    series = series.ffill().bfill()
    logger.info(f"NaN fill applied, remaining NaNs: {series.isna().sum()}")

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "HeadAlignmentScore": series.round(6)
    })

    csv_path = os.path.join(session_folder, "HeadAlignmentScore.csv")
    df.to_csv(csv_path, index=False)
    avg_score = df['HeadAlignmentScore'].mean()
    logger.info(f"CSV saved: {csv_path} | Avg Alignment Score: {avg_score:.4f}")

    print(f"✅ Head Alignment Score CSV saved: {csv_path}")
    return df