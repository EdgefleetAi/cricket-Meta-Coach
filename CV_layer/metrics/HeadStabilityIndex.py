# metrics/head_stability_index.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging


# utils & config import (nee project lo correct path ivvu)
from utils import get_point, angle_between, detect_handedness
from config import VISIBILITY_THRESH, SMOOTH_WINDOW,k
mp_pose = mp.solutions.pose
def extract_head_stability_index(
    video_path,
    session_folder,
    logger=None,                      # sensitivity (higher = stricter on movement)
    min_shoulder_width_px=25.0   # reject far/low-res frames
):
    """
    Head Stability Index (HSI) - Production Grade:
      1) HeadCenter(t) = mean of visible (Nose + Eyes + Ears)
      2) d(t) = ||HeadCenter(t) - HeadCenter(t-1)||
      3) d_norm(t) = d(t) / shoulder_width(t)
      4) HSI(t) = exp(-k * d_norm_smoothed(t)) → 1.0 = perfect still, low = unstable

    Output CSV: session_folder / "HeadStabilityIndex.csv"
    Returns: DataFrame
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Head Stability Index for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    # Landmarks for head center
    HEAD_LMS = [
        mp_pose.PoseLandmark.NOSE.value,
        mp_pose.PoseLandmark.LEFT_EYE.value,
        mp_pose.PoseLandmark.RIGHT_EYE.value,
        mp_pose.PoseLandmark.LEFT_EAR.value,
        mp_pose.PoseLandmark.RIGHT_EAR.value,
    ]

    LS_ID = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RS_ID = mp_pose.PoseLandmark.RIGHT_SHOULDER.value

    raw_dnorm = []
    prev_head_center = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting HeadStabilityIndex"):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Frame read failed at {frame_idx}")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                logger.debug(f"No landmarks in frame {frame_idx}")
                raw_dnorm.append(np.nan)
                prev_head_center = None
                continue

            lm = res.pose_landmarks.landmark

            # Shoulders for scale
            LS, visLS = get_point(lm, LS_ID, w, h)
            RS, visRS = get_point(lm, RS_ID, w, h)

            if min(visLS, visRS) < VISIBILITY_THRESH:
                logger.debug(f"Low shoulder visibility in frame {frame_idx}")
                raw_dnorm.append(np.nan)
                prev_head_center = None
                continue

            shoulder_width = float(np.linalg.norm(LS - RS))
            if shoulder_width < min_shoulder_width_px:
                logger.debug(f"Shoulder width too small ({shoulder_width:.1f} px) in frame {frame_idx}")
                raw_dnorm.append(np.nan)
                prev_head_center = None
                continue

            # Head center
            head_pts = []
            for idx in HEAD_LMS:
                P, visP = get_point(lm, idx, w, h)
                if visP >= VISIBILITY_THRESH:
                    head_pts.append(P)

            if len(head_pts) < 2:
                logger.debug(f"Insufficient head landmarks in frame {frame_idx}")
                raw_dnorm.append(np.nan)
                prev_head_center = None
                continue

            head_center = np.mean(np.stack(head_pts, axis=0), axis=0)

            # Frame-to-frame motion
            if prev_head_center is None:
                raw_dnorm.append(0.0)
                prev_head_center = head_center
                continue

            d_px = float(np.linalg.norm(head_center - prev_head_center))
            d_norm = d_px / (shoulder_width + 1e-9)
            raw_dnorm.append(d_norm)
            prev_head_center = head_center

    cap.release()
    logger.info("Processing completed")

    # Smooth normalized motion
    s = pd.Series(raw_dnorm)
    s = s.clip(upper=s.quantile(0.95))  # Outlier cap (95th percentile)
    if SMOOTH_WINDOW > 1:
        s = s.rolling(window=SMOOTH_WINDOW, min_periods=1, center=True).mean()
    s = s.ffill().bfill()

    # Convert motion -> stability index (0..1)
    hsi = np.exp(-k * s.to_numpy(dtype=np.float32))
    hsi = np.clip(hsi, 0.0, 1.0)

    df = pd.DataFrame({
        "frame": np.arange(len(hsi)),
        "HeadStabilityIndex": hsi.round(6)
    })

    csv_path = os.path.join(session_folder, "HeadStabilityIndex.csv")
    df.to_csv(csv_path, index=False)
    avg_hsi = df['HeadStabilityIndex'].mean()
    logger.info(f"CSV saved: {csv_path} | Avg Head Stability Index: {avg_hsi:.4f}")

    print(f"✅ Head Stability Index CSV saved: {csv_path}")
    return df