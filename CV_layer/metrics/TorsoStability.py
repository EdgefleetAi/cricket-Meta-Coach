# metrics/torso_stability_index.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging

mp_pose = mp.solutions.pose

# Utils import (nee project lo correct path ivvu)
from utils import get_point

def extract_torso_stability_index(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=5,
    burn_in_frames=15,
    ref_frame_start=15,
    ref_frame_end=30,
    k=12.0,                     # Sensitivity for exp decay (higher = stricter)
    min_shoulder_width_px=25.0
):
    """
    Torso Stability Index - Production Grade (Finalized):
    - Torso center: Mid-shoulder to Mid-hip average
    - Frame-to-frame displacement normalized by shoulder width
    - Score 0–1: 1 = perfectly stable, lower = more wobble
    - Relative to stance reference (baseline subtracted)
    - CSV: frame, TorsoStabilityIndex
    - Saves to session_folder + "TorsoStabilityIndex.csv"
    - Best for side-on view (forward/backward movement clear)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Torso Stability Index for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_dnorm = []
    prev_torso_center = None
    ref_dnorm = None

    LS_ID = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RS_ID = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    LH_ID = mp_pose.PoseLandmark.LEFT_HIP.value
    RH_ID = mp_pose.PoseLandmark.RIGHT_HIP.value

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting TorsoStabilityIndex"):
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
                prev_torso_center = None
                continue

            lm = res.pose_landmarks.landmark

            ls, vLS = get_point(lm, LS_ID, w, h)
            rs, vRS = get_point(lm, RS_ID, w, h)
            lh, vLH = get_point(lm, LH_ID, w, h)
            rh, vRH = get_point(lm, RH_ID, w, h)

            min_vis = min(vLS, vRS, vLH, vRH)
            if min_vis < visibility_thresh:
                logger.debug(f"Low visibility in frame {frame_idx}")
                raw_dnorm.append(np.nan)
                prev_torso_center = None
                continue

            shoulder_width = float(np.linalg.norm(ls - rs))
            if shoulder_width < min_shoulder_width_px:
                logger.debug(f"Shoulder width too small ({shoulder_width:.1f}px) in frame {frame_idx}")
                raw_dnorm.append(np.nan)
                prev_torso_center = None
                continue

            mid_shoulder = (ls + rs) / 2.0
            mid_hip = (lh + rh) / 2.0
            torso_center = (mid_shoulder + mid_hip) / 2.0

            if prev_torso_center is None:
                raw_dnorm.append(0.0)
                prev_torso_center = torso_center
                continue

            d_px = float(np.linalg.norm(torso_center - prev_torso_center))
            d_norm = d_px / (shoulder_width + 1e-9)

            # Burn-in period
            if frame_idx < burn_in_frames:
                raw_dnorm.append(0.0)
                prev_torso_center = torso_center
                continue

            # Set reference d_norm from stable stance frames
            if frame_idx == ref_frame_end and ref_dnorm is None:
                valid_dnorms = [d for d in raw_dnorm[ref_frame_start:ref_frame_end+1] if not np.isnan(d)]
                ref_dnorm = np.mean(valid_dnorms) if valid_dnorms else d_norm
                logger.info(f"Reference d_norm set: {ref_dnorm:.4f} (frames {ref_frame_start}-{ref_frame_end})")

            # Relative displacement (deviation from stance baseline)
            if ref_dnorm is not None:
                relative_dnorm = abs(d_norm - ref_dnorm)
            else:
                relative_dnorm = d_norm

            raw_dnorm.append(relative_dnorm)
            prev_torso_center = torso_center

    cap.release()
    logger.info("Processing completed")

    s = pd.Series(raw_dnorm)
    if smooth_window > 1:
        s = s.rolling(window=smooth_window, min_periods=1, center=True).mean()

    # Outlier clip (95th percentile)
    if len(s) > 10:
        s = s.clip(upper=s.quantile(0.95))

    s = s.ffill().bfill()

    # Stability index: exp decay on relative displacement
    hsi = np.exp(-k * s.to_numpy(dtype=np.float32))
    hsi = np.clip(hsi, 0.0, 1.0)

    df = pd.DataFrame({
        "frame": np.arange(len(hsi)),
        "TorsoStabilityIndex": np.round(hsi, 6)
    })

    csv_path = os.path.join(session_folder, "TorsoStabilityIndex.csv")
    df.to_csv(csv_path, index=False)
    avg_hsi = df['TorsoStabilityIndex'].mean()
    logger.info(f"CSV saved: {csv_path} | Avg Torso Stability Index: {avg_hsi:.4f}")

    # Phase summary
    stance_avg = df.iloc[burn_in_frames:ref_frame_end]['TorsoStabilityIndex'].mean()
    impact_avg = df.iloc[ref_frame_end:]['TorsoStabilityIndex'].mean()
    logger.info(f"Stance phase avg: {stance_avg:.4f} | Impact/follow avg: {impact_avg:.4f}")

    print(f"✅ Torso Stability Index CSV saved: {csv_path}")
    return df