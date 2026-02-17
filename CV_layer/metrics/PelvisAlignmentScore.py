# metrics/pelvis_alignment_score.py

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

def extract_pelvis_alignment_score(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=5,
    burn_in_frames=15,
    ref_frame_start=15,
    ref_frame_end=30,
    k=0.1,                      # Sensitivity for exp decay (higher = stricter on tilt)
    min_hip_width_px=30.0
):
    """
    Pelvis Alignment Score - Production Grade (Finalized):
    - Combines pelvis tilt (hip line vs horizontal) + vertical symmetry (hip y-diff)
    - Score 0–1: 1 = perfectly aligned (neutral tilt + symmetric hips)
    - Relative to stance reference (baseline subtracted)
    - CSV: frame, PelvisAlignmentScore
    - Saves to session_folder + "PelvisAlignmentScore.csv"
    - Best for front-on or semi-front view
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Pelvis Alignment Score for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_scores = []
    ref_tilt_dev = None
    last_valid_width = 100.0

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

        for frame_idx in tqdm(range(total_frames), desc="Extracting PelvisAlignmentScore"):
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

            ls, vLS = get_point(lm, LS_ID, w, h)
            rs, vRS = get_point(lm, RS_ID, w, h)
            lh, vLH = get_point(lm, LH_ID, w, h)
            rh, vRH = get_point(lm, RH_ID, w, h)

            min_vis = min(vLH, vRH, vLS, vRS)
            if min_vis < visibility_thresh:
                logger.debug(f"Low visibility in frame {frame_idx}")
                raw_scores.append(np.nan)
                continue

            # Hip width for scale
            hip_width = abs(lh[0] - rh[0])
            if hip_width < min_hip_width_px:
                logger.debug(f"Hip width too small ({hip_width:.2f}px) in frame {frame_idx} - using fallback")
                hip_width = max(last_valid_width, min_hip_width_px)
            else:
                last_valid_width = hip_width

            # Hip vector (left to right)
            if lh[0] > rh[0]:
                lh, rh = rh, lh  # Ensure left < right
            hip_vec = rh - lh

            # Horizontal reference
            horizontal_vec = np.array([1.0, 0.0], dtype=np.float32)

            # Pelvis tilt angle (acute)
            tilt_angle = angle_between(hip_vec, horizontal_vec)
            tilt_dev = min(tilt_angle, 180 - tilt_angle)  # Acute only

            # Hip y-difference symmetry
            hip_y_diff = abs(lh[1] - rh[1])
            symmetry_ratio = hip_y_diff / (hip_width + 1e-9)
            symmetry_ratio = min(symmetry_ratio, 1.0)
            symmetry_score = 1 - symmetry_ratio
            symmetry_score = np.clip(symmetry_score, 0.0, 1.0)

            # Burn-in period
            if frame_idx < burn_in_frames:
                raw_scores.append(1.0)
                continue

            # Set reference tilt deviation from stable stance frames
            if frame_idx == ref_frame_end and ref_tilt_dev is None:
                valid_tilts = [t for t in raw_scores[ref_frame_start:ref_frame_end+1] if not np.isnan(t)]
                ref_tilt_dev = np.mean(valid_tilts) if valid_tilts else tilt_dev
                logger.info(f"Reference pelvis tilt deviation set: {ref_tilt_dev:.4f}° (frames {ref_frame_start}-{ref_frame_end})")

            # Relative tilt deviation
            if ref_tilt_dev is not None:
                relative_tilt = abs(tilt_dev - ref_tilt_dev)
            else:
                relative_tilt = tilt_dev

            # Final score: exp decay on tilt + symmetry
            exp_term = np.exp(-k * relative_tilt)
            alignment_score = exp_term * symmetry_score
            alignment_score = np.clip(alignment_score, 0.0, 1.0)

            raw_scores.append(alignment_score)

    cap.release()
    logger.info("Processing completed")

    series = pd.Series(raw_scores)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    series = series.ffill().bfill()
    logger.info(f"NaN fill applied, remaining NaNs: {series.isna().sum()}")

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "PelvisAlignmentScore": series.round(6)
    })

    csv_path = os.path.join(session_folder, "PelvisAlignmentScore.csv")
    df.to_csv(csv_path, index=False)
    avg_score = df['PelvisAlignmentScore'].mean()
    logger.info(f"CSV saved: {csv_path} | Avg Pelvis Alignment Score: {avg_score:.4f}")

    # Phase summary
    stance_avg = df.iloc[burn_in_frames:ref_frame_end]['PelvisAlignmentScore'].mean()
    impact_avg = df.iloc[ref_frame_end:]['PelvisAlignmentScore'].mean()
    logger.info(f"Stance phase avg: {stance_avg:.4f} | Impact/follow avg: {impact_avg:.4f}")

    print(f"✅ Pelvis Alignment Score CSV saved: {csv_path}")
    return df