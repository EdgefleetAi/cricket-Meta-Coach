# metrics/backfoot_anchor_stability_index.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging

mp_pose = mp.solutions.pose

# Utils import
from utils import get_point, detect_handedness

def extract_backfoot_anchor_stability_index(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=3,
    burn_in_fraction=0.25,
    handedness_check_frames=30,
    k=20.0,  # Sensitivity for exp decay
    max_stability=1.0
):
    """
    Back Foot Anchor Stability Index - Production Grade (Automatic):
    - Back foot (ankle) displacement from reference (post-burn-in)
    - Stability = exp(-k * normalized displacement) [0–1]
    - Handedness auto-detect from utils
    - Dynamic burn-in for reference set
    - CSV: frame, BackFootAnchorStabilityIndex (NaN early)
    - Saves to session_folder + "BackFootAnchorStabilityIndex.csv"
    - Best for side-on view
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting BackFoot Anchor Stability for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_index = [np.nan] * total_frames
    landmarks_list = []
    valid_count = 0
    vis_fail = 0

    burn_in_frames = max(15, int(total_frames * burn_in_fraction))
    logger.info(f"Dynamic burn-in: {burn_in_frames} frames ({burn_in_frames/total_frames*100:.1f}%)")

    handedness = 'right'  # Safe default
    backfoot_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    reference_backfoot = None
    last_valid_backfoot = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting BackFootAnchorStabilityIndex"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                raw_index[frame_idx] = np.nan
                landmarks_list.append(None)
                continue

            lm = res.pose_landmarks.landmark

            ankle, vAnkle = get_point(lm, backfoot_idx, w, h)

            if vAnkle < visibility_thresh:
                vis_fail += 1
                # Carry last valid
                if last_valid_backfoot is not None:
                    ankle = last_valid_backfoot
                else:
                    raw_index[frame_idx] = np.nan
                    landmarks_list.append(None)
                    continue

            last_valid_backfoot = ankle

            landmarks_list.append(res.pose_landmarks)
            valid_count += 1

            # Handedness detect once
            if handedness == 'right' and valid_count >= handedness_check_frames:
                try:
                    handedness = detect_handedness(landmarks_list, w, h, num_frames=handedness_check_frames)
                    logger.info(f"Handedness detected after {valid_count} valid frames: {handedness}")
                except Exception as e:
                    logger.warning(f"Handedness detection failed: {e} - default 'right'")
                    handedness = 'right'

            # Update backfoot index based on handedness
            if handedness == 'right':
                backfoot_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value
            else:
                backfoot_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value

            # Set reference backfoot after burn-in
            if frame_idx >= burn_in_frames and reference_backfoot is None:
                reference_backfoot = ankle
                raw_index[frame_idx] = max_stability  # 1.0 at anchor set
                logger.info(f"Back foot reference set at frame {frame_idx} ({handedness})")

            # Calculate stability if reference set
            if reference_backfoot is not None:
                d_px = np.linalg.norm(ankle - reference_backfoot)
                # Normalize by shoulder width (approx stability)
                ls, vLS = get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
                rs, vRS = get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
                shoulder_width = np.linalg.norm(ls - rs) if min(vLS, vRS) >= visibility_thresh else 100.0
                d_norm = d_px / (shoulder_width + 1e-6)
                stability = np.exp(-k * d_norm)
                stability = np.clip(stability, 0.0, max_stability)
                raw_index[frame_idx] = stability

    cap.release()
    logger.info("Processing completed")
    logger.info(f"Valid frames: {valid_count}/{total_frames} ({valid_count/total_frames*100:.1f}%)")
    logger.info(f"Visibility fails: {vis_fail}")

    # Smoothing
    series = pd.Series(raw_index)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    series = series.ffill().bfill()

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "BackFootAnchorStabilityIndex": series.round(6)
    })

    csv_path = os.path.join(session_folder, "BackFootAnchorStabilityIndex.csv")

    # Permission fix
    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
            logger.info(f"Removed existing locked CSV: {csv_path}")
        except Exception as e:
            logger.warning(f"Could not remove existing CSV: {e}")

    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV saved: {csv_path}")
    except PermissionError as e:
        logger.error(f"Permission denied on CSV write: {e}")
        logger.error("Close Excel or any app using the file and rerun")
        return df

    avg_stability = df['BackFootAnchorStabilityIndex'].mean(skipna=True)
    min_stability = df['BackFootAnchorStabilityIndex'].min(skipna=True)

    logger.info(f"Avg BackFoot Anchor Stability Index: {avg_stability:.4f}")
    logger.info(f"Min BackFoot Anchor Stability Index: {min_stability:.4f} (lower = more movement)")

    print(f"✅ BackFoot Anchor Stability CSV saved: {csv_path}")
    print(f"  - Avg Stability Index: {avg_stability:.4f} (higher = more stable)")
    print(f"  - Min Stability Index: {min_stability:.4f}")

    return df