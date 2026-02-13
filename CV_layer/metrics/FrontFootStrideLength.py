# metrics/frontfoot_stride_length_cm.py

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

def extract_frontfoot_stride_length_cm(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=3,
    shoulder_width_cm=40.0,
    min_shoulder_px=40.0,
    max_stride_cm=150.0,
    burn_in_fraction=0.25,
    handedness_check_frames=30
):
    """
    Front Foot Stride Length (cm) - Production Grade (Automatic & Robust):
    - Distance between back ankle and front ankle (handedness auto-detect)
    - Scale using average shoulder width (40 cm default)
    - Dynamic burn-in skip
    - CSV: frame, FrontFootStrideLength_cm (NaN early)
    - Saves to session_folder + "FrontFootStrideLength_cm.csv"
    - Best for side-on view
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Front Foot Stride Length for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    landmarks_list = []  # For handedness
    shoulder_widths = []
    valid_count = 0
    vis_fail = 0

    burn_in_frames = max(15, int(total_frames * burn_in_fraction))
    logger.info(f"Dynamic burn-in: {burn_in_frames} frames ({burn_in_frames/total_frames*100:.1f}%)")

    handedness = 'right'  # Safe default
    front_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
    back_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    # Pass 1: Collect landmarks for handedness & shoulder calibration
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Pass 1: Collecting landmarks"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                landmarks_list.append(None)
                continue

            lm = res.pose_landmarks.landmark

            # Shoulder width for scale
            ls, vLS = get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            rs, vRS = get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)
            if min(vLS, vRS) >= visibility_thresh:
                shoulder_px = np.linalg.norm(ls - rs)
                if shoulder_px >= min_shoulder_px:
                    shoulder_widths.append(shoulder_px)

            landmarks_list.append(res.pose_landmarks)
            valid_count += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset for Pass 2

    # Handedness detect
    if valid_count >= handedness_check_frames:
        try:
            handedness = detect_handedness([lm for lm in landmarks_list if lm is not None], w, h, num_frames=handedness_check_frames)
            logger.info(f"Handedness detected: {handedness}")
        except Exception as e:
            logger.warning(f"Handedness detection failed: {e} - default 'right'")
            handedness = 'right'
    else:
        logger.warning("Not enough valid frames for handedness – default 'right'")
        handedness = 'right'

    # Ankle indices based on handedness
    if handedness == 'right':
        front_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
        back_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value
    else:
        front_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value
        back_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value

    # Scale factor from average shoulder
    if shoulder_widths:
        avg_shoulder_px = np.mean(shoulder_widths)
        scale_factor = shoulder_width_cm / avg_shoulder_px
        logger.info(f"Avg shoulder width px: {avg_shoulder_px:.2f}, Scale factor: {scale_factor:.4f}")
    else:
        scale_factor = 0.4  # Fallback (approx)
        logger.warning("No valid shoulder width – fallback scale factor 0.4")

    # Pass 2: Calculate stride
    raw_stride = [np.nan] * total_frames

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Pass 2: Calculating stride"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                raw_stride[frame_idx] = np.nan
                continue

            lm = res.pose_landmarks.landmark

            front_ankle, vFront = get_point(lm, front_ankle_idx, w, h)
            back_ankle, vBack = get_point(lm, back_ankle_idx, w, h)

            if min(vFront, vBack) < visibility_thresh:
                raw_stride[frame_idx] = np.nan
                continue

            stride_px = np.linalg.norm(front_ankle - back_ankle)
            stride_cm = stride_px * scale_factor
            stride_cm = min(stride_cm, max_stride_cm)  # Cap outliers

            raw_stride[frame_idx] = stride_cm

    cap.release()

    # Smoothing
    series = pd.Series(raw_stride)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    series = series.ffill().bfill()

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "FrontFootStrideLength_cm": series.round(2)
    })

    csv_path = os.path.join(session_folder, "FrontFootStrideLength_cm.csv")

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

    avg_stride = df['FrontFootStrideLength_cm'].mean(skipna=True)
    max_stride = df['FrontFootStrideLength_cm'].max(skipna=True)

    logger.info(f"Avg Front Foot Stride Length: {avg_stride:.2f} cm")
    logger.info(f"Max Front Foot Stride Length: {max_stride:.2f} cm")

    print(f"✅ Front Foot Stride Length CSV saved: {csv_path}")
    print(f"  - Avg Stride: {avg_stride:.2f} cm")
    print(f"  - Max Stride: {max_stride:.2f} cm")

    return df