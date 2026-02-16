# metrics/step_direction_deviation_deg.py

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging

mp_pose = mp.solutions.pose

# Utils import
from utils import get_point, angle_between, detect_handedness

def extract_step_direction_deviation_deg(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=5,
    burn_in_fraction=0.25,
    handedness_check_frames=30
):
    """
    Step Direction Deviation - Production Grade (Automatic & Handedness Integrated):
    - Front foot stride vector vs ideal horizontal (down the pitch)
    - Deviation in degrees (0° = perfect straight stride)
    - Handedness auto-detect from utils (first 30 valid frames)
    - Dynamic burn-in skip
    - CSV: frame, StepDirectionDeviation_deg (NaN early)
    - Saves to session_folder + "StepDirectionDeviation_deg.csv"
    - Best for side-on view
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Step Direction Deviation for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_deviation = [np.nan] * total_frames
    landmarks_list = []  # For handedness detection
    valid_count = 0
    vis_fail = 0

    burn_in_frames = max(15, int(total_frames * burn_in_fraction))
    logger.info(f"Dynamic burn-in: {burn_in_frames} frames ({burn_in_frames/total_frames*100:.1f}%)")

    handedness = 'right'  # Safe default
    front_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
    back_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting StepDirectionDeviation_deg"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                raw_deviation[frame_idx] = np.nan
                landmarks_list.append(None)
                continue

            lm = res.pose_landmarks.landmark

            front_ankle, vFront = get_point(lm, front_ankle_idx, w, h)
            back_ankle, vBack = get_point(lm, back_ankle_idx, w, h)

            min_vis = min(vFront, vBack)
            if min_vis < visibility_thresh:
                vis_fail += 1
                raw_deviation[frame_idx] = np.nan
                landmarks_list.append(None)
                continue

            landmarks_list.append(res.pose_landmarks)  # Full pose_landmarks for detect_handedness
            valid_count += 1

            # Handedness detect once
            if handedness == 'right' and valid_count >= handedness_check_frames:
                try:
                    handedness = detect_handedness(landmarks_list, w, h, num_frames=handedness_check_frames)
                    logger.info(f"Handedness detected after {valid_count} valid frames: {handedness}")
                except Exception as e:
                    logger.warning(f"Handedness detection failed: {e} - default 'right'")
                    handedness = 'right'

            # Update ankle indices based on handedness
            if handedness == 'right':
                front_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
                back_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value
            else:
                front_ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value
                back_ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value

            # Burn-in skip
            if frame_idx < burn_in_frames:
                raw_deviation[frame_idx] = np.nan
                continue

            stride_vec = front_ankle - back_ankle

            # Ideal horizontal vector (down the pitch)
            ideal_vec = np.array([1.0, 0.0])

            deviation_deg = angle_between(stride_vec, ideal_vec)
            deviation_deg = min(deviation_deg, 180.0 - deviation_deg)  # Acute only

            # Cap outliers
            deviation_deg = min(deviation_deg, 45.0)

            raw_deviation[frame_idx] = deviation_deg

    cap.release()
    logger.info("Processing completed")
    logger.info(f"Valid frames: {valid_count}/{total_frames} ({valid_count/total_frames*100:.1f}%)")
    logger.info(f"Visibility fails: {vis_fail}")

    # Smoothing
    series = pd.Series(raw_deviation)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    series = series.ffill().bfill()

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "StepDirectionDeviation_deg": series.round(4)
    })

    csv_path = os.path.join(session_folder, "StepDirectionDeviation_deg.csv")

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

    avg_dev = df['StepDirectionDeviation_deg'].mean(skipna=True)
    max_dev = df['StepDirectionDeviation_deg'].max(skipna=True)

    logger.info(f"Avg Step Direction Deviation: {avg_dev:.2f}°")
    logger.info(f"Max Step Direction Deviation: {max_dev:.2f}°")

    print(f"✅ Step Direction Deviation CSV saved: {csv_path}")
    print(f"  - Avg Deviation: {avg_dev:.2f}°")
    print(f"  - Max Deviation: {max_dev:.2f}°")

    return df