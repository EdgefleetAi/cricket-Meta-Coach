# metrics/head_lateral_deviation_deg.py   ← Final Sensitive Version

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import logging

mp_pose = mp.solutions.pose

from utils import get_point

def extract_head_lateral_deviation_deg(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=7,
    burn_in_fraction=0.20,
    ref_frames_after_burn=30,
    multiplier=40.0,           # Increased for sensitivity
    max_deviation_cap=45.0
):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

    logger.info(f"Starting Head Lateral Deviation for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    head_x_list = [np.nan] * total_frames
    raw_deviation = [np.nan] * total_frames

    burn_in_frames = max(15, int(total_frames * burn_in_fraction))
    ref_start = burn_in_frames
    ref_end = burn_in_frames + ref_frames_after_burn

    logger.info(f"Burn-in: {burn_in_frames} frames | Ref window: {ref_start} to {ref_end}")

    ref_head_x = None

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting HeadLateralDeviation_deg"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                head_x_list[frame_idx] = np.nan
                raw_deviation[frame_idx] = np.nan
                continue

            lm = res.pose_landmarks.landmark

            nose, vN = get_point(lm, mp_pose.PoseLandmark.NOSE.value, w, h)
            le, vLE = get_point(lm, mp_pose.PoseLandmark.LEFT_EAR.value, w, h)
            re, vRE = get_point(lm, mp_pose.PoseLandmark.RIGHT_EAR.value, w, h)
            ls, vLS = get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            rs, vRS = get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)

            if min(vN, vLE, vRE, vLS, vRS) < visibility_thresh:
                head_x_list[frame_idx] = np.nan
                raw_deviation[frame_idx] = np.nan
                continue

            head_x = (nose[0] + le[0] + re[0]) / 3.0
            head_x_list[frame_idx] = head_x

            # Normalize by IMAGE WIDTH (more consistent than head/shoulder width)
            image_width = w

            if frame_idx < burn_in_frames:
                raw_deviation[frame_idx] = 0.0
                continue

            if frame_idx == ref_end and ref_head_x is None:
                valid_ref = [x for x in head_x_list[ref_start:ref_end+1] if not np.isnan(x)]
                ref_head_x = np.mean(valid_ref) if valid_ref else head_x
                logger.info(f"Reference head x set: {ref_head_x:.2f} px (avg of {len(valid_ref)} frames)")

            if ref_head_x is None:
                raw_deviation[frame_idx] = 0.0
                continue

            d_x = abs(head_x - ref_head_x)
            d_norm = d_x / image_width

            deviation_deg = np.degrees(np.arctan(d_norm)) * multiplier

            # Minimum floor for small real movements
            if d_x > 1.0 and deviation_deg < 2.0:
                deviation_deg = 2.0

            deviation_deg = min(deviation_deg, max_deviation_cap)
            raw_deviation[frame_idx] = deviation_deg

    cap.release()

    series = pd.Series(raw_deviation)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    series = series.ffill().bfill()

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "HeadLateralDeviation_deg": series.round(4)
    })

    csv_path = os.path.join(session_folder, "HeadLateralDeviation_deg.csv")

    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except:
            pass

    try:
        df.to_csv(csv_path, index=False)
        logger.info(f"CSV saved: {csv_path}")
    except PermissionError:
        logger.error("Permission denied on CSV write. Close Excel and rerun.")
        return df

    avg_dev = df['HeadLateralDeviation_deg'].mean(skipna=True)
    max_dev = df['HeadLateralDeviation_deg'].max(skipna=True)

    logger.info(f"Avg Head Lateral Deviation: {avg_dev:.2f}°")
    logger.info(f"Max Head Lateral Deviation: {max_dev:.2f}°")

    print(f"✅ Head Lateral Deviation CSV saved: {csv_path}")
    print(f"  - Avg Deviation: {avg_dev:.2f}°")
    print(f"  - Max Deviation: {max_dev:.2f}°")

    return df