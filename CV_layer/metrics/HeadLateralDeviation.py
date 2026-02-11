# metrics/head_lateral_deviation.py

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
    smooth_window=5,
    burn_in_frames=15,
    ref_frame_start=15,
    ref_frame_end=30,
    max_deviation_cap=45.0
):
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # DEBUG level for frame-by-frame logs
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
        logger.addHandler(handler)

        file_handler = logging.FileHandler(os.path.join(session_folder, "head_lateral_deviation_debug.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s'))
        logger.addHandler(file_handler)

    logger.info(f"Starting Head Lateral Deviation for: {video_path}")
    logger.info(f"Params: visibility_thresh={visibility_thresh}, smooth_window={smooth_window}, burn_in={burn_in_frames}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_deviation = []
    head_x_list = []
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
                logger.warning(f"Frame read failed at {frame_idx}")
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                logger.debug(f"Frame {frame_idx:04d} | No landmarks detected")
                raw_deviation.append(np.nan)
                head_x_list.append(np.nan)
                continue

            lm = res.pose_landmarks.landmark

            nose, vN = get_point(lm, mp_pose.PoseLandmark.NOSE.value, w, h)
            le, vLE = get_point(lm, mp_pose.PoseLandmark.LEFT_EAR.value, w, h)
            re, vRE = get_point(lm, mp_pose.PoseLandmark.RIGHT_EAR.value, w, h)
            ls, vLS = get_point(lm, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            rs, vRS = get_point(lm, mp_pose.PoseLandmark.RIGHT_SHOULDER.value, w, h)

            min_vis = min(vN, vLE, vRE, vLS, vRS)
            if min_vis < visibility_thresh:
                logger.debug(f"Frame {frame_idx:04d} | Low visibility: {min_vis:.3f}")
                raw_deviation.append(np.nan)
                head_x_list.append(np.nan)
                continue

            # Head center x
            head_x = (nose[0] + le[0] + re[0]) / 3.0
            head_x_list.append(head_x)

            shoulder_width = abs(ls[0] - rs[0]) + 1e-9

            # Burn-in period
            if frame_idx < burn_in_frames:
                logger.debug(f"Frame {frame_idx:04d} | Burn-in skip | head_x={head_x:.2f}")
                raw_deviation.append(0.0)
                continue

            # Set reference head_x
            if frame_idx == ref_frame_end and ref_head_x is None:
                valid_ref_x = [x for x in head_x_list[ref_frame_start:ref_frame_end+1] if not np.isnan(x)]
                if valid_ref_x:
                    ref_head_x = np.mean(valid_ref_x)
                    logger.info(f"Reference head x set: {ref_head_x:.2f} (frames {ref_frame_start}-{ref_frame_end})")
                else:
                    ref_head_x = head_x
                    logger.warning(f"No valid ref frames – using current head_x={head_x:.2f}")

            # Deviation calculation
            if ref_head_x is None:
                logger.debug(f"Frame {frame_idx:04d} | No ref yet | head_x={head_x:.2f}")
                raw_deviation.append(0.0)
                continue

            d_x = abs(head_x - ref_head_x)
            d_norm = d_x / shoulder_width
            deviation_deg = np.degrees(np.arctan(d_norm)) * 6.0  # Adjusted sensitivity

            # Soft cap (prevent extreme noise)
            #deviation_deg = min(deviation_deg, max_deviation_cap)

            # Outlier smoothing (sudden jumps)
            if frame_idx > 50:
                prev_avg = np.nanmean(raw_deviation[max(0, frame_idx-20):frame_idx])
                if deviation_deg > prev_avg * 2.5:
                    deviation_deg = prev_avg * 1.5
                    logger.debug(f"Frame {frame_idx:04d} | Outlier clipped | raw={np.degrees(np.arctan(d_norm))*10:.2f} → {deviation_deg:.2f}")

            raw_deviation.append(deviation_deg)

            # Frame level debug log
            logger.debug(
                f"Frame {frame_idx:04d} | "
                f"head_x={head_x:.2f} | "
                f"ref_x={ref_head_x:.2f} | "
                f"d_x={d_x:.2f} | "
                f"shoulder_w={shoulder_width:.2f} | "
                f"d_norm={d_norm:.4f} | "
                f"dev_deg={deviation_deg:.2f}"
            )

    cap.release()
    logger.info("Processing completed")

    series = pd.Series(raw_deviation)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    series = series.ffill().bfill()
    logger.info(f"NaN fill applied, remaining NaNs: {series.isna().sum()}")

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "HeadLateralDeviation_deg": series.round(4)
    })

    csv_path = os.path.join(session_folder, "HeadLateralDeviation_deg.csv")
    df.to_csv(csv_path, index=False)
    avg_dev = df['HeadLateralDeviation_deg'].mean()
    logger.info(f"CSV saved: {csv_path} | Avg Head Lateral Deviation: {avg_dev:.2f}°")

    # Phase summary
    stance_avg = df.iloc[burn_in_frames:ref_frame_end]['HeadLateralDeviation_deg'].mean()
    impact_avg = df.iloc[ref_frame_end:]['HeadLateralDeviation_deg'].mean()
    logger.info(f"Stance phase avg: {stance_avg:.2f}° | Impact/follow avg: {impact_avg:.2f}°")

    print(f"✅ Head Lateral Deviation CSV saved: {csv_path}")
    return df