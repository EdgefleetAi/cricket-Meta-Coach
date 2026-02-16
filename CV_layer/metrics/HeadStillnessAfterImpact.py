# metrics/head_stillness_after_impact_deg.py

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

def extract_head_stillness_after_impact_deg(
    video_path,
    session_folder,
    logger=None,
    visibility_thresh=0.5,
    smooth_window=5,
    burn_in_fraction=0.3,            # Max 30% skip
    followthrough_frames=30,         # Frames after impact to measure
    shoulder_norm_factor=0.1,        # Normalize displacement by shoulder width * factor
    max_stillness_cap=45.0
):
    """
    Head Stillness After Impact - Production Grade (Automatic):
    - Head center (nose) displacement from impact reference
    - Cumulative angle deviation (degrees) post-impact
    - Impact frame auto-approximated (middle-end fallback)
    - CSV: frame, HeadStillnessAfterImpact_deg (NaN before impact)
    - Saves to session_folder + "HeadStillnessAfterImpact_deg.csv"
    - Best for side-on view
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    logger.info(f"Starting Head Stillness After Impact for: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video loaded: {total_frames} frames")

    raw_stillness = [np.nan] * total_frames
    head_positions = []
    reference_head = None
    impact_frame = None

    LS_ID = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    RS_ID = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    NOSE_ID = mp_pose.PoseLandmark.NOSE.value

    # Dynamic burn-in
    burn_in_frames = max(20, int(total_frames * burn_in_fraction))
    logger.info(f"Dynamic burn-in: {burn_in_frames} frames")

    # Approximate impact: middle-end fallback (or use previous metric peak)
    approx_impact = max(burn_in_frames + 10, int(total_frames * 0.6))
    logger.info(f"Approximate impact frame (fallback): {approx_impact}")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        for frame_idx in tqdm(range(total_frames), desc="Extracting HeadStillnessAfterImpact_deg"):
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            if not res.pose_landmarks:
                head_positions.append(None)
                raw_stillness[frame_idx] = np.nan
                continue

            lm = res.pose_landmarks.landmark

            nose, vN = get_point(lm, NOSE_ID, w, h)
            ls, vLS = get_point(lm, LS_ID, w, h)
            rs, vRS = get_point(lm, RS_ID, w, h)

            if min(vN, vLS, vRS) < visibility_thresh:
                head_positions.append(None)
                raw_stillness[frame_idx] = np.nan
                continue

            head_positions.append(nose)

            # Shoulder width for normalization
            shoulder_width = np.linalg.norm(ls - rs)

            # Impact frame set (fallback or manual)
            if impact_frame is None and frame_idx >= approx_impact:
                impact_frame = frame_idx
                reference_head = nose
                raw_stillness[frame_idx] = 0.0
                logger.info(f"Impact frame set (auto): {impact_frame}")

            if reference_head is not None and frame_idx > impact_frame:
                if nose is not None:
                    d_px = np.linalg.norm(nose - reference_head)
                    # Realistic angle: atan(d_px / shoulder_width * factor)
                    stillness_deg = np.degrees(np.arctan(d_px / (shoulder_width * shoulder_norm_factor)))
                    stillness_deg = min(stillness_deg, max_stillness_cap)
                    raw_stillness[frame_idx] = stillness_deg
                else:
                    raw_stillness[frame_idx] = np.nan

    cap.release()
    logger.info("Processing completed")

    # Smoothing (post-impact only)
    series = pd.Series(raw_stillness)
    if smooth_window > 1:
        series = series.rolling(window=smooth_window, min_periods=1, center=True).mean()

    series = series.ffill().bfill()

    df = pd.DataFrame({
        "frame": np.arange(len(series)),
        "HeadStillnessAfterImpact_deg": series.round(4)
    })

    csv_path = os.path.join(session_folder, "HeadStillnessAfterImpact_deg.csv")
    df.to_csv(csv_path, index=False)

    # Post-impact stats
    if impact_frame is not None:
        post_impact = df['HeadStillnessAfterImpact_deg'].iloc[impact_frame:]
        post_avg = post_impact.mean(skipna=True)
        post_max = post_impact.max(skipna=True)
        logger.info(f"Post-impact Avg (after frame {impact_frame}): {post_avg:.2f}°")
        logger.info(f"Post-impact Max: {post_max:.2f}°")
    else:
        post_avg = np.nan
        post_max = np.nan

    overall_avg = df['HeadStillnessAfterImpact_deg'].mean(skipna=True)
    logger.info(f"CSV saved: {csv_path} | Overall Avg Head Stillness: {overall_avg:.2f}°")

    print(f"✅ Head Stillness After Impact CSV saved: {csv_path}")
    print(f"  - Post-impact Avg: {post_avg:.2f}°")
    print(f"  - Post-impact Max: {post_max:.2f}°")
    print(f"  - Overall Avg: {overall_avg:.2f}°")

    return df