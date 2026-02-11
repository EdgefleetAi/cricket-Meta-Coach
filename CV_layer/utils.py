
import logging
import mediapipe as mp
import numpy as np
from config import LOG_LEVEL

mp_pose = mp.solutions.pose
logger = logging.getLogger(__name__)

def setup_logging(log_file):
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

def get_point(lm, idx, w, h):
    p = lm[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32), float(p.visibility)

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    n1 = np.linalg.norm(v1) + 1e-9
    n2 = np.linalg.norm(v2) + 1e-9
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def detect_handedness(landmarks_list, width, height, num_frames=30):
    right_wrist_x_sum = left_wrist_x_sum = 0.0
    count = 0

    for lm in landmarks_list[:num_frames]:
        if lm is None:
            continue
        rw_x = lm.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * width
        lw_x = lm.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width
        right_wrist_x_sum += rw_x
        left_wrist_x_sum += lw_x
        count += 1

    if count == 0:
        logger.warning("No data for handedness – defaulting to 'right'")
        return 'right'

    avg_right_x = right_wrist_x_sum / count
    avg_left_x = left_wrist_x_sum / count

    # Right-handed: left wrist (top hand) more forward/rightward
    if avg_left_x > avg_right_x + 10:
        return 'right'
    else:
        return 'left'