# ideal_engine.py

from ideal_targets import IDEAL_TARGETS

def evaluate_against_ideal(metric_name, value):
    ideal_info = IDEAL_TARGETS.get(metric_name)

    if not ideal_info or "avg" not in ideal_info:
        return {
            "ideal_range": None,
            "ideal_status": "No ideal defined",
            "deviation_from_ideal": None
        }

    low, high = ideal_info["avg"]

    if low <= value <= high:
        return {
            "ideal_range": (low, high),
            "ideal_status": "Within Ideal Range",
            "deviation_from_ideal": 0
        }

    if value < low:
        return {
            "ideal_range": (low, high),
            "ideal_status": "Below Ideal Range",
            "deviation_from_ideal": round(low - value, 2)
        }

    return {
        "ideal_range": (low, high),
        "ideal_status": "Above Ideal Range",
        "deviation_from_ideal": round(value - high, 2)
    }
