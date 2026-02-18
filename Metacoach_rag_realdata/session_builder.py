# session_builder.py

from ideal_engine import evaluate_against_ideal

def make_chunk(session_id, metric_obj, phase, metric_description, view_type,
               player_id="player_01", player_segment="intermediate"):

    # 🔹 Ensure numeric safety (no structural change)
    metric_obj["mean"] = float(metric_obj["mean"])
    metric_obj["std"] = float(metric_obj["std"])
    metric_obj["min"] = float(metric_obj["min"])
    metric_obj["max"] = float(metric_obj["max"])

    ideal_eval = evaluate_against_ideal(
        metric_obj["metric"],
        metric_obj["mean"]
    )

    metadata = {
        "player_id": player_id,
        "player_segment": player_segment,
        "session_id": session_id,
        "view": view_type,
        "metric": metric_obj["metric"],
        "phase": phase,
        "values": metric_obj,   # ✅ unchanged
        "interpretation": {
            "mean": round(metric_obj["mean"], 2),
            "std": round(metric_obj["std"], 2),
            "min": round(metric_obj["min"], 2),
            "max": round(metric_obj["max"], 2),
            "ideal_range": ideal_eval["ideal_range"],
            "ideal_status": ideal_eval["ideal_status"],
            "deviation_from_ideal": ideal_eval["deviation_from_ideal"]
        }
    }

    text = (
        f"Player: {player_id}\n"
        f"Session: {session_id}\n"
        f"View: {view_type}\n"
        f"Metric: {metadata['metric']}\n"
        f"Mean: {metadata['interpretation']['mean']}\n"
        f"Ideal Range: {metadata['interpretation']['ideal_range']}\n"
        f"Ideal Status: {metadata['interpretation']['ideal_status']}\n"
        f"Deviation: {metadata['interpretation']['deviation_from_ideal']}\n"
    )

    return metadata, text
