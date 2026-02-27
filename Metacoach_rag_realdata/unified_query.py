# unified_query_engine.py

import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from load_index import load_or_create_index

# --------------------------------------------------
# LOAD INDEX + EMBEDDER (GLOBAL)
# --------------------------------------------------

print("Loading index...")
index, metadatas, documents = load_or_create_index()
print("Index loaded. Total chunks:", len(documents))

print("Loading embedder...")
embedder = SentenceTransformer("BAAI/bge-m3")
print("Embedder loaded.")

# --------------------------------------------------
# 1) QUERY INDEX
# --------------------------------------------------

def query_index(query, top_k=5):
    q_emb = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, ids = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({
            "score": float(score),
            "metadata": metadatas[idx],
            "text": documents[idx]
        })
    return results


# --------------------------------------------------
# 2) KNOWN METRICS (ALL 30)
# --------------------------------------------------

KNOWN_METRICS = [
    "HeadAngle_deg", "HeadStabilityIndex", "HeadVerticalDrift_deg",
    "HeadLateralDeviation_deg", "HeadAlignmentScore",
    "SpineAngle_deg", "TorsoRotation_deg", "TorsoStabilityIndex",
    "LateralBend_deg", "PelvisAlignmentScore",
    "FrontFootStrideLength_cm", "BackFootAnchorStabilityIndex",
    "WeightTransferEfficiency_pct", "StepDirectionDeviation_deg",
    "FootContactTime_ms",
    "BackliftAngle_deg", "SwingPlaneDeviation_deg", "BatSpeed_kph",
    "ImpactTiming_ms", "FollowThroughAngle_deg",
    "BalanceStabilityIndex", "CenterOfMassShift_cm",
    "FollowThroughLength_deg", "PostImpactRecoveryTime_ms",
    "HeadStillnessAfterImpact_deg",
    "LineJudgmentAccuracy_pct", "LengthJudgmentAccuracy_pct",
    "ShotAppropriatenessScore", "ReactionTime_ms", "RiskFactorScore",
]

# --------------------------------------------------
# 3) ALIASES
# --------------------------------------------------

METRIC_ALIASES = {
    "bat speed": "BatSpeed_kph",
    "bat speeed": "BatSpeed_kph",
    "bat spead": "BatSpeed_kph",
    "speed of bat": "BatSpeed_kph",
    "impact timing": "ImpactTiming_ms",
    "timing": "ImpactTiming_ms",
    "backlift": "BackliftAngle_deg",
    "swing plane": "SwingPlaneDeviation_deg",
    "swingplan": "SwingPlaneDeviation_deg",
    "follow through": "FollowThroughAngle_deg",

    "head stability": "HeadStabilityIndex",
    "hed stabillty": "HeadStabilityIndex",
    "head stab": "HeadStabilityIndex",
    "head angle": "HeadAngle_deg",
    "head tilt": "HeadAngle_deg",
    "vertical drift": "HeadVerticalDrift_deg",
    "lateral deviation": "HeadLateralDeviation_deg",
    "head alignment": "HeadAlignmentScore",

    "spine angle": "SpineAngle_deg",
    "torso rotation": "TorsoRotation_deg",
    "rotation of torso": "TorsoRotation_deg",
    "torso stability": "TorsoStabilityIndex",
    "stability of torso": "TorsoStabilityIndex",
    "lateral bend": "LateralBend_deg", 
    "pelvis alignment": "PelvisAlignmentScore",

    "stride length": "FrontFootStrideLength_cm",
    "front foot stride": "FrontFootStrideLength_cm",
    "back foot stability": "BackFootAnchorStabilityIndex",
    "weight transfer": "WeightTransferEfficiency_pct",
    "step direction": "StepDirectionDeviation_deg",
    "foot contact time": "FootContactTime_ms",

    "balance stability": "BalanceStabilityIndex",
    "center of mass": "CenterOfMassShift_cm",
    "com shift": "CenterOfMassShift_cm",
    "recovery time": "PostImpactRecoveryTime_ms",
    "head stillness": "HeadStillnessAfterImpact_deg",

    "line judgment": "LineJudgmentAccuracy_pct",
    "length judgment": "LengthJudgmentAccuracy_pct",
    "shot appropriateness": "ShotAppropriatenessScore",
    "reaction time": "ReactionTime_ms",
    "risk factor": "RiskFactorScore",
}

# --------------------------------------------------
# 4) EXTRACT METRICS
# --------------------------------------------------

def extract_metrics_from_query(q: str):
    q_low = q.lower()
    found = set()

    for m in KNOWN_METRICS:
        if m.lower() in q_low:
            found.add(m)

    for alias, metric in METRIC_ALIASES.items():
        if alias in q_low:
            found.add(metric)

    return list(found)

# --------------------------------------------------
# 5) EXTRACT SESSIONS 
# --------------------------------------------------

def extract_session_ids(q: str):
    q = q.lower()
    matches = re.findall(r"session[_\s]*(\d{1,2})", q)

    session_ids = []
    for m in matches:
        n = int(m)
        session_ids.append(f"player_01_session_{n:02d}")

    seen = set()
    out = []
    for s in session_ids:
        if s not in seen:
            out.append(s)
            seen.add(s)

    return out

# --------------------------------------------------
# 6) UNIFIED QUERY 
# --------------------------------------------------

def unified_query(q, top_k=10, wide_k=120):
    session_ids = extract_session_ids(q)
    metrics = extract_metrics_from_query(q)

    print("Sessions detected:", session_ids)
    print("Metrics detected:", metrics)

    if not metrics and not session_ids:
        return query_index(q, top_k)

    raw = query_index(q, top_k=wide_k)

    final_results = []

    def pick_best(metric=None, session_id=None, pool=None):
        pool = pool if pool is not None else raw
        filtered = pool

        if metric:
            filtered = [r for r in filtered if r["metadata"].get("metric") == metric]

        if session_id:
            filtered = [r for r in filtered if r["metadata"].get("session_id") == session_id]

        if not filtered:
            return None

        filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)
        return filtered[0]

    if metrics and session_ids:
        for sid in session_ids:
            for m in metrics:
                best = pick_best(metric=m, session_id=sid, pool=raw)

                if best is None:
                    forced_q = f"{m} {sid}"
                    forced_raw = query_index(forced_q, top_k=wide_k)
                    best = pick_best(metric=m, session_id=sid, pool=forced_raw)

                if best is None:
                    print(f"⚠ Missing chunk for metric={m}, session={sid}")
                    continue

                final_results.append(best)

        return final_results

    if metrics and not session_ids:
        for m in metrics:
            best = pick_best(metric=m, session_id=None, pool=raw)

            if best is None:
                forced_raw = query_index(m, top_k=wide_k)
                best = pick_best(metric=m, session_id=None, pool=forced_raw)

            if best is None:
                print(f"⚠ No chunks found for metric: {m}")
                continue

            final_results.append(best)

        return final_results

    if session_ids and not metrics:
        for sid in session_ids:
            best_for_session = [r for r in raw if r["metadata"].get("session_id") == sid]

            if not best_for_session:
                forced_raw = query_index(sid, top_k=wide_k)
                best_for_session = [r for r in forced_raw if r["metadata"].get("session_id") == sid]

            best_for_session = sorted(best_for_session, key=lambda x: x["score"], reverse=True)
            final_results.extend(best_for_session[:top_k])

        return final_results[:top_k]

    return raw[:top_k]


# --------------------------------------------------
# 7) SIMPLE TEST LOOP
# --------------------------------------------------

if __name__ == "__main__":

    while True:
        q = input("\nEnter query (type 'exit' to stop): ")

        if q.lower() == "exit":
            break

        results = unified_query(q)

        print("\nRESULTS:")
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            print("\n--- Result", i, "---")
            print("Score:", round(r["score"], 4))
            print("Session:", meta.get("session_id"))
            print("Metric:", meta.get("metric"))
            print("Mean:", meta.get("interpretation", {}).get("mean"))
            print("Ideal Status:", meta.get("interpretation", {}).get("ideal_status"))
