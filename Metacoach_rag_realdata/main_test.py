# main_test.py

from load_index import load_or_create_index, save_index
from s3_ingest import build_metric_summary
from session_builder import make_chunk
from embed_and_store import add_chunk
import boto3
from config import BUCKET_NAME

# -----------------------------
# 1️⃣ Load Index
# -----------------------------
index, metadatas, documents = load_or_create_index()

# -----------------------------
# 2️⃣ Generate Session ID
# -----------------------------
existing_sessions = list(set([m["session_id"] for m in metadatas]))
session_number = len(existing_sessions) + 1
SESSION_ID = f"player_01_session_{session_number:02d}"

print("Creating:", SESSION_ID)

# -----------------------------
# 3️⃣ List ALL CSVs under csvs/test/
# -----------------------------
s3 = boto3.client("s3")

response = s3.list_objects_v2( 
    Bucket=BUCKET_NAME,
    Prefix="csvs/test/"
)

all_csvs = [
    obj["Key"]
    for obj in response.get("Contents", [])
    if obj["Key"].endswith(".csv")
]

if not all_csvs:
    print("❌ No CSV files found. Check S3 path.")
    exit()

print(f"Found {len(all_csvs)} CSV files")

# -----------------------------
# 4️⃣ Process Each CSV
# -----------------------------
for key in all_csvs:

    # Auto-detect view
    if "front_" in key:
        view_type = "front_on"
    elif "side_" in key:
        view_type = "side_on"
    else:
        print("Skipping unknown folder:", key)
        continue

    print("Processing:", key)

    metric_summary = build_metric_summary(key)

    meta, doc = make_chunk(
        session_id=SESSION_ID,
        metric_obj=metric_summary,
        phase="stance → impact",
        metric_description=metric_summary["metric"],
        view_type=view_type
    )

    add_chunk(index, metadatas, documents, meta, doc)

# -----------------------------
# 5️⃣ Save Index
# -----------------------------
save_index(index, metadatas, documents)

print("✅ Test ingestion complete.")
print("Total vectors in index:", index.ntotal)
