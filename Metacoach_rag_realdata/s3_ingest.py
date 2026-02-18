# s3_ingest.py

import boto3
import pandas as pd
from io import BytesIO
from config import BUCKET_NAME, FRONT_PREFIX, SIDE_PREFIX

s3 = boto3.client("s3")

def list_all_test_csvs():
    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix="csvs/test/"
    )

    keys = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".csv")
    ]

    return keys


def read_csv_from_s3(key):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    df = pd.read_csv(BytesIO(obj["Body"].read()))
    return df

def build_metric_summary(key):
    df = read_csv_from_s3(key)

    metric_name = key.split("/")[-1].replace(".csv", "")

    print(f"\nProcessing {metric_name}")
    print("Columns:", df.columns.tolist())

    # Force select metric column by name
    if metric_name in df.columns:
        values = df[metric_name]
    else:
        raise ValueError(
            f"Metric column {metric_name} not found in {key}. "
            f"Available columns: {df.columns.tolist()}"
        )

    return {
        "metric": metric_name,
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "unit": ""
    }

