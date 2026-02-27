# upload_test_videos.py
import boto3
import os

BUCKET_NAME = "metacoach-videos-s3-bucket"
LOCAL_FOLDER = r"C:\Users\srira\Downloads\MetaCoach\CV_layer\videos"  # ← change if needed
S3_PREFIX = "videos/raw/test/"  # S3 lo test folder create avuthundi

s3 = boto3.client('s3')

files_to_upload = [
    "front_1.MOV",
    "front_2.MOV",
    "front_3.MOV",
    "side_1.MOV",
    "side_2.MOV"
]

uploaded = 0

for file_name in files_to_upload:
    local_path = os.path.join(LOCAL_FOLDER, file_name)
    if not os.path.exists(local_path):
        print(f"File missing: {file_name}")
        continue
    
    s3_key = S3_PREFIX + file_name
    try:
        s3.upload_file(local_path, BUCKET_NAME, s3_key)
        print(f"Success: {file_name} → s3://{BUCKET_NAME}/{s3_key}")
        uploaded += 1
    except Exception as e:
        print(f"Error {file_name}: {str(e)}")

print(f"\nUploaded {uploaded}/{len(files_to_upload)} test videos!")