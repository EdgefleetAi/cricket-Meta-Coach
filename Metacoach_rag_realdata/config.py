from pathlib import Path

# ========= BASE DIRECTORY (VERY IMPORTANT FIX) =========
BASE_DIR = Path(__file__).resolve().parent.parent

# ========= LOCAL INDEX PATH =========
INDEX_DIR = BASE_DIR / "rag_index_real"
INDEX_DIR.mkdir(exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "unified_hnsw.index"
METADATA_PATH = INDEX_DIR / "metadatas.json"
DOCS_PATH = INDEX_DIR / "documents.txt"

# ========= AWS CONFIG =========
BUCKET_NAME = "metacoach-videos-s3-bucket"
FRONT_PREFIX = "csvs/test/front/"
SIDE_PREFIX = "csvs/test/side/"

# ========= MODEL =========
EMBEDDING_MODEL = "BAAI/bge-m3"
