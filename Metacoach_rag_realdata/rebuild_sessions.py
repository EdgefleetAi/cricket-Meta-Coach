import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

# -----------------------------
# PATHS
# -----------------------------
INDEX_DIR = Path("rag_index_real")
FAISS_INDEX_PATH = INDEX_DIR / "unified_hnsw.index"
METADATA_PATH = INDEX_DIR / "metadatas.json"
DOCS_PATH = INDEX_DIR / "documents.txt"

EMBEDDING_MODEL = "BAAI/bge-m3"

# -----------------------------
# LOAD OLD DATA
# -----------------------------
print("Loading old metadata...")

with open(METADATA_PATH, "r") as f:
    metadatas = json.load(f)

with open(DOCS_PATH, "r") as f:
    documents = [line.strip() for line in f.readlines()]

print("Total old chunks:", len(metadatas))

# -----------------------------
# FILTER ONLY SESSION 03
# -----------------------------
filtered_meta = []
filtered_docs = []

for meta, doc in zip(metadatas, documents):
    if meta["session_id"] == "player_01_session_03":
        meta["session_id"] = "player_01_session_01"  # rename
        filtered_meta.append(meta)
        filtered_docs.append(doc.replace("player_01_session_03", "player_01_session_01"))

print("Remaining chunks after filter:", len(filtered_meta))

# -----------------------------
# REBUILD FAISS INDEX
# -----------------------------
print("Loading embedder...")
embedder = SentenceTransformer(EMBEDDING_MODEL)

print("Re-embedding...")
embeddings = embedder.encode(
    filtered_docs,
    normalize_embeddings=True,
    convert_to_numpy=True
).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexHNSWFlat(dimension, 32)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 50

index.add(embeddings)

# -----------------------------
# SAVE NEW CLEAN INDEX
# -----------------------------
faiss.write_index(index, str(FAISS_INDEX_PATH))

with open(METADATA_PATH, "w") as f:
    json.dump(filtered_meta, f, indent=2)

with open(DOCS_PATH, "w") as f:
    for d in filtered_docs:
        f.write(d.strip() + "\n")

print("✅ Rebuild complete.")
print("New total chunks:", index.ntotal)
