import faiss
import json
import os
from config import FAISS_INDEX_PATH, METADATA_PATH, DOCS_PATH

def load_or_create_index(dimension=1024):
    if FAISS_INDEX_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))

        with open(METADATA_PATH, "r") as f:
            metadatas = json.load(f)

        with open(DOCS_PATH, "r") as f:
            documents = [line.strip() for line in f.readlines()]

        print("Loaded existing index:", index.ntotal)

    else:
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50

        metadatas = []
        documents = []

        print("Created new index")

    return index, metadatas, documents


def save_index(index, metadatas, documents):
    import json
    from config import FAISS_INDEX_PATH, METADATA_PATH, DOCS_PATH

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with open(METADATA_PATH, "w") as f:
        json.dump(metadatas, f, indent=2)

    with open(DOCS_PATH, "w") as f:
        for d in documents:
            f.write(d.strip() + "\n")

    print("Index saved successfully.")
