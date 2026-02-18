# embed_and_store.py

from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL
import numpy as np

embedder = SentenceTransformer(EMBEDDING_MODEL)

def add_chunk(index, metadatas, documents, meta_new, doc_new):
    new_emb = embedder.encode(
        [doc_new],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    index.add(new_emb)
    metadatas.append(meta_new)
    documents.append(doc_new.replace("\n", " "))
