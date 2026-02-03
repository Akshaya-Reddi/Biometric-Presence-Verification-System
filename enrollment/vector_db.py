import faiss
import numpy as np
import os
import json

INDEX_PATH = "vector_index.faiss"
METADATA_PATH = "vector_metadata.json"

DIMENSION = 512   # FaceNet embedding size

index = None
metadata = []

# Load or Create FAISS Index

def load_index():

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

        return index, metadata

    # Create new index if not exists
    index = faiss.IndexFlatIP(DIMENSION)   # Inner Product (cosine compatible)
    metadata = []

    return index, metadata

# Save Index + Metadata

def save_index(index, metadata):

    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)


# Add New Identity Vector

def add_identity(vector, user_id):

    index, metadata = load_index()

    vector = np.array(vector).astype("float32").reshape(1, -1)

    index.add(vector)

    metadata.append({
        "user_id": user_id
    })

    save_index(index, metadata)

# Search Closest Identity

def search_identity(query_vector, top_k=3):

    index, metadata = load_index()

    if index.ntotal == 0:
        return []

    query_vector = np.array(query_vector).astype("float32").reshape(1, -1)

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < len(metadata):
            results.append({
                "user_id": metadata[idx]["user_id"],
                "score": float(score)
            })

    return results


