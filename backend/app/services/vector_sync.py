import faiss
import numpy as np

DIM = 512   # FaceNet dimension

class VectorIndex:
    def __init__(self):
        self.index = faiss.IndexFlatIP(DIM)
        self.user_ids = []

    def load(self, embeddings):
        """
        embeddings = [
            {"user_id": "...", "vector": np.array([...])}
        ]
        """
        self.index.reset()
        self.user_ids = []

        vectors = []

        for item in embeddings:
            vectors.append(item["vector"])
            self.user_ids.append(item["user_id"])

        if not vectors:
            return

        matrix = np.vstack(vectors).astype("float32")
        faiss.normalize_L2(matrix)

        self.index.add(matrix)

    def search(self, query_vector, top_k=3):
        if self.index.ntotal == 0:
            return []

        query = query_vector.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append({
                "user_id": self.user_ids[idx],
                "score": float(score)
            })

        return results
