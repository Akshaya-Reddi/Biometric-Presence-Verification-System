import numpy as np

def validate_multiframe(embeddings):
    """
    embeddings: list of vectors (list[list[float]])
    returns:
        stable: bool
        stability_score: float
        final_embedding: np.array
    """

    if not embeddings or len(embeddings) < 3:
        return False, 0.0, None

    vectors = np.array(embeddings).astype("float32")

    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Mean embedding
    mean_vector = np.mean(vectors, axis=0)
    mean_vector = mean_vector / np.linalg.norm(mean_vector)

    # Stability score = mean cosine similarity to mean vector
    sims = np.dot(vectors, mean_vector)
    stability_score = float(np.mean(sims))

    stable = stability_score >= 0.65

    return stable, stability_score, mean_vector
