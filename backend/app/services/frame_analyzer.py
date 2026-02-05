import numpy as np

def compute_stability(embeddings):
    """
    embeddings: list of numpy vectors
    returns stability score (0–1)
    """
    if len(embeddings) < 2:
        return 0.0

    distances = []

    for i in range(len(embeddings) - 1):
        a = embeddings[i]
        b = embeddings[i + 1]
        dist = np.linalg.norm(a - b)
        distances.append(dist)

    avg_dist = np.mean(distances)

    # Convert to stability score
    stability = max(0.0, 1.0 - avg_dist)

    return float(stability)

