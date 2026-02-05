import numpy as np

def validate_liveness(liveness_scores):
    """
    liveness_scores: list[float]
    """

    if not liveness_scores:
        return False, 0.0

    avg_score = float(np.mean(liveness_scores))

    # Production-safe threshold
    return avg_score >= 0.5, avg_score
