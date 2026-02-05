def compute_liveness_consistency(scores):
    """
    scores: list of liveness scores across frames
    """
    if not scores:
        return 0.0

    avg = sum(scores) / len(scores)

    variance = sum((s - avg) ** 2 for s in scores) / len(scores)

    # Penalize unstable liveness
    consistency = max(0.0, avg - variance)

    return float(consistency)
