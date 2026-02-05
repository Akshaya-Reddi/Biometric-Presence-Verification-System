def adaptive_face_threshold(stability_score: float):
    """
    Adjust face match threshold based on stability.
    More stable frames allow slightly lower threshold.
    """
    base = 0.75

    if stability_score > 0.85:
        return base - 0.05
    elif stability_score < 0.60:
        return base + 0.05

    return base


def adaptive_liveness_threshold(stability_score: float):
    """
    If stability is poor, demand higher liveness confidence.
    """
    base = 0.65

    if stability_score < 0.60:
        return base + 0.05

    return base

def combined_confidence(match, liveness, stability):
    """
    Weighted confidence score.
    """
    return (
        match * 0.5 +
        liveness * 0.3 +
        stability * 0.2
    )
