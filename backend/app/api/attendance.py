from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import uuid
import numpy as np

from app.database.deps import get_db
from app.schemas.attendance_request import AttendanceRequest
from app.services.attendance_service import decide_attendance
from app.repositories.attendance_repo import create_attendance
from app.services.multiframe_validator import validate_multiframe
from app.services.liveness_validator import validate_liveness
from app.services.vector_sync import VectorIndex
from app.repositories.device_repo import is_registered_device
from app.security.request_guard import is_fresh_request
from app.security.signature import verify_signature

router = APIRouter()

@router.post("/attendance/mark")
def mark_attendance(
    request: AttendanceRequest,
    db: Session = Depends(get_db)
):
    user_uuid = uuid.UUID(request.user_id)
    session_uuid = uuid.UUID(request.session_id)
    attendance_id = uuid.uuid4()

    # Device trust
    if not is_registered_device(db, request.device_id):
        return {"error": "Untrusted device"}

    # Replay protection
    if not is_fresh_request(request.timestamp):
        return {"error": "Request expired"}

    payload = f"{request.user_id}:{request.session_id}:{request.timestamp}"

    if not verify_signature(payload, request.signature):
        return {"error": "Invalid request signature"}

    # Liveness validation
    live_ok, liveness_score = validate_liveness(request.liveness_scores)
    if not live_ok:
        return {"error": "Liveness failed"}

    # Multi-frame stability
    stable, stability_score, final_embedding = validate_multiframe(request.embeddings)
    if not stable:
        return {"error": "Identity unstable"}

    # FAISS match
    results = VectorIndex().search(np.array(final_embedding))
    match_confidence = results[0]["score"] if results else 0.0

    # Attendance decision
    status, reason = decide_attendance(
        db=db,
        user_id=user_uuid,
        session_id=session_uuid,
        match_confidence=match_confidence,
        liveness_score=liveness_score,
        stability_score=stability_score,
        attendance_id=attendance_id
    )

    record = create_attendance(
        db=db,
        id=attendance_id,
        user_id=user_uuid,
        session_id=session_uuid,
        status=status,
        confidence=match_confidence,
        stability=stability_score,
        liveness=liveness_score
    )

    db.commit()

    return {
        "status": status.value,
        "reason": reason,
        "confidence": match_confidence,
        "stability": stability_score,
        "liveness": liveness_score,
        "attendance_id": str(record.id)
    }
