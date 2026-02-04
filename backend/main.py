from fastapi import FastAPI
from app.database.session import init_db
from app.api.enrollment import router as enrollment_router
from app.api import verify

app = FastAPI(title="Biometric Presence System")

app.include_router(enrollment_router, prefix="/api")
app.include_router(verify.router, prefix="/verify", tags=["verification"])

@app.on_event("startup")
def startup():
    init_db()

@app.get("/")
def health_check():
    return {"status": "ok", "service": "biometric-backend"}

