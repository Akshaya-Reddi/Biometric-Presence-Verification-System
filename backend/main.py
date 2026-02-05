from fastapi import FastAPI

from app.api import attendance

app = FastAPI(title="Biometric Presence Verification System")

# Register routers
app.include_router(attendance.router, tags=["Attendance"])
