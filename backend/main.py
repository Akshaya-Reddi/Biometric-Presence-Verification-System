from fastapi import FastAPI

from app.database.base_class import Base
from app.database.session import engine
from app.api import attendance
from app.api import debug

app = FastAPI(title="Biometric Presence Verification System")

#Create tables
Base.metadata.create_all(bind=engine)
# Register routers
app.include_router(attendance.router, tags=["Attendance"])

app.include_router(debug.router)
