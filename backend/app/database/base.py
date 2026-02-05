from sqlalchemy.orm import declarative_base

from app.models.attendance import Attendance
from app.models.audit_log import AuditLog
from app.models.devices import Device   # ← MUST exist

Base = declarative_base()
