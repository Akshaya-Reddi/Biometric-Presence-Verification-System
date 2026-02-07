from app.database.base_class import Base

# Import all models so metadata is registered
from app.models.user import User
from app.models.attendance import Attendance
from app.models.audit_log import AuditLog
from app.models.devices import Device
from app.models.confidence_log import ConfidenceLog