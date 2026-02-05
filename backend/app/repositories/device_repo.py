from sqlalchemy.orm import Session
from app.models.devices import Device

def is_registered_device(db: Session, device_id: str) -> bool:
    return db.query(Device).filter(Device.device_id == device_id).first() is not None
