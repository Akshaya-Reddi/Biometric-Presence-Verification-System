from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database.base import Base

DATABASE_URL = "sqlite:///./biometric.db"

engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_timeout=30,
    pool_recycle=1800,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

def init_db():
    Base.metadata.create_all(bind=engine)
