"""
SQLite database engine and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from config import settings
from storage.models import Base

# NullPool: SQLite는 연결 풀이 불필요하고, QueuePool의 백그라운드 스레드가
# non-daemon이라 Ctrl+C 시 프로세스 종료를 막는 문제를 방지한다.
# check_same_thread=False: Streamlit 멀티스레드 환경에서 SQLite 접근 허용.
engine = create_engine(
    settings.SQLITE_URL,
    echo=False,
    poolclass=NullPool,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all tables if they do not exist yet."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Return a new database session (caller must close)."""
    return SessionLocal()
