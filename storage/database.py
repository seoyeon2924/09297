"""
SQLite database engine and session management.
"""

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from config import settings
from storage.models import Base

# StaticPool: 단일 연결을 재사용하여 매 요청마다 SQLite 파일을
# 여닫는 오버헤드를 제거한다. (NullPool 대비 ~30ms/query 절감)
# check_same_thread=False: Streamlit/FastAPI 멀티스레드 환경에서 접근 허용.
engine = create_engine(
    settings.SQLITE_URL,
    echo=False,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragma(dbapi_conn, connection_record):
    """WAL 모드 + 동기 NORMAL로 쓰기 성능 향상."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()
SessionLocal = sessionmaker(bind=engine)


def init_db() -> None:
    """Create all tables if they do not exist yet."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """Return a new database session (caller must close)."""
    return SessionLocal()
