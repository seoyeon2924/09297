"""
ChromaDB 데이터 폴더 삭제 + SQLite 문서/청크 테이블 초기화.
실행: 프로젝트 루트에서 가상환경 활성화 후
      python scripts/reset_docs_and_chroma.py
"""

import shutil
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings
from storage.database import SessionLocal, init_db
from storage.models import Chunk, ReferenceDocument


def main() -> None:
    # 1. ChromaDB 디렉터리 삭제 (서버가 켜져 있으면 실패 → 서버 끈 뒤 다시 실행)
    chroma_path = Path(settings.CHROMA_PERSIST_DIR)
    if chroma_path.exists():
        try:
            shutil.rmtree(chroma_path)
            print(f"[1] ChromaDB 삭제 완료: {chroma_path}")
        except PermissionError:
            print(f"[1] ChromaDB 폴더 사용 중이라 삭제 불가. Chroma 서버를 끄고 다시 실행하거나, 서버 끈 뒤 해당 폴더를 수동 삭제하세요: {chroma_path}")
    else:
        print(f"[1] ChromaDB 경로 없음 (스킵): {chroma_path}")

    # 2. SQLite 문서/청크 초기화 (청크 먼저 - FK)
    session = SessionLocal()
    try:
        deleted_chunks = session.query(Chunk).delete()
        deleted_docs = session.query(ReferenceDocument).delete()
        session.commit()
        print(f"[2] SQLite 초기화: 문서 {deleted_docs}건, 청크 {deleted_chunks}건 삭제")
    except Exception as e:
        session.rollback()
        print(f"[2] SQLite 초기화 실패: {e}")
        raise
    finally:
        session.close()

    # 3. 테이블 존재 보장
    init_db()
    print("[3] DB 테이블 확인 완료")


if __name__ == "__main__":
    main()
