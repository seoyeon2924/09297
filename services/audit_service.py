"""
Audit Service — event logging and KPI helpers.
"""

from typing import Optional

from storage.repository import AuditRepository


class AuditService:

    @staticmethod
    def log_event(
        event_type: str,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        actor: Optional[str] = None,
        detail: Optional[dict] = None,
    ) -> str:
        """Log a single audit event. Returns log id."""
        return AuditRepository.create_log(
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            actor=actor,
            detail=detail,
        )

    @staticmethod
    def get_logs(
        entity_id: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        return AuditRepository.list_logs(entity_id=entity_id, limit=limit)
