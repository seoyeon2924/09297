"""
Review Service — review request CRUD and status transitions.
"""

from storage.repository import ReviewRepository, AuditRepository


class ReviewService:

    @staticmethod
    def create_request(
        product_name: str,
        category: str,
        broadcast_type: str,
        requested_by: str,
        items: list[dict],
    ) -> dict:
        result = ReviewRepository.create_request(
            product_name=product_name,
            category=category,
            broadcast_type=broadcast_type,
            requested_by=requested_by,
            items=items,
        )
        AuditRepository.create_log(
            event_type="REQUEST_CREATE",
            entity_type="ReviewRequest",
            entity_id=result["id"],
            actor=requested_by,
            detail={
                "product_name": product_name,
                "item_count": len(items),
            },
        )
        return result

    @staticmethod
    def list_requests(status_filter: str | None = None) -> list[dict]:
        return ReviewRepository.list_requests(status_filter=status_filter)

    @staticmethod
    def get_detail(request_id: str) -> dict | None:
        return ReviewRepository.get_detail(request_id)

    @staticmethod
    def submit_decision(
        request_id: str,
        decision: str,
        comment: str,
        decided_by: str,
    ) -> dict:
        result = ReviewRepository.create_human_decision(
            request_id=request_id,
            decision=decision,
            comment=comment,
            decided_by=decided_by,
        )
        AuditRepository.create_log(
            event_type="HUMAN_DECIDE",
            entity_type="ReviewRequest",
            entity_id=request_id,
            actor=decided_by,
            detail={"decision": decision},
        )
        return result
