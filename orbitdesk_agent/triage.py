from orbitdesk_agent.state import Classification


OUT_OF_SCOPE_TERMS = ("refund", "legal advice", "medical advice", "financial advice")
ESCALATION_TERMS = (
    "two consecutive render_failed",
    "two render_failed",
    "repeated connector_internal_error",
    "credential exposure",
)
VAGUE_CONNECTION_TERMS = ("sync is not working", "data sync is not working", "connection is broken")


def classify_query(query: str) -> Classification:
    normalized = query.lower()
    if any(term in normalized for term in OUT_OF_SCOPE_TERMS):
        return "out_of_scope"
    has_repeated_render_failure = "render_failed" in normalized and "two" in normalized and (
        "row" in normalized or "consecutive" in normalized
    )
    if has_repeated_render_failure or any(term in normalized for term in ESCALATION_TERMS):
        return "requires_escalation"
    if any(term in normalized for term in VAGUE_CONNECTION_TERMS):
        return "requires_clarification"
    return "answerable"
