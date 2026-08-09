from orbitdesk_agent.state import RetrievedDocument
from orbitdesk_agent.workflow import build_workflow


class StubRetriever:
    def search(self, _query: str) -> list[RetrievedDocument]:
        return [
            {
                "source_id": "KB-003",
                "title": "Workspace Settings and Timezones",
                "passage": "Open the schedule, review next-run time, and save the schedule. Resaving changes future run times only.",
                "score": 0.9,
            },
            {
                "source_id": "KB-004",
                "title": "Scheduled Exports",
                "passage": "Use Run now after correcting the cause. A manual run does not alter the next-run time.",
                "score": 0.8,
            },
        ]


def stub_generator(_query: str, _documents: list[RetrievedDocument]) -> str:
    return "Open the schedule, review the next-run time, and save the schedule."


def run(query: str, force_failure: bool = False):
    workflow = build_workflow(StubRetriever(), None, stub_generator)
    return workflow.invoke(
        {
            "query": query,
            "retry_count": 0,
            "node_log": [],
            "force_first_verification_failure": force_failure,
        }
    )


def test_clarification_route_skips_retrieval_and_generation():
    result = run("Our data sync is not working. Can you tell me how to fix it?")
    assert result["classification"] == "requires_clarification"
    assert result["node_log"] == ["triage", "clarification_response"]


def test_answerable_route_retrieves_evidence_and_finalizes():
    result = run("What should I check after changing the workspace timezone?")
    assert result["classification"] == "answerable"
    assert [source["source_id"] for source in result["sources"]] == ["KB-003", "KB-004"]
    assert result["node_log"] == ["triage", "retrieval", "generation", "verification", "finalize"]


def test_escalation_route_returns_human_handoff():
    result = run("Two export runs in a row failed with render_failed for the same dashboard.")
    assert result["classification"] == "requires_escalation"
    assert result["requires_human"] is True
    assert result["node_log"] == ["triage", "escalation_response"]


def test_out_of_scope_route_refuses_refund_request():
    result = run("Issue a refund and write legal advice.")
    assert result["classification"] == "out_of_scope"
    assert result["node_log"] == ["triage", "out_of_scope_response"]


def test_verification_failure_retries_once_then_succeeds():
    result = run("How can a missed export be recovered after a timezone change?", force_failure=True)
    assert result["classification"] == "answerable"
    assert result["retry_count"] == 1
    assert result["node_log"] == [
        "triage",
        "retrieval",
        "generation",
        "verification",
        "revision",
        "verification",
        "finalize",
    ]
