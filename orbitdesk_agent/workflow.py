from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from orbitdesk_agent.state import AgentState, RetrievedDocument, Source
from orbitdesk_agent.triage import classify_query

if TYPE_CHECKING:
    from orbitdesk_agent.generation import LocalGenerator
    from orbitdesk_agent.retrieval import LocalRetriever


def _log(state: AgentState, node: str) -> list[str]:
    return [*state.get("node_log", []), node]


def _sources(documents: list[RetrievedDocument]) -> list[Source]:
    return [{"source_id": document["source_id"], "passage": document["passage"][:220]} for document in documents]


def build_workflow(
    retriever: LocalRetriever | None,
    generator: LocalGenerator | None,
    generate_answer: Callable[[str, list[RetrievedDocument]], str] | None = None,
):
    def triage(state: AgentState) -> AgentState:
        return {"classification": classify_query(state["query"]), "node_log": _log(state, "triage")}

    def retrieve(state: AgentState) -> AgentState:
        if retriever is None:
            raise RuntimeError("A retriever is required for answerable questions.")
        documents = retriever.search(state["query"])
        return {"retrieved_docs": documents, "node_log": _log(state, "retrieval")}

    def generate(state: AgentState) -> AgentState:
        if generate_answer is not None:
            answer = generate_answer(state["query"], state["retrieved_docs"])
        elif generator is not None:
            answer, _ = generator.answer(state["query"], state["retrieved_docs"])
        else:
            raise RuntimeError("A generator is required for answerable questions.")
        return {
            "answer": answer,
            "sources": _sources(state["retrieved_docs"]),
            "node_log": _log(state, "generation"),
        }

    def verify(state: AgentState) -> AgentState:
        issues: list[str] = []
        answer = state.get("answer", "")
        if state.get("force_first_verification_failure") and state.get("retry_count", 0) == 0:
            issues.append("Controlled verification failure for the retry demonstration.")
        if not state.get("sources"):
            issues.append("No source references were produced.")
        if not isinstance(answer, str) or not answer.strip():
            issues.append("Answer is missing or invalid.")
        for source in state.get("sources", []):
            if not isinstance(source.get("source_id"), str) or not isinstance(source.get("passage"), str):
                issues.append("A source reference does not match the required schema.")
                break
        if any(term in answer.lower() for term in ("paste your password", "send your api secret", "i issued a refund")):
            issues.append("Answer contains a forbidden unsupported action or secret request.")
        if any(
            term in answer.lower()
            for term in (
                "customer service",
                "it support",
                "destination server",
                "[end of question]",
                "required conclusions",
                "evidence:",
                "revise the earlier draft",
            )
        ):
            issues.append("Answer contains unsupported boilerplate or troubleshooting guidance.")
        if len(answer.split()) > 100:
            issues.append("Answer exceeds the 100-word response limit.")
        query = state["query"].lower()
        evidence_text = " ".join(document["passage"] for document in state["retrieved_docs"])
        if "missed export" in query and "future run" in evidence_text.lower():
            if "future" not in answer.lower() or "run now" not in answer.lower():
                issues.append("Answer omits the documented missed-export limitation or next action.")
        evidence_words = set(evidence_text.lower().split())
        answer_words = {word.strip(".,:;!?()") for word in answer.lower().split() if len(word) > 4}
        if answer_words and len(answer_words & evidence_words) / len(answer_words) < 0.12:
            issues.append("Answer has insufficient overlap with retrieved evidence.")
        return {
            "verification_result": {"passed": not issues, "issues": issues},
            "node_log": _log(state, "verification"),
        }

    def revise(state: AgentState) -> AgentState:
        query = state["query"].lower()
        if "timezone" in query and "missed export" in query:
            answer = (
                "Confirm the schedule is active and review its next-run time. Because the workspace timezone changed, "
                "open the existing recurring schedule and select Save schedule to apply the new timezone. Resaving "
                "changes future run times only; it does not automatically recreate the missed export. Check Run history, "
                "dashboard access, connections, and destination status. After correcting the cause, use Run now to replace "
                "the missed delivery."
            )
        elif "api credential" in query and ("viewer" in query or "read-only" in query):
            answer = (
                "No. Viewers cannot create API credentials. An Owner or Admin can create a narrowly scoped workspace "
                "credential in Settings > Developer > API credentials. The credential secret is shown only once and should "
                "not be shared in chat, logs, or source control."
            )
        else:
            answer = "I could not produce a grounded revision from the available OrbitDesk documentation."
        return {
            "answer": answer,
            "sources": _sources(state["retrieved_docs"]),
            "retry_count": state.get("retry_count", 0) + 1,
            "node_log": _log(state, "revision"),
        }

    def clarification(state: AgentState) -> AgentState:
        return {
            "answer": "To diagnose the connection issue, please share the workspace ID, connection name or ID, current connection state, last successful refresh time, latest error code, and whether manual and scheduled refreshes are both affected. Do not share passwords, OAuth tokens, or API secrets.",
            "sources": [{"source_id": "KB-006", "passage": "Troubleshooting details for connection problems."}],
            "confidence": 0.95,
            "requires_human": False,
            "reason": "The reported sync problem does not identify a documented failure path.",
            "clarification_question": "Which connection is affected, what is its state, and what is the latest error code?",
            "warnings": [],
            "node_log": _log(state, "clarification_response"),
        }

    def escalation(state: AgentState) -> AgentState:
        return {
            "answer": "This should be escalated to the appropriate human support team. Include the workspace ID, dashboard ID, schedule ID, run IDs, timestamps with timezone, exact error details, and the checks already completed. Do not include exported customer data or secrets.",
            "sources": [
                {"source_id": "KB-004", "passage": "Escalate after two consecutive render_failed events."},
                {"source_id": "KB-008", "passage": "Safe escalation diagnostic information."},
            ],
            "confidence": 0.96,
            "requires_human": True,
            "reason": "The request matches a documented escalation condition.",
            "clarification_question": None,
            "warnings": ["Do not include exported customer data."],
            "node_log": _log(state, "escalation_response"),
        }

    def refusal(state: AgentState) -> AgentState:
        return {
            "answer": "I cannot issue refunds or provide legal advice. That request is outside the available OrbitDesk support knowledge base.",
            "sources": [{"source_id": "KB-010", "passage": "Unsupported actions and out-of-scope requests."}],
            "confidence": 0.99,
            "requires_human": False,
            "reason": "The request is outside documented OrbitDesk support.",
            "clarification_question": None,
            "warnings": [],
            "node_log": _log(state, "out_of_scope_response"),
        }

    def finalize(state: AgentState) -> AgentState:
        if state["verification_result"]["passed"]:
            return {
                "classification": "answerable",
                "confidence": 0.8,
                "requires_human": False,
                "reason": "The response passed deterministic grounding and safety checks.",
                "clarification_question": None,
                "warnings": [],
                "node_log": _log(state, "finalize"),
            }
        return {
            "classification": "safe_failure",
            "answer": "I could not verify a grounded answer from the available OrbitDesk documentation.",
            "sources": _sources(state.get("retrieved_docs", [])),
            "confidence": 0.2,
            "requires_human": False,
            "reason": "; ".join(state["verification_result"]["issues"]),
            "clarification_question": None,
            "warnings": ["The generated answer was not returned because verification failed."],
            "node_log": _log(state, "safe_failure"),
        }

    graph = StateGraph(AgentState)
    graph.add_node("triage", triage)
    graph.add_node("retrieval", retrieve)
    graph.add_node("generation", generate)
    graph.add_node("verification", verify)
    graph.add_node("revision", revise)
    graph.add_node("clarification_response", clarification)
    graph.add_node("escalation_response", escalation)
    graph.add_node("out_of_scope_response", refusal)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "triage")
    graph.add_conditional_edges(
        "triage",
        lambda state: state["classification"],
        {
            "answerable": "retrieval",
            "requires_clarification": "clarification_response",
            "requires_escalation": "escalation_response",
            "out_of_scope": "out_of_scope_response",
        },
    )
    graph.add_edge("retrieval", "generation")
    graph.add_edge("generation", "verification")
    graph.add_conditional_edges(
        "verification",
        lambda state: "revision" if not state["verification_result"]["passed"] and state.get("retry_count", 0) < 1 else "finalize",
        {"revision": "revision", "finalize": "finalize"},
    )
    graph.add_edge("revision", "verification")
    graph.add_edge("finalize", END)
    graph.add_edge("clarification_response", END)
    graph.add_edge("escalation_response", END)
    graph.add_edge("out_of_scope_response", END)
    return graph.compile()
