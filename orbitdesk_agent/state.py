from typing import Literal, TypedDict


Classification = Literal[
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
    "safe_failure",
]


class Source(TypedDict):
    source_id: str
    passage: str


class RetrievedDocument(TypedDict):
    source_id: str
    title: str
    passage: str
    score: float


class VerificationResult(TypedDict):
    passed: bool
    issues: list[str]


class AgentState(TypedDict, total=False):
    query: str
    classification: Classification
    retrieved_docs: list[RetrievedDocument]
    answer: str
    sources: list[Source]
    confidence: float
    requires_human: bool
    reason: str
    clarification_question: str | None
    warnings: list[str]
    verification_result: VerificationResult
    retry_count: int
    node_log: list[str]
    force_first_verification_failure: bool
