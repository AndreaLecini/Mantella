"""
Minimal prompt builder: converts the ACTIVE propositions of a DAG into a
compact text block, to be injected as a new template variable (`{belief_state}`)
alongside `{conversation_summary}` in PromptDefinitions — the same mechanism
Mantella already uses for the native summary, see Context.generate_system_message().

Deliberately simple: a predicate+value dictionary -> template. Extending it
with new predicates means adding a line here, nowhere else.
"""

from __future__ import annotations

from .dag import BeliefStateDAG, Statement
from .entities import CommitmentClause, CommitmentStatus, TrustLevel, Predicate


def _describe(stmt: Statement) -> str | None:
    

    if stmt.predicate == Predicate.POSSESSION:
        verb = "owns" if stmt.value else "no longer owns"
        return f"{stmt.subject.value} {verb} {stmt.object.value}."

    if stmt.predicate == Predicate.BETRAYAL:
        if stmt.value:
            return f"{stmt.subject.value} betrayed {stmt.object.value}."
        return None  

    if stmt.predicate == Predicate.TRUST:
        labels = {
            TrustLevel.HOSTILE: "is hostile towards",
            TrustLevel.LOW: "does not trust",
            TrustLevel.MEDIUM: "has moderate trust in",
            TrustLevel.HIGH: "deeply trusts",
        }
        return f"{stmt.subject.value} {labels[stmt.value]} {stmt.object.value}."

    if stmt.predicate == Predicate.INTENTION_TO_FOLLOW:
        verb = "is willing to follow" if stmt.value else "is not willing to follow"
        return f"{stmt.subject.value} {verb} {stmt.object.value}."

    if stmt.predicate == Predicate.COMMITMENT:
        clauses = {
            CommitmentClause.WAIT: "to wait",
            CommitmentClause.DELIVER_ITEM: "to deliver an item",
        }
        statuses = {
            CommitmentStatus.PENDING: "pending",
            CommitmentStatus.FULFILLED: "fulfilled",
            CommitmentStatus.VIOLATED: "broken",
        }
        clause_text = clauses.get(stmt.value, "a commitment")
        status_text = statuses.get(stmt.commitment_status, "")
        return f"{stmt.subject.value} promised {stmt.object.value} {clause_text} ({status_text})."

    return None


def generate_belief_state_text(dag: BeliefStateDAG) -> str:
    """The text to assign to the `{belief_state}` template variable. One line
    per ACTIVE proposition with a defined template, ordered by `created_at`
    (oldest to newest) for readability."""
    active = sorted(dag.active(), key=lambda s: s.created_at)
    lines = [line for line in (_describe(s) for s in active) if line is not None]
    return "\n".join(lines)
