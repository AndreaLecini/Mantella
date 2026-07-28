"""
The engine: resolve(), insert(), propagate(), transition_commitment(),
insert_or_transition().

No periodic scan, no state beyond the DAG itself: every call to insert() is a
single, deterministic step (§4, §7 of the spec).
"""

from __future__ import annotations

import dataclasses

from .dag import BeliefStateDAG, Statement, create_statement
from .entities import (
    RULES,
    SOURCE_TYPE_TIER,
    CommitmentStatus,
    Predicate,
    SourceType,
    Status,
    StatementType,
)


def resolve(old: Statement, new: Statement) -> Statement:
    """Resolution procedure (§6): source_type tier, then recency.
    Pure function — mutates nothing, only returns the winner."""
    old_tier = SOURCE_TYPE_TIER[old.source_type]
    new_tier = SOURCE_TYPE_TIER[new.source_type]
    if new_tier != old_tier:
        return new if new_tier > old_tier else old
    # same tier -> tie-break on created_at (never on insertion order)
    return new if new.created_at >= old.created_at else old


def insert(dag: BeliefStateDAG, candidate: Statement) -> Statement:
    """Inserts `candidate` into the DAG. Returns the proposition that ends up
    ACTIVE for its key after insertion (may be `candidate` itself, the
    pre-existing proposition if it won, or the same pre-existing proposition
    if the value was identical — see the note on idempotency below).

    Raises ValueError if subject/object are not instances of the Enum type
    expected by the predicate (Actor/Item, closed set — see entities.py) —
    validation happens here, at entry, not at Statement creation:
    create_statement() stays a pure operation.
    """
    dag.validate_entities(candidate)

    existing = dag.active_for_key(candidate.key)

    if existing is None:
        candidate.status = Status.ACTIVE
        dag.register(candidate)
        propagate(dag, candidate)
        return candidate

    if existing.value == candidate.value:
        # NOTE: correction relative to the spec's original pseudocode, which
        # in this branch assigned ACTIVE to the new node regardless, without
        # explicitly superseding the old one — violating the "at most one
        # ACTIVE node per key" invariant (two identical nodes would both stay
        # ACTIVE). Here the redundant candidate is not activated: it is
        # logged for audit but the active proposition remains the existing
        # one.
        candidate.status = Status.SUPERSEDED
        dag.register(candidate)
        return existing

    # different value, same key -> direct conflict, apply §6
    winner = resolve(existing, candidate)
    if winner is candidate:
        existing.status = Status.SUPERSEDED
        candidate.status = Status.ACTIVE
        dag.register(existing)
        dag.register(candidate)
        propagate(dag, candidate)
        return candidate
    else:
        candidate.status = Status.SUPERSEDED
        dag.register(candidate)  # persisted for audit, never silently discarded
        return existing


def propagate(dag: BeliefStateDAG, trigger: Statement) -> list[Statement]:
    """Applies the rule table (§5) to the proposition that just became
    ACTIVE. For each satisfied rule, generates the derived proposition and
    inserts it recursively (the same insert() function, no separate
    mechanism for cascades)."""
    results = []
    for rule in RULES:
        if trigger.predicate != rule.trigger_predicate:
            continue
        if not rule.trigger_condition(trigger):
            continue

        target_subject, target_object = rule.bind(trigger)
        candidate = create_statement(
            predicate=rule.target_predicate,
            subject=target_subject,
            object=target_object,
            value=rule.target_value,
            source_type=SourceType.DERIVED_CASCADE,
            created_at=trigger.created_at,
        )
        # depends_on edge: relation between two nodes, registered on the
        # graph (not as a field of `candidate`) — before insertion, as per §7.
        dag.register_depends_on_edge(candidate.id, trigger.id)
        results.append(insert(dag, candidate))
    return results


def transition_commitment(
    dag: BeliefStateDAG,
    commitment: Statement,
    new_status: CommitmentStatus,
    source_type: SourceType,
    created_at: float,
) -> None:
    """In-place state transition (§3, §7): does NOT generate a new node,
    mutates the same `id`. Propagation only happens after the mutation
    (rules conditioned on commitment_status are evaluated here)."""
    if commitment.type != StatementType.COMMITMENT:
        raise ValueError("transition_commitment requires a Commitment proposition")

    commitment.commitment_status = new_status
    commitment.source_type = source_type
    commitment.created_at = created_at
    dag.register(commitment)
    propagate(dag, commitment)


def insert_or_transition(dag: BeliefStateDAG, claim: Statement) -> Statement:
    """Applies one extracted claim to `dag` - the shared entry point for both
    the player-side pipeline (Conversation.__extract_and_apply_beliefs) and
    the NPC-side one (ActionVerifier): a COMMITMENT claim that names the same
    clause as an already-ACTIVE commitment (same key, same value - eg still
    "promised to wait") but a different commitment_status (eg PENDING ->
    FULFILLED) describes a status transition of that existing commitment, not
    a new/conflicting proposition. insert() can't express that on its own:
    its idempotency check only compares .value, so a same-clause claim would
    be treated as a redundant duplicate and dropped - commitment_status would
    never change. transition_commitment() exists specifically for this
    (in-place mutation + re-propagation); this function is what decides when
    to use it instead of a normal insert().

    Always inserts a copy (dataclasses.replace()), never `claim` itself: both
    callers may need to apply the same extracted claim to more than one NPC's
    DAG, and insert()/propagate() mutate .status/.commitment_status in place
    - sharing one Statement object across DAGs would let a mutation meant for
    one NPC's belief state leak into another's.
    """
    existing = dag.active_for_key(claim.key)
    is_commitment_status_change = (
        claim.predicate == Predicate.COMMITMENT
        and existing is not None
        and existing.value == claim.value
        and existing.commitment_status != claim.commitment_status
    )
    if is_commitment_status_change:
        transition_commitment(dag, existing, claim.commitment_status, claim.source_type, claim.created_at)
        return existing
    return insert(dag, dataclasses.replace(claim))
