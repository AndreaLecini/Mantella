"""
The graph node (Statement) and the container that holds it (BeliefStateDAG).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from .entities import (
    CommitmentStatus,
    PREDICATE_ENTITY_DOMAIN,
    PREDICATE_TYPE,
    PREDICATE_VALUE_DOMAIN,
    Predicate,
    SourceType,
    Status,
    StatementType,
)

# Process-local, starts at 1 every run - a fresh process has no memory of ids
# already handed out by a previous one. A DAG loaded from a file saved by an
# earlier process can therefore contain ids ("S00001", ...) this process is
# about to generate again. ensure_id_counter_past() (called by
# serialization.deserialize(), the one place ids from outside this process
# enter memory) guards against that: it's the only thing allowed to advance
# this counter other than _next_id(), so a fresh process always starts past
# whatever it has ever loaded, never colliding with a persisted id.
_next_id_value = 1


def _next_id() -> str:
    global _next_id_value
    generated = f"S{_next_id_value:05d}"
    _next_id_value += 1
    return generated


def ensure_id_counter_past(existing_ids: Iterable[str]) -> None:
    """Advances the id counter so it never hands out an id already present in
    `existing_ids`. Must be called with every id a DAG had *before* this
    process touched it (ie right after loading from a file) - otherwise the
    first statement this process creates can silently collide with one
    already active in the loaded DAG, corrupting BeliefStateDAG._by_id (see
    the id-collision bug this was written to fix: a stale entry survives in
    _active_by_key while _by_id[id] gets silently overwritten by the new,
    unrelated statement - one proposition vanishes from active() and another
    appears to be duplicated)."""
    global _next_id_value
    for existing_id in existing_ids:
        if existing_id.startswith("S") and existing_id[1:].isdigit():
            _next_id_value = max(_next_id_value, int(existing_id[1:]) + 1)


@dataclass
class Statement:
    """A proposition — the node of the DAG. Fields consistent with spec §3."""

    id: str
    type: StatementType
    predicate: Predicate
    subject: object  # instance of Actor or Item, never a free string
    object: object    # instance of Actor or Item, never a free string
    value: object
    status: Status
    source_type: SourceType
    created_at: float
    commitment_status: Optional[CommitmentStatus] = None

    @property
    def key(self) -> tuple[StatementType, Predicate, object, object]:
        """Identity key (§3): two propositions with the same key are in
        direct conflict if they have a different `value`."""
        return (self.type, self.predicate, self.subject, self.object)


def create_statement(
    predicate: Predicate,
    subject: object,
    object: object,
    value: object,
    source_type: SourceType,
    created_at: float,
    *,
    commitment_status: Optional[CommitmentStatus] = None,
    id: Optional[str] = None,
) -> Statement:
    """Factory that validates type/predicate/value before constructing the
    Statement. Always use this instead of instantiating `Statement` directly:
    this is where 'never free text' is guaranteed to hold at write time too."""

    stmt_type = PREDICATE_TYPE[predicate]

    value_domain = PREDICATE_VALUE_DOMAIN[predicate]
    if not isinstance(value, value_domain):
        raise ValueError(
            f"Invalid value for {predicate.value}: expected {value_domain.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        )

    if stmt_type == StatementType.COMMITMENT and commitment_status is None:
        commitment_status = CommitmentStatus.PENDING
    if stmt_type != StatementType.COMMITMENT and commitment_status is not None:
        raise ValueError("commitment_status is only allowed for type=Commitment")

    return Statement(
        id=id or _next_id(),
        type=stmt_type,
        predicate=predicate,
        subject=subject,
        object=object,
        value=value,
        status=Status.ACTIVE,  # provisional: finally decided by insert()
        commitment_status=commitment_status,
        source_type=source_type,
        created_at=created_at,
    )


class BeliefStateDAG:
    """Container for the DAG of a single NPC. Maintains:
    - the full log (every proposition ever inserted, for audit — §4);
    - an index key -> id of the proposition currently ACTIVE for that key
      (invariant: at most one id per key, by construction).
    """

    def __init__(self, npc: str) -> None:
        self.npc = npc
        self._log: list[Statement] = []
        self._active_by_key: dict[tuple, str] = {}
        self._by_id: dict[str, Statement] = {}
        # Only edge type in the spec (§4): depends_on, audit/provenance only,
        # never read by the resolver. Lives here, not on Statement — it's a
        # relation between two nodes, not an attribute of a node.
        self._depends_on_edges: list[tuple[str, str]] = []  # (cascade_id, trigger_id)

    # -- validation ---------------------------------------------------------

    def validate_entities(self, stmt: Statement) -> None:
        """Verifies that subject/object are instances of the Enum type
        expected by the predicate (§PREDICATE_ENTITY_DOMAIN) — plain
        isinstance, no state to consult: the vocabulary is closed at the
        Python type level, not populated at runtime. Called by insert()
        before accepting any write."""
        subject_type, object_type = PREDICATE_ENTITY_DOMAIN[stmt.predicate]
        if not isinstance(stmt.subject, subject_type):
            raise ValueError(
                f"Invalid subject for {stmt.predicate.value}: expected "
                f"{subject_type.__name__}, got {stmt.subject!r}"
            )
        if not isinstance(stmt.object, object_type):
            raise ValueError(
                f"Invalid object for {stmt.predicate.value}: expected "
                f"{object_type.__name__}, got {stmt.object!r}"
            )

    # -- reading -------------------------------------------------------------

    def active_for_key(self, key: tuple) -> Optional[Statement]:
        stmt_id = self._active_by_key.get(key)
        return self._by_id[stmt_id] if stmt_id else None

    def get(self, stmt_id: str) -> Statement:
        return self._by_id[stmt_id]

    def active(self) -> list[Statement]:
        return [self._by_id[i] for i in self._active_by_key.values()]

    def log(self) -> list[Statement]:
        """Full audit trail, including SUPERSEDED propositions — never
        silently discarded (explicitly verified in M3, see tests)."""
        return list(self._log)

    def trigger_of(self, cascade_id: str) -> Optional[str]:
        """Given the id of a proposition generated by cascade, returns the id
        of the trigger proposition that originated it (audit) — None if
        `cascade_id` was never generated by propagate()."""
        for c_id, t_id in reversed(self._depends_on_edges):
            if c_id == cascade_id:
                return t_id
        return None

    def depends_on_edges(self) -> list[tuple[str, str]]:
        """All depends_on edges registered so far, as (cascade_id, trigger_id)
        pairs — for inspection/full audit of the graph."""
        return list(self._depends_on_edges)

    # -- writing (used only by engine.py) -----------------------------------

    def register(self, stmt: Statement) -> None:
        if stmt not in self._log:
            self._log.append(stmt)
        self._by_id[stmt.id] = stmt
        if stmt.status == Status.ACTIVE:
            self._active_by_key[stmt.key] = stmt.id
        elif self._active_by_key.get(stmt.key) == stmt.id:
            # shouldn't happen (once ACTIVE, a statement doesn't go back to
            # SUPERSEDED without someone else taking its place for the key),
            # but remove the reference if it matches, just to be safe.
            del self._active_by_key[stmt.key]

    def register_depends_on_edge(self, cascade_id: str, trigger_id: str) -> None:
        """Registers the edge: `cascade_id` was generated by propagate() from
        proposition `trigger_id`. Never read by resolve()."""
        self._depends_on_edges.append((cascade_id, trigger_id))
