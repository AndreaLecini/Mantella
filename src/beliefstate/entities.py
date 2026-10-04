"""
Closed registries for the Belief State Node.

Everything here is deliberately closed (Enum): predicates, types, source_type,
and entities (actors/items). The DAG only covers the set involved in the
experiments, closing the entities too, not just the predicates, keeps every
proposition comparable and the whole prototype controllable and testable.
Extending the domain (new NPC, new item, new predicate) is always an explicit
change to this file, never a runtime registration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class StatementType(Enum):
    FACT = "Fact"
    INTENTION = "Intention"
    RELATIONAL_STATE = "RelationalState"
    COMMITMENT = "Commitment"


class Predicate(Enum):
    POSSESSION = "POSSESSION"
    BETRAYAL = "BETRAYAL"
    TRUST = "TRUST"
    INTENTION_TO_FOLLOW = "INTENTION_TO_FOLLOW"
    COMMITMENT = "COMMITMENT"


class SourceType(Enum):
    LLM_GENERATED = "LLM_GENERATED"
    PLAYER_DIALOGUE = "PLAYER_DIALOGUE"
    ENGINE = "ENGINE"
    DERIVED_CASCADE = "DERIVED_CASCADE"


class Status(Enum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"


class CommitmentStatus(Enum):
    PENDING = "PENDING"
    FULFILLED = "FULFILLED"
    VIOLATED = "VIOLATED"


class TrustLevel(Enum):
    HOSTILE = "hostile"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CommitmentClause(Enum):
    WAIT = "WAIT"
    DELIVER_ITEM = "DELIVER_ITEM"


class Actor(Enum):
    """Actors involved in the experiments — closed set, not extensible at
    runtime. Adding an NPC means adding a line here, at scenario design time,
    not registering it dynamically."""
    AGNIS = "Agnis"
    PLAYER = "Player"
    LYDIA = "Lydia"
    AELA = "Aela"
    BELETHOR = "Belethor"
    JARL_BALGRUUF = "Jarl_Balgruuf"


class Item(Enum):
    """Items involved in the experiments — same principle as `Actor`."""
    ANCESTRAL_SWORD = "Ancestral_Sword"
    IRON_AXE = "Iron_Axe"
    GOLDEN_RING = "Golden_Ring"
    DAGGER = "Dagger"


PREDICATE_TYPE: dict[Predicate, StatementType] = {
    Predicate.POSSESSION: StatementType.FACT,
    Predicate.BETRAYAL: StatementType.FACT,
    Predicate.TRUST: StatementType.RELATIONAL_STATE,
    Predicate.INTENTION_TO_FOLLOW: StatementType.INTENTION,
    Predicate.COMMITMENT: StatementType.COMMITMENT,
}


PREDICATE_VALUE_DOMAIN: dict[Predicate, type] = {
    Predicate.POSSESSION: bool,
    Predicate.BETRAYAL: bool,
    Predicate.TRUST: TrustLevel,
    Predicate.INTENTION_TO_FOLLOW: bool,
    Predicate.COMMITMENT: CommitmentClause,
}


PREDICATE_ENTITY_DOMAIN: dict[Predicate, tuple[type, type]] = {
    Predicate.POSSESSION: (Actor, Item),
    Predicate.BETRAYAL: (Actor, Actor),
    Predicate.TRUST: (Actor, Actor),
    Predicate.INTENTION_TO_FOLLOW: (Actor, Actor),
    Predicate.COMMITMENT: (Actor, Actor),
}


SOURCE_TYPE_TIER: dict[SourceType, int] = {
    SourceType.LLM_GENERATED: 0,
    SourceType.PLAYER_DIALOGUE: 1,
    SourceType.ENGINE: 2,
    SourceType.DERIVED_CASCADE: 3,
}


@dataclass(frozen=True)
class Rule:
    """table of bindings. `bind` computes the (subject, object) of the
    target proposition from the trigger proposition that satisfied the
    condition — for the three rules in the spec this is always a subject/
    object swap, but kept as a function so future rules aren't restricted to
    that binding."""

    name: str
    trigger_predicate: Predicate
    trigger_condition: Callable[["Statement"], bool]  
    target_predicate: Predicate
    target_value: object
    bind: Callable[["Statement"], tuple[object, object]]  


def _swap_subject_object(trigger: "Statement") -> tuple[object, object]:  
    return trigger.object, trigger.subject


RULES: list[Rule] = [
    Rule(
        name="R1_betrayal_invalidates_trust",
        trigger_predicate=Predicate.BETRAYAL,
        trigger_condition=lambda s: s.value is True,
        target_predicate=Predicate.TRUST,
        target_value=TrustLevel.LOW,
        bind=_swap_subject_object,
    ),
    Rule(
        name="R2_betrayal_invalidates_intention",
        trigger_predicate=Predicate.BETRAYAL,
        trigger_condition=lambda s: s.value is True,
        target_predicate=Predicate.INTENTION_TO_FOLLOW,
        target_value=False,
        bind=_swap_subject_object,
    ),
    Rule(
        name="R3_violated_commitment_invalidates_trust",
        trigger_predicate=Predicate.COMMITMENT,
        trigger_condition=lambda s: s.commitment_status == CommitmentStatus.VIOLATED,
        target_predicate=Predicate.TRUST,
        target_value=TrustLevel.LOW,
        bind=_swap_subject_object,
    ),
]


_trigger_predicates = {r.trigger_predicate for r in RULES}
_target_predicates = {r.target_predicate for r in RULES}
assert _trigger_predicates.isdisjoint(_target_predicates), (
    "Rule table is not acyclic: a target predicate also appears as a "
    "trigger. Design constraint violated (§5)."
)
