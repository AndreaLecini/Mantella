"""
ActionVerifier, checks the NPC's own generated dialogue against its belief
state, the way ClaimExtractor + Conversation.__extract_and_apply_beliefs
check the player's.

Mirrors ClaimExtractor's "text in, candidates out" extraction (via
ClaimExtractor.extract_npc_claims(), every claim tagged SourceType.LLM_GENERATED
— the lowest source tier, see entities.SOURCE_TYPE_TIER), but does its own
DAG-aware filtering on the way in: a claim that conflicts with an already-
ACTIVE statement for the same key (same identity, different value) is never
inserted here — not even to lose a normal insert()/resolve() tier tie-break,
which an LLM_GENERATED-tier claim almost always would anyway. It's counted
and logged instead. The point of this stage is to measure how often the
NPC's own narration contradicts what it's already established to believe,
which a silent tier-based resolution would hide from view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src import utils
from src.character_manager import Character

from src.beliefstate.dag import BeliefStateDAG, Statement
from src.beliefstate.engine import insert_or_transition
from src.beliefstate.extraction import ClaimExtractor

logger = utils.get_logger()


@dataclass
class VerificationResult:
    """The outcome of verifying one piece of NPC dialogue."""
    total_claims: int
    conflicting_claims: int
    accepted: list[Statement] = field(default_factory=list)

    @property
    def conflict_rate(self) -> float | None:
    
        return self.conflicting_claims / self.total_claims if self.total_claims else None


class ActionVerifier:


    def __init__(self, claim_extractor: ClaimExtractor) -> None:
        self.__claim_extractor = claim_extractor
        self.__session_total_claims = 0
        self.__session_total_conflicts = 0

    @utils.time_it
    def verify(
        self,
        speaker: Character,
        npc_text: str,
        dag: BeliefStateDAG,
        involved_characters: list[Character],
        created_at: float,
    ) -> VerificationResult:
        
        claims = self.__claim_extractor.extract_npc_claims(speaker, npc_text, involved_characters, created_at)

        accepted: list[Statement] = []
        conflicts = 0
        for claim in claims:
            existing = dag.active_for_key(claim.key)
            is_conflict = existing is not None and existing.value != claim.value
            if is_conflict:
                conflicts += 1
                logger.warning(
                    f"Action Verifier: {speaker.name} said something that contradicts an established belief - "
                    f"claim {claim.predicate.value}({claim.subject.value}, {claim.object.value})={claim.value!r} "
                    f"conflicts with existing value {existing.value!r} (source {existing.source_type.value}). "
                    f"Rejected, not inserted."
                )
                continue
            insert_or_transition(dag, claim)
            accepted.append(claim)

        self.__session_total_claims += len(claims)
        self.__session_total_conflicts += conflicts

        if claims:
            logger.log(28, (
                f"Action Verifier: {conflicts}/{len(claims)} claims conflicted this turn "
                f"(session total: {self.__session_total_conflicts}/{self.__session_total_claims})"
            ))

        return VerificationResult(total_claims=len(claims), conflicting_claims=conflicts, accepted=accepted)

    @property
    def session_conflict_rate(self) -> float | None:
        
        return self.__session_total_conflicts / self.__session_total_claims if self.__session_total_claims else None
