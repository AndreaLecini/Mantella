"""
Tests for ActionVerifier: extracts claims from the NPC's own dialogue and
applies each one to the DAG via engine.insert_or_transition() - UNLESS it
conflicts with an already-ACTIVE statement for the same key (same identity,
different value), in which case it must be logged and counted, never
inserted, and reflected in the returned conflict ratio.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.beliefstate import Actor, Item, Predicate, SourceType, TrustLevel, create_statement
from src.beliefstate.dag import BeliefStateDAG
from src.beliefstate.engine import insert
from src.beliefstate.extraction import ClaimExtractor
from src.beliefstate.verifier import ActionVerifier, VerificationResult
from src.character_manager import Character
from src.config.config_loader import ConfigLoader
from src.games.equipment import Equipment
from src.llm.client_base import ClientBase


def make_npc(name: str, ref_id: str = "000001") -> Character:
    return Character(
        base_id=ref_id,
        ref_id=ref_id,
        name=name,
        gender=0,
        race="[Race <NordRace (00013746)>]",
        is_player_character=False,
        bio="",
        is_in_combat=False,
        is_enemy=False,
        relationship_rank=0,
        is_generic_npc=False,
        ingame_voice_model="MaleEvenToned",
        tts_voice_model="MaleEvenToned",
        csv_in_game_voice_model="MaleEvenToned",
        advanced_voice_model="MaleEvenToned",
        voice_accent="en",
        equipment=Equipment({}),
        custom_character_values=None,
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    return MagicMock(spec=ClientBase)


@pytest.fixture
def extractor(mock_llm_client: MagicMock, default_config: ConfigLoader) -> ClaimExtractor:
    return ClaimExtractor(mock_llm_client, default_config)


@pytest.fixture
def verifier(extractor: ClaimExtractor) -> ActionVerifier:
    return ActionVerifier(extractor)


@pytest.fixture
def agnis() -> Character:
    return make_npc("Agnis")


@pytest.fixture
def dag() -> BeliefStateDAG:
    return BeliefStateDAG(npc="Agnis")


class TestVerificationResult:
    def test_conflict_rate_computed(self):
        result = VerificationResult(total_claims=4, conflicting_claims=1)
        assert result.conflict_rate == 0.25

    def test_conflict_rate_none_when_nothing_extracted(self):
        result = VerificationResult(total_claims=0, conflicting_claims=0)
        assert result.conflict_rate is None


class TestActionVerifierNoConflict:
    def test_non_conflicting_claim_is_inserted(self, verifier: ActionVerifier, mock_llm_client: MagicMock, agnis: Character, dag: BeliefStateDAG):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "POSSESSION", "subject": "Agnis", "object": "Ancestral_Sword", "value": True}
        ])

        result = verifier.verify(agnis, "I still have the sword.", dag, [agnis], created_at=1.0)

        assert result.total_claims == 1
        assert result.conflicting_claims == 0
        assert result.conflict_rate == 0.0
        assert len(result.accepted) == 1
        active = dag.active_for_key(result.accepted[0].key)
        assert active is not None
        assert active.value is True
        assert active.source_type == SourceType.LLM_GENERATED

    def test_brand_new_commitment_is_inserted(self, verifier: ActionVerifier, mock_llm_client: MagicMock, agnis: Character, dag: BeliefStateDAG):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "COMMITMENT", "subject": "Agnis", "object": "Player", "value": "WAIT"}
        ])

        result = verifier.verify(agnis, "I promise to wait here.", dag, [agnis], created_at=1.0)

        assert result.conflicting_claims == 0
        assert len(dag.active()) == 1


class TestActionVerifierConflict:
    def test_conflicting_claim_is_rejected_not_inserted(self, verifier: ActionVerifier, mock_llm_client: MagicMock, agnis: Character, dag: BeliefStateDAG):
        """Existing belief: Agnis owns the sword. The NPC then says (via
        generated dialogue) that it does NOT own the sword - a direct
        contradiction that must be rejected, not silently resolved by tier."""
        existing = create_statement(Predicate.POSSESSION, Actor.AGNIS, Item.ANCESTRAL_SWORD, True, SourceType.ENGINE, created_at=0.5)
        insert(dag, existing)

        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "POSSESSION", "subject": "Agnis", "object": "Ancestral_Sword", "value": False}
        ])

        result = verifier.verify(agnis, "I no longer have the sword.", dag, [agnis], created_at=1.0)

        assert result.total_claims == 1
        assert result.conflicting_claims == 1
        assert result.conflict_rate == 1.0
        assert result.accepted == []
        # the original belief must be untouched - not overwritten, not superseded
        active = dag.active_for_key(existing.key)
        assert active.value is True
        assert active.id == existing.id

    def test_mixed_conflicting_and_non_conflicting_claims_ratio(self, verifier: ActionVerifier, mock_llm_client: MagicMock, agnis: Character, dag: BeliefStateDAG):
        existing = create_statement(Predicate.TRUST, Actor.AGNIS, Actor.PLAYER, TrustLevel.LOW, SourceType.PLAYER_DIALOGUE, created_at=0.5)
        insert(dag, existing)

        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "TRUST", "subject": "Agnis", "object": "Player", "value": "high"},  # conflicts (low != high)
            {"predicate": "POSSESSION", "subject": "Agnis", "object": "Ancestral_Sword", "value": True},  # no existing claim -> no conflict
        ])

        result = verifier.verify(agnis, "I trust you deeply, and I have the sword.", dag, [agnis], created_at=1.0)

        assert result.total_claims == 2
        assert result.conflicting_claims == 1
        assert result.conflict_rate == 0.5
        assert len(result.accepted) == 1
        assert result.accepted[0].predicate == Predicate.POSSESSION
        # trust must remain LOW - the conflicting claim was rejected
        assert dag.active_for_key(existing.key).value == TrustLevel.LOW

    def test_commitment_status_transition_is_not_a_conflict(self, verifier: ActionVerifier, mock_llm_client: MagicMock, agnis: Character, dag: BeliefStateDAG):
        """Same clause, different commitment_status -> a legitimate
        transition (engine.insert_or_transition), not a contradiction."""
        from src.beliefstate.entities import CommitmentClause, CommitmentStatus

        commitment = create_statement(
            Predicate.COMMITMENT, Actor.AGNIS, Actor.PLAYER, CommitmentClause.WAIT,
            SourceType.PLAYER_DIALOGUE, created_at=0.5,
        )
        insert(dag, commitment)

        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "COMMITMENT", "subject": "Agnis", "object": "Player", "value": "WAIT", "commitment_status": "FULFILLED"}
        ])

        result = verifier.verify(agnis, "I have waited as promised.", dag, [agnis], created_at=1.0)

        assert result.conflicting_claims == 0
        assert len(result.accepted) == 1
        active = dag.active_for_key(commitment.key)
        assert active.commitment_status == CommitmentStatus.FULFILLED
        assert active.id == commitment.id  # transitioned in place, not a new node


class TestActionVerifierSessionTracking:
    def test_session_totals_accumulate_across_calls(self, verifier: ActionVerifier, mock_llm_client: MagicMock, agnis: Character, dag: BeliefStateDAG):
        existing = create_statement(Predicate.TRUST, Actor.AGNIS, Actor.PLAYER, TrustLevel.LOW, SourceType.PLAYER_DIALOGUE, created_at=0.5)
        insert(dag, existing)

        assert verifier.session_conflict_rate is None

        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "TRUST", "subject": "Agnis", "object": "Player", "value": "high"}
        ])
        verifier.verify(agnis, "I trust you.", dag, [agnis], created_at=1.0)
        assert verifier.session_conflict_rate == 1.0

        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "POSSESSION", "subject": "Agnis", "object": "Ancestral_Sword", "value": True}
        ])
        verifier.verify(agnis, "I have the sword.", dag, [agnis], created_at=2.0)
        # session total: 1 conflict out of 2 claims across both calls
        assert verifier.session_conflict_rate == 0.5

    def test_empty_extraction_does_not_affect_session_rate(self, verifier: ActionVerifier, mock_llm_client: MagicMock, agnis: Character, dag: BeliefStateDAG):
        mock_llm_client.request_call_with_overridden_params.return_value = "[]"

        result = verifier.verify(agnis, "Nice weather today.", dag, [agnis], created_at=1.0)

        assert result.total_claims == 0
        assert result.conflict_rate is None
        assert verifier.session_conflict_rate is None
