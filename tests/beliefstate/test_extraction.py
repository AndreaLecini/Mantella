import json
from unittest.mock import MagicMock

import pytest

from src.beliefstate.extraction import ClaimExtractor
from src.beliefstate.entities import (
    Actor,
    CommitmentStatus,
    Predicate,
    SourceType,
    TrustLevel,
)
from src.character_manager import Character
from src.config.config_loader import ConfigLoader
from src.games.equipment import Equipment
from src.llm.client_base import ClientBase


def make_npc(name: str, ref_id: str = "000001") -> Character:
    """Minimal Character for a non-player NPC, named to match an Actor member."""
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
    """A mocked ClientBase - no real API calls are made in this test module."""
    return MagicMock(spec=ClientBase)


@pytest.fixture
def extractor(mock_llm_client: MagicMock, default_config: ConfigLoader) -> ClaimExtractor:
    return ClaimExtractor(mock_llm_client, default_config)


@pytest.fixture
def agnis() -> Character:
    return make_npc("Agnis")


class TestExtractClaimsHappyPath:
    def test_single_claim_from_player_dialogue(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "TRUST", "subject": "Agnis", "object": "Player", "value": "high", "source": "player"}
        ])

        result = extractor.extract_claims("I will always help you.", "", [agnis], created_at=10.0)

        assert len(result) == 1
        stmt = result[0]
        assert stmt.predicate == Predicate.TRUST
        assert stmt.subject == Actor.AGNIS
        assert stmt.object == Actor.PLAYER
        assert stmt.value == TrustLevel.HIGH
        assert stmt.source_type == SourceType.PLAYER_DIALOGUE
        assert stmt.created_at == 10.0

    def test_claims_tagged_by_originating_input(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        """A single extraction call handling both inputs must tag each claim
        by which input it came from - not default everything to one tier."""
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "POSSESSION", "subject": "Agnis", "object": "Ancestral_Sword", "value": True, "source": "player"},
            {"predicate": "BETRAYAL", "subject": "Player", "object": "Agnis", "value": True, "source": "game_event"},
        ])

        result = extractor.extract_claims("I gave you the sword.", "Player attacked Agnis.", [agnis], created_at=1.0)

        assert len(result) == 2
        by_predicate = {s.predicate: s for s in result}
        assert by_predicate[Predicate.POSSESSION].source_type == SourceType.PLAYER_DIALOGUE
        assert by_predicate[Predicate.BETRAYAL].source_type == SourceType.ENGINE

    def test_commitment_status_defaults_to_pending(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "COMMITMENT", "subject": "Agnis", "object": "Player", "value": "WAIT", "source": "player"}
        ])

        result = extractor.extract_claims("I promise to wait here.", "", [agnis], created_at=1.0)

        assert len(result) == 1
        assert result[0].commitment_status == CommitmentStatus.PENDING

    def test_commitment_status_explicit_value_respected(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {
                "predicate": "COMMITMENT", "subject": "Agnis", "object": "Player", "value": "WAIT",
                "source": "player", "commitment_status": "FULFILLED",
            }
        ])

        result = extractor.extract_claims("I waited as promised.", "", [agnis], created_at=1.0)

        assert result[0].commitment_status == CommitmentStatus.FULFILLED

    def test_markdown_code_fenced_response_is_parsed(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = (
            '```json\n[{"predicate": "POSSESSION", "subject": "Agnis", '
            '"object": "Ancestral_Sword", "value": true, "source": "player"}]\n```'
        )

        result = extractor.extract_claims("I gave you the sword.", "", [agnis], created_at=1.0)

        assert len(result) == 1
        assert result[0].predicate == Predicate.POSSESSION


class TestExtractClaimsClosedVocabularyEnforcement:
    def test_unknown_predicate_dropped_silently(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "FLIES", "subject": "Agnis", "object": "Player", "value": True, "source": "player"}
        ])

        result = extractor.extract_claims("Agnis can fly.", "", [agnis], created_at=1.0)

        assert result == []

    def test_actor_outside_conversation_dropped(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        """Only Agnis is involved in this conversation; a claim about Lydia
        must be dropped even though Lydia is a valid Actor in general."""
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "TRUST", "subject": "Lydia", "object": "Player", "value": "high", "source": "player"}
        ])

        result = extractor.extract_claims("Lydia trusts you deeply.", "", [agnis], created_at=1.0)

        assert result == []

    def test_invalid_value_for_predicate_dropped(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "TRUST", "subject": "Agnis", "object": "Player", "value": "extremely high", "source": "player"}
        ])

        result = extractor.extract_claims("Agnis trusts you a lot.", "", [agnis], created_at=1.0)

        assert result == []

    def test_wrong_entity_type_for_predicate_dropped(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        """POSSESSION expects (Actor, Item); an Actor as object is invalid."""
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "POSSESSION", "subject": "Agnis", "object": "Player", "value": True, "source": "player"}
        ])

        result = extractor.extract_claims("Agnis owns you.", "", [agnis], created_at=1.0)

        assert result == []

    def test_missing_source_tag_dropped(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "TRUST", "subject": "Agnis", "object": "Player", "value": "high"}
        ])

        result = extractor.extract_claims("Agnis trusts you.", "", [agnis], created_at=1.0)

        assert result == []

    def test_one_bad_claim_does_not_drop_the_batch(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "NOT_A_PREDICATE", "subject": "Agnis", "object": "Player", "value": True, "source": "player"},
            {"predicate": "TRUST", "subject": "Agnis", "object": "Player", "value": "high", "source": "player"},
        ])

        result = extractor.extract_claims("Agnis trusts you.", "", [agnis], created_at=1.0)

        assert len(result) == 1
        assert result[0].predicate == Predicate.TRUST


class TestExtractClaimsMalformedResponses:
    def test_malformed_json_returns_empty(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = "this is not json"

        result = extractor.extract_claims("Hello there.", "", [agnis], created_at=1.0)

        assert result == []

    def test_non_array_json_returns_empty(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps({"predicate": "TRUST"})

        result = extractor.extract_claims("Hello there.", "", [agnis], created_at=1.0)

        assert result == []

    def test_empty_response_returns_empty(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = None

        result = extractor.extract_claims("Hello there.", "", [agnis], created_at=1.0)

        assert result == []

    def test_llm_exception_returns_empty_not_raised(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.side_effect = Exception("connection error")

        result = extractor.extract_claims("Hello there.", "", [agnis], created_at=1.0)

        assert result == []


class TestExtractClaimsSkipsLlmCallWhenPointless:
    def test_no_involved_actors_skips_llm_call(self, extractor: ClaimExtractor, mock_llm_client: MagicMock):
        stranger = make_npc("Whiterun Guard")

        result = extractor.extract_claims("Hello there.", "", [stranger], created_at=1.0)

        assert result == []
        mock_llm_client.request_call_with_overridden_params.assert_not_called()

    def test_blank_text_skips_llm_call(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        result = extractor.extract_claims("   ", "", [agnis], created_at=1.0)

        assert result == []
        mock_llm_client.request_call_with_overridden_params.assert_not_called()

    def test_no_claims_in_response_calls_llm_once(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = "[]"

        result = extractor.extract_claims("Hello there.", "", [agnis], created_at=1.0)

        assert result == []
        mock_llm_client.request_call_with_overridden_params.assert_called_once()


class TestExtractNpcClaims:
    """extract_npc_claims() is the Action Verifier's half of extraction: same
    parsing/validation machinery as extract_claims(), but the source is
    always LLM_GENERATED (never asked of the model - one fewer required
    field it could get wrong) and the text is what the NPC itself said."""

    def test_single_claim_tagged_llm_generated(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "POSSESSION", "subject": "Agnis", "object": "Ancestral_Sword", "value": True}
        ])

        result = extractor.extract_npc_claims(agnis, "I still have the ancestral sword.", [agnis], created_at=1.0)

        assert len(result) == 1
        assert result[0].predicate == Predicate.POSSESSION
        assert result[0].subject == Actor.AGNIS
        assert result[0].source_type == SourceType.LLM_GENERATED

    def test_no_source_field_required_in_response(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        """Even if the model includes a "source" key anyway, it's ignored -
        extract_npc_claims() always stamps LLM_GENERATED regardless."""
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "TRUST", "subject": "Agnis", "object": "Player", "value": "high", "source": "player"}
        ])

        result = extractor.extract_npc_claims(agnis, "I trust you completely.", [agnis], created_at=1.0)

        assert len(result) == 1
        assert result[0].source_type == SourceType.LLM_GENERATED

    def test_unknown_speaker_skips_llm_call(self, extractor: ClaimExtractor, mock_llm_client: MagicMock):
        stranger = make_npc("Whiterun Guard")

        result = extractor.extract_npc_claims(stranger, "I have the sword.", [stranger], created_at=1.0)

        assert result == []
        mock_llm_client.request_call_with_overridden_params.assert_not_called()

    def test_blank_text_skips_llm_call(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        result = extractor.extract_npc_claims(agnis, "   ", [agnis], created_at=1.0)

        assert result == []
        mock_llm_client.request_call_with_overridden_params.assert_not_called()

    def test_closed_vocabulary_still_enforced(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
            {"predicate": "FLIES", "subject": "Agnis", "object": "Player", "value": True}
        ])

        result = extractor.extract_npc_claims(agnis, "I can fly.", [agnis], created_at=1.0)

        assert result == []

    def test_malformed_response_returns_empty(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.return_value = "not json"

        result = extractor.extract_npc_claims(agnis, "I have the sword.", [agnis], created_at=1.0)

        assert result == []

    def test_llm_exception_returns_empty_not_raised(self, extractor: ClaimExtractor, mock_llm_client: MagicMock, agnis: Character):
        mock_llm_client.request_call_with_overridden_params.side_effect = Exception("connection error")

        result = extractor.extract_npc_claims(agnis, "I have the sword.", [agnis], created_at=1.0)

        assert result == []


class TestQuestionGuard:
    """A question asserts nothing - the model must not be told (or allowed)
    to infer a claim's truth value merely from being asked about it. Since
    this is a prompt-level instruction, not something enforceable by Python
    parsing, the test is that the guard text is actually present in both
    rendered prompts (player-side and NPC-side)."""

    def test_question_guard_present_in_player_side_prompt(self):
        prompt = ClaimExtractor._ClaimExtractor__build_system_prompt({Actor.AGNIS, Actor.PLAYER})
        assert "must not produce a claim" in prompt
        assert "question" in prompt.lower()

    def test_question_guard_present_in_npc_side_prompt(self):
        prompt = ClaimExtractor._ClaimExtractor__build_npc_system_prompt(Actor.AGNIS, {Actor.AGNIS, Actor.PLAYER})
        assert "must not produce a claim" in prompt
        assert "question" in prompt.lower()


class TestCommitmentFieldGuard:
    """"value" and "commitment_status" are easy for a weaker model to
    conflate (eg putting "VIOLATED" in "value", which then fails
    PREDICATE_VALUE_DOMAIN validation and silently drops the whole claim,
    since "value" for COMMITMENT is only ever "WAIT"/"DELIVER_ITEM"). Same
    as the question guard, this is a prompt-level fix - the test just proves
    the disambiguating instruction and example are actually present in both
    rendered prompts."""

    def test_guard_present_in_player_side_prompt(self):
        prompt = ClaimExtractor._ClaimExtractor__build_system_prompt({Actor.AGNIS, Actor.PLAYER})
        assert "SEPARATE fields" in prompt
        assert '"value": "VIOLATED"' in prompt  # the "not this" example
        assert '"commitment_status": "VIOLATED"' in prompt  # the "this" example

    def test_guard_present_in_npc_side_prompt(self):
        prompt = ClaimExtractor._ClaimExtractor__build_npc_system_prompt(Actor.AGNIS, {Actor.AGNIS, Actor.PLAYER})
        assert "SEPARATE fields" in prompt
        assert '"value": "VIOLATED"' in prompt
        assert '"commitment_status": "VIOLATED"' in prompt
