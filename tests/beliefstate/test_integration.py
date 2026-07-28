"""
Integration test for Task 3: once extraction (writer) and prompt building
(reader) go through the same BeliefStateManager instance, an inserted claim
must be visible to get_prompt_text() immediately - purely in-memory, with no
disk read or write in between. The DAG cache in BeliefStateManager already
provides this (get_dag() returns the same live instance every time within its
lifetime); these tests exist to prove that stays true now that extraction is
wired in, not to change any behaviour.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.beliefstate import engine
from src.beliefstate.entities import Predicate
from src.beliefstate.extraction import ClaimExtractor
from src.beliefstate.manager import BeliefStateManager
from src.character_manager import Character
from src.config.config_loader import ConfigLoader
from src.games.equipment import Equipment
from src.llm.client_base import ClientBase


def make_npc(name: str, ref_id: str = "000F4B") -> Character:
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


class FakeGame:
    """Just enough of Gameable's interface for BeliefStateManager: a
    conversation_folder_path pointing at a pytest tmp_path, never the user's
    real save folder."""
    def __init__(self, conversation_folder_path: str) -> None:
        self._conversation_folder_path = conversation_folder_path

    @property
    def conversation_folder_path(self) -> str:
        return self._conversation_folder_path


@pytest.fixture
def belief_state_manager(tmp_path) -> BeliefStateManager:
    return BeliefStateManager(FakeGame(str(tmp_path)))


@pytest.fixture
def mock_llm_client() -> MagicMock:
    return MagicMock(spec=ClientBase)


@pytest.fixture
def claim_extractor(mock_llm_client: MagicMock, default_config: ConfigLoader) -> ClaimExtractor:
    return ClaimExtractor(mock_llm_client, default_config)


def test_extracted_claim_is_immediately_visible_in_prompt_text(
    belief_state_manager: BeliefStateManager,
    claim_extractor: ClaimExtractor,
    mock_llm_client: MagicMock,
):
    """Player makes a promise -> extracted as a COMMITMENT claim -> inserted
    -> the very next get_prompt_text() call for that NPC includes it, with
    zero disk reads/writes anywhere in the sequence."""
    agnis = make_npc("Agnis")
    world_id = "TestWorld"

    mock_llm_client.request_call_with_overridden_params.return_value = json.dumps([
        {"predicate": "COMMITMENT", "subject": "Agnis", "object": "Player", "value": "WAIT", "source": "player"}
    ])

    with patch("src.beliefstate.manager.load_from_file") as mock_load, \
         patch("src.beliefstate.manager.save_to_file") as mock_save:

        # Task 2 step 1: the DAG is loaded/created up front, before extraction
        dag = belief_state_manager.get_dag(agnis, world_id)
        assert dag.active() == []
        assert belief_state_manager.get_prompt_text([agnis], world_id) == ""

        claims = claim_extractor.extract_claims(
            "I promise I'll wait for you right here.", "", [agnis], created_at=5.0,
        )
        assert len(claims) == 1
        assert claims[0].predicate == Predicate.COMMITMENT

        for claim in claims:
            engine.insert(dag, claim)

        prompt_text = belief_state_manager.get_prompt_text([agnis], world_id)
        assert prompt_text == "Agnis promised Player to wait (pending)."

        mock_load.assert_not_called()
        mock_save.assert_not_called()


def test_get_dag_returns_the_same_live_instance(belief_state_manager: BeliefStateManager):
    """get_dag() must return the SAME BeliefStateDAG instance on repeated
    calls within the manager's lifetime - this identity is what lets a writer
    (extraction/insertion) and a reader (prompt building) see the same live
    state without any explicit hand-off or reload."""
    agnis = make_npc("Agnis")
    world_id = "TestWorld"

    first = belief_state_manager.get_dag(agnis, world_id)
    second = belief_state_manager.get_dag(agnis, world_id)

    assert first is second


def test_insert_via_one_reference_is_visible_via_another(belief_state_manager: BeliefStateManager):
    """Even without going through ClaimExtractor: anything inserted through
    one get_dag() reference must show up in get_prompt_text(), proving
    BeliefStateManager itself is the single shared source of truth mid-
    conversation (the file on disk is never consulted again once cached)."""
    from src.beliefstate.entities import Actor, Item, SourceType
    from src.beliefstate.dag import create_statement

    agnis = make_npc("Agnis")
    world_id = "TestWorld"

    writer_dag = belief_state_manager.get_dag(agnis, world_id)
    statement = create_statement(
        Predicate.POSSESSION, Actor.AGNIS, Item.ANCESTRAL_SWORD, True, SourceType.ENGINE, created_at=1.0,
    )
    engine.insert(writer_dag, statement)

    reader_dag = belief_state_manager.get_dag(agnis, world_id)
    assert reader_dag is writer_dag
    assert "Agnis owns Ancestral_Sword." in belief_state_manager.get_prompt_text([agnis], world_id)
