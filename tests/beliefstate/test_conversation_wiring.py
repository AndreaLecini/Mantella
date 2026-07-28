"""
Regression test: a claim extracted mid-conversation must not just land in the
DAG - it must also become visible in the *system message already sitting in
the message thread*, since that's what actually gets sent to the LLM. The
system message is normally built once per conversation and only rebuilt when
actors change (see Conversation.__update_conversation_type) or the
conversation reloads; __extract_and_apply_beliefs() has to explicitly refresh
it, or a belief captured this turn would only ever show up in a *future*
conversation, never the current one.
"""

import json
from unittest.mock import MagicMock

import pytest

from src.beliefstate import (
    Actor,
    CommitmentClause,
    CommitmentStatus,
    Item,
    Predicate,
    SourceType,
    TrustLevel,
    create_statement,
)
from src.beliefstate.dag import BeliefStateDAG
from src.beliefstate.engine import insert, insert_or_transition
from src.beliefstate.manager import BeliefStateManager
from src.beliefstate.extraction import ClaimExtractor
from src.beliefstate.verifier import ActionVerifier
from src.conversation.context import Context
from src.conversation.conversation import Conversation
from src.output_manager import ChatManager
from src.character_manager import Character
from src.characters_manager import Characters
from src.games.equipment import Equipment
from src.config.config_loader import ConfigLoader
from src.games.skyrim import Skyrim
from src.llm.llm_client import LLMClient
from src.llm.messages import AssistantMessage
from src.llm.sentence import Sentence
from src.llm.sentence_content import SentenceContent, SentenceTypeEnum
from src.remember.summaries import Summaries


def make_character(name: str, ref_id: str, is_player: bool) -> Character:
    return Character(
        base_id=ref_id,
        ref_id=ref_id,
        name=name,
        gender=0,
        race="[Race <NordRace (00013746)>]",
        is_player_character=is_player,
        bio=f"You are {name}." if not is_player else "",
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
        custom_character_values={"mantella_pc_description": "", "mantella_pc_voiceplayerinput": False} if is_player else None,
    )


@pytest.fixture
def belief_state_manager(skyrim: Skyrim) -> BeliefStateManager:
    return BeliefStateManager(skyrim)


@pytest.fixture
def mock_extraction_llm_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def claim_extractor(mock_extraction_llm_client: MagicMock, default_config: ConfigLoader) -> ClaimExtractor:
    return ClaimExtractor(mock_extraction_llm_client, default_config)


@pytest.fixture
def action_verifier(claim_extractor: ClaimExtractor) -> ActionVerifier:
    return ActionVerifier(claim_extractor)


@pytest.fixture
def wired_context(
    default_config: ConfigLoader,
    llm_client: LLMClient,
    default_rememberer: Summaries,
    english_language_info: dict,
    belief_state_manager: BeliefStateManager,
    claim_extractor: ClaimExtractor,
    action_verifier: ActionVerifier,
) -> Context:
    return Context("SmokeWorld", default_config, llm_client, default_rememberer, english_language_info, belief_state_manager, claim_extractor, action_verifier)


@pytest.fixture
def wired_conversation(wired_context: Context, llm_client: LLMClient, default_rememberer: Summaries, skyrim: Skyrim) -> Conversation:
    mock_chat_manager = MagicMock(spec=ChatManager)
    mock_chat_manager.discarded_character_name = None
    mock_chat_manager.listen_requested = False
    conversation = Conversation(wired_context, mock_chat_manager, default_rememberer, llm_client, None, False, False, skyrim)
    # avoid spinning up a real generation thread / real streaming LLM call
    conversation._Conversation__start_generating_npc_sentences = MagicMock()
    return conversation


def test_belief_extracted_mid_conversation_is_reflected_in_system_message(
    wired_conversation: Conversation,
    mock_extraction_llm_client: MagicMock,
):
    player = make_character("Dragonborn", "000014", is_player=True)
    agnis = make_character("Agnis", "000F4B", is_player=False)

    wired_conversation.add_or_update_character([player, agnis])
    # mirrors GameStateManager.__update_context(): actor changes only take
    # effect (ie the system message gets built) once update_context() runs
    wired_conversation.update_context("Skyrim", 12, None, None, None, None, None, 1.0)

    # sanity check: the belief isn't in the initial prompt yet
    initial_system_text = wired_conversation._Conversation__messages[0].text
    assert "Ancestral_Sword" not in initial_system_text

    mock_extraction_llm_client.request_call_with_overridden_params.return_value = json.dumps([
        {"predicate": "POSSESSION", "subject": "Agnis", "object": "Ancestral_Sword", "value": False, "source": "player"}
    ])

    wired_conversation.process_player_input("You don't have the sword anymore.")

    updated_system_text = wired_conversation._Conversation__messages[0].text
    assert "Agnis no longer owns Ancestral_Sword." in updated_system_text


def test_no_claims_extracted_leaves_system_message_untouched(
    wired_conversation: Conversation,
    mock_extraction_llm_client: MagicMock,
):
    player = make_character("Dragonborn", "000014", is_player=True)
    agnis = make_character("Agnis", "000F4B", is_player=False)
    wired_conversation.add_or_update_character([player, agnis])
    # mirrors GameStateManager.__update_context(): actor changes only take
    # effect (ie the system message gets built) once update_context() runs
    wired_conversation.update_context("Skyrim", 12, None, None, None, None, None, 1.0)

    initial_system_text = wired_conversation._Conversation__messages[0].text

    mock_extraction_llm_client.request_call_with_overridden_params.return_value = "[]"
    wired_conversation.process_player_input("Nice weather today.")

    assert wired_conversation._Conversation__messages[0].text == initial_system_text


class TestInsertOrTransitionCommitmentBugfix:
    """engine.insert()'s idempotency check only compares .value (the clause),
    never .commitment_status - so a claim re-asserting the same clause with a
    new status used to be silently dropped as a duplicate, and
    engine.transition_commitment() (built for exactly this) was never called
    from anywhere. engine.insert_or_transition() is the fix: it detects a
    same-clause / different-status claim and routes it to
    transition_commitment() instead of insert(). Shared by both
    Conversation.__extract_and_apply_beliefs (player-side) and
    ActionVerifier.verify (NPC-side, see test_verifier.py)."""

    def test_status_change_transitions_the_existing_node_in_place(self):
        dag = BeliefStateDAG(npc="Agnis")
        commitment = create_statement(
            Predicate.COMMITMENT, Actor.AGNIS, Actor.PLAYER, CommitmentClause.WAIT,
            SourceType.PLAYER_DIALOGUE, created_at=1.0,
        )
        insert(dag, commitment)

        fulfillment_claim = create_statement(
            Predicate.COMMITMENT, Actor.AGNIS, Actor.PLAYER, CommitmentClause.WAIT,
            SourceType.PLAYER_DIALOGUE, created_at=2.0, commitment_status=CommitmentStatus.FULFILLED,
        )
        insert_or_transition(dag, fulfillment_claim)

        active = dag.active_for_key(commitment.key)
        assert active.commitment_status == CommitmentStatus.FULFILLED
        assert active.id == commitment.id  # same node, mutated in place - not a new one

    def test_violated_status_fires_r3_trust_cascade(self):
        dag = BeliefStateDAG(npc="Agnis")
        commitment = create_statement(
            Predicate.COMMITMENT, Actor.AGNIS, Actor.PLAYER, CommitmentClause.WAIT,
            SourceType.PLAYER_DIALOGUE, created_at=1.0,
        )
        insert(dag, commitment)

        violation_claim = create_statement(
            Predicate.COMMITMENT, Actor.AGNIS, Actor.PLAYER, CommitmentClause.WAIT,
            SourceType.ENGINE, created_at=2.0, commitment_status=CommitmentStatus.VIOLATED,
        )
        insert_or_transition(dag, violation_claim)

        assert dag.active_for_key(commitment.key).commitment_status == CommitmentStatus.VIOLATED
        # R3: a violated commitment invalidates trust (swapped subject/object)
        trust_statements = [s for s in dag.active() if s.predicate == Predicate.TRUST]
        assert len(trust_statements) == 1
        assert trust_statements[0].subject == Actor.PLAYER
        assert trust_statements[0].object == Actor.AGNIS
        assert trust_statements[0].value == TrustLevel.LOW

    def test_brand_new_commitment_still_goes_through_insert(self):
        dag = BeliefStateDAG(npc="Agnis")
        commitment = create_statement(
            Predicate.COMMITMENT, Actor.AGNIS, Actor.PLAYER, CommitmentClause.WAIT,
            SourceType.PLAYER_DIALOGUE, created_at=1.0,
        )

        insert_or_transition(dag, commitment)

        active = dag.active_for_key(commitment.key)
        # __insert_or_transition inserts a dataclasses.replace() copy (not
        # the same object, see __insert_or_transition's own docstring on
        # why), so compare by id/value rather than identity
        assert active.id == commitment.id
        assert active.commitment_status == CommitmentStatus.PENDING

    def test_different_clause_goes_through_normal_conflict_resolution(self):
        dag = BeliefStateDAG(npc="Agnis")
        wait_commitment = create_statement(
            Predicate.COMMITMENT, Actor.AGNIS, Actor.PLAYER, CommitmentClause.WAIT,
            SourceType.PLAYER_DIALOGUE, created_at=1.0,
        )
        insert(dag, wait_commitment)

        deliver_claim = create_statement(
            Predicate.COMMITMENT, Actor.AGNIS, Actor.PLAYER, CommitmentClause.DELIVER_ITEM,
            SourceType.PLAYER_DIALOGUE, created_at=2.0,
        )
        insert_or_transition(dag, deliver_claim)

        active = dag.active_for_key(wait_commitment.key)
        assert active.value == CommitmentClause.DELIVER_ITEM  # newer claim won via normal resolve()
        assert active.id == deliver_claim.id  # a genuinely new node, not a mutated old one

    def test_redundant_same_status_claim_is_a_no_op(self):
        dag = BeliefStateDAG(npc="Agnis")
        commitment = create_statement(
            Predicate.COMMITMENT, Actor.AGNIS, Actor.PLAYER, CommitmentClause.WAIT,
            SourceType.PLAYER_DIALOGUE, created_at=1.0,
        )
        insert(dag, commitment)

        redundant_claim = create_statement(
            Predicate.COMMITMENT, Actor.AGNIS, Actor.PLAYER, CommitmentClause.WAIT,
            SourceType.PLAYER_DIALOGUE, created_at=2.0,
        )
        insert_or_transition(dag, redundant_claim)

        active = dag.active_for_key(commitment.key)
        assert active.id == commitment.id
        assert active.commitment_status == CommitmentStatus.PENDING


def test_extracted_fulfillment_updates_prompt_text_end_to_end(
    wired_conversation: Conversation,
    mock_extraction_llm_client: MagicMock,
):
    """The user's actual reported scenario, reproduced end to end: a
    commitment already exists (eg pre-seeded before the session), the player
    says something that fulfills it, and the very next prompt reflects the
    updated status - not the commitment silently staying PENDING forever."""
    player = make_character("Dragonborn", "000014", is_player=True)
    agnis = make_character("Agnis", "000F4B", is_player=False)
    wired_conversation.add_or_update_character([player, agnis])
    wired_conversation.update_context("Skyrim", 12, None, None, None, None, None, 1.0)

    belief_state_manager = wired_conversation.context.belief_state_manager
    dag = belief_state_manager.get_dag(agnis, wired_conversation.context.world_id)
    insert(dag, create_statement(
        Predicate.COMMITMENT, Actor.AGNIS, Actor.PLAYER, CommitmentClause.WAIT,
        SourceType.PLAYER_DIALOGUE, created_at=1.0,
    ))
    assert "(pending)" in belief_state_manager.get_prompt_text([agnis], wired_conversation.context.world_id)

    mock_extraction_llm_client.request_call_with_overridden_params.return_value = json.dumps([
        {
            "predicate": "COMMITMENT", "subject": "Agnis", "object": "Player", "value": "WAIT",
            "source": "player", "commitment_status": "FULFILLED",
        }
    ])
    wired_conversation.process_player_input("Thank you for waiting for me, as you promised.")

    updated_system_text = wired_conversation._Conversation__messages[0].text
    assert "Agnis promised Player to wait (fulfilled)." in updated_system_text
    assert "(pending)" not in updated_system_text


def test_action_verifier_rejects_contradicting_npc_response_end_to_end(
    wired_conversation: Conversation,
    mock_extraction_llm_client: MagicMock,
):
    """The Action Verifier's actual job, reproduced through the real
    Conversation wiring (not just ActionVerifier in isolation): the NPC's
    own generated line contradicts an established belief -> the
    contradiction is detected and rejected, not inserted, and the belief
    stays exactly as it was."""
    player = make_character("Dragonborn", "000014", is_player=True)
    agnis = make_character("Agnis", "000F4B", is_player=False)
    wired_conversation.add_or_update_character([player, agnis])
    wired_conversation.update_context("Skyrim", 12, None, None, None, None, None, 1.0)

    belief_state_manager = wired_conversation.context.belief_state_manager
    dag = belief_state_manager.get_dag(agnis, wired_conversation.context.world_id)
    insert(dag, create_statement(
        Predicate.POSSESSION, Actor.AGNIS, Item.ANCESTRAL_SWORD, True,
        SourceType.ENGINE, created_at=1.0,
    ))

    # Simulate Agnis having just spoken a contradicting line - appends a
    # completed AssistantMessage directly, the same shape
    # retrieve_sentence_from_queue() would leave behind after streaming,
    # without needing to drive the real streaming pipeline.
    assistant_message = AssistantMessage(wired_conversation.context.config)
    sentence = Sentence(
        SentenceContent(agnis, "I do not have the ancestral sword.", SentenceTypeEnum.SPEECH, False),
        voice_file="", voice_line_duration=0.0,
    )
    assistant_message.add_sentence(sentence)
    wired_conversation._Conversation__messages.add_message(assistant_message)

    # process_player_input() calls __verify_last_npc_response() (verifying
    # Agnis's prior line) before __extract_and_apply_beliefs() (extracting
    # the player's own new line) - two separate extraction calls, in order.
    mock_extraction_llm_client.request_call_with_overridden_params.side_effect = [
        json.dumps([{"predicate": "POSSESSION", "subject": "Agnis", "object": "Ancestral_Sword", "value": False}]),
        "[]",
    ]
    wired_conversation.process_player_input("Okay, never mind.")

    final_text = belief_state_manager.get_prompt_text([agnis], wired_conversation.context.world_id)
    assert "Agnis owns Ancestral_Sword." in final_text
    assert "no longer owns" not in final_text
