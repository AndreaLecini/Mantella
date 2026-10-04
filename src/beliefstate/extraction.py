"""
ClaimExtractor, turns free text (what the player said, what happened in-game)
into candidate belief-state Statements.

Mirrors src/llm/function_client.py's shape: a separate, structured-output LLM
call distinct from the main conversation-generation call. Where FunctionClient
constrains the LLM to a tool-calling schema, this constrains it to the closed
predicate/entity/value vocabulary in entities.py, via a strict JSON schema
described in the prompt plus strict parsing on the way back, never letting
the LLM invent a predicate or entity name.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src import utils
from src.character_manager import Character
from src.config.config_loader import ConfigLoader
from src.llm.client_base import ClientBase
from src.llm.message_thread import message_thread
from src.llm.messages import UserMessage

from src.beliefstate.dag import Statement, create_statement
from src.beliefstate.entities import (
    PREDICATE_ENTITY_DOMAIN,
    PREDICATE_VALUE_DOMAIN,
    Actor,
    CommitmentStatus,
    Item,
    Predicate,
    SourceType,
)
from src.beliefstate.manager import actor_for_name

logger = utils.get_logger()

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_QUESTION_GUARD = (
    "Only extract a claim from a sentence that clearly ASSERTS something as "
    "fact. A question (eg \"Do you have the sword?\", \"Will you wait for "
    "me?\") asserts nothing by itself and must not produce a claim, even if "
    "you can guess a likely answer from context - extract only what is "
    "explicitly stated, never what is merely asked."
)

_DIRECTION_GUARD = (
    "For any predicate where both \"subject\" and \"object\" are Actors, "
    "assign them purely from grammar: the grammatical agent (who performs "
    "the action) is always \"subject\", the grammatical patient/recipient is "
    "always \"object\". Never fall back on genre expectations about who "
    "usually plays which role in a Skyrim quest - eg assuming NPCs are the "
    "ones who promise or offer to follow, or that the player is the one "
    "betrayed. A given sentence can go either way; only its own grammar "
    "decides subject vs object."
)

_BETRAYAL_DIRECTION_GUARD = (
    "For \"BETRAYAL\": \"subject\" is whoever DID the betraying, \"object\" is "
    "whoever WAS betrayed - never the reverse, and never guess the direction "
    "from tone or who seems sympathetic. Determine it purely from grammar: "
    "the grammatical agent (the one performing the action of betraying) is "
    "\"subject\"; the grammatical patient (the one betrayal was done to) is "
    "\"object\". Example: \"I have betrayed you\" said by the player about "
    "an NPC means the PLAYER did the betraying - "
    "{\"predicate\": \"BETRAYAL\", \"subject\": \"Player\", \"object\": \"<the NPC>\", \"value\": true}, "
    "NOT the NPC as subject. \"You betrayed me\" is the opposite: the person "
    "\"you\" refers to is \"subject\", \"Player\" is \"object\"."
)

_COMMITMENT_FIELD_GUARD = (
    "\"COMMITMENT\" has two SEPARATE fields - do not confuse them:\n"
    "- \"value\" is WHAT was promised: always exactly \"WAIT\" or "
    "\"DELIVER_ITEM\", never a status word.\n"
    "- \"commitment_status\" is whether that promise was kept: \"PENDING\" "
    "(default), \"FULFILLED\", or \"VIOLATED\".\n"
    "\"PENDING\"/\"FULFILLED\"/\"VIOLATED\" must NEVER appear in \"value\" - "
    "they only ever belong in \"commitment_status\". Example, a broken "
    "promise to wait: "
    "{\"predicate\": \"COMMITMENT\", \"value\": \"WAIT\", \"commitment_status\": \"VIOLATED\", ...} "
    "- NOT {\"value\": \"VIOLATED\", ...}."
)

_SOURCE_TAG_TO_SOURCE_TYPE: dict[str, SourceType] = {
    "player": SourceType.PLAYER_DIALOGUE,
    "player_dialogue": SourceType.PLAYER_DIALOGUE,
    "game_event": SourceType.ENGINE,
    "game_events": SourceType.ENGINE,
    "event": SourceType.ENGINE,
}


class ClaimExtractor:
    """LLM-backed extraction of closed-vocabulary claims from a single
    conversation turn's player text and in-game events."""


    _MAX_TOKENS_OVERRIDE = 1500

    @utils.time_it
    def __init__(self, client: ClientBase, config: ConfigLoader) -> None:
        self.__client = client
        self.__config = config

    @utils.time_it
    def extract_claims(
        self,
        player_text: str,
        game_events_text: str,
        involved_characters: list[Character],
        created_at: float,
    ) -> list[Statement]:
        """Extracts candidate Statements from this turn's player text and
        in-game events. Does not insert them into any DAG — see module
        docstring.

        Args:
            player_text: what the player said this turn (may be empty).
            game_events_text: the in-game events text for this turn (may be
                empty), e.g. UserMessage.get_ingame_events_text().
            involved_characters: the non-player NPCs present in the
                conversation. Restricts which Actor names the LLM is allowed
                to use, both to keep the prompt small and to stop it
                inventing claims about characters who aren't present.
            created_at: the in-game timestamp to stamp on any resulting
                Statement (see Context.game_days — never wall-clock time).

        Returns:
            list[Statement]: candidate statements, already validated against
            the closed vocabulary. May be empty.
        """
        involved_actors = {
            actor for c in involved_characters if (actor := actor_for_name(c.name)) is not None
        }
        allowed_actors = involved_actors | {Actor.PLAYER}

        if not involved_actors:
            return []
        if not player_text.strip() and not game_events_text.strip():
            return []

        try:
            raw_response = self.__request(player_text, game_events_text, allowed_actors)
        except Exception as e:
            logger.error(f"Claim extraction LLM error: {e}. Skipping extraction for this turn.")
            return []

        if not raw_response or not raw_response.strip():
            logger.debug("Claim extraction: empty LLM response, no claims extracted.")
            return []

        claim_dicts = self.__parse_response(raw_response)
        if claim_dicts is None:
            return []

        statements: list[Statement] = []
        for claim_dict in claim_dicts:
            statement = self.__build_statement(claim_dict, allowed_actors, created_at)
            if statement is not None:
                statements.append(statement)
        return statements

    @utils.time_it
    def extract_npc_claims(
        self,
        speaker: Character,
        npc_text: str,
        involved_characters: list[Character],
        created_at: float,
    ) -> list[Statement]:
        
        speaker_actor = actor_for_name(speaker.name)
        if speaker_actor is None:
            return []

        involved_actors = {
            actor for c in involved_characters if (actor := actor_for_name(c.name)) is not None
        }
        allowed_actors = involved_actors | {Actor.PLAYER, speaker_actor}

        if not npc_text.strip():
            return []

        try:
            raw_response = self.__request_npc(speaker_actor, npc_text, allowed_actors)
        except Exception as e:
            logger.error(f"Claim extraction (NPC response) LLM error: {e}. Skipping verification for this turn.")
            return []

        if not raw_response or not raw_response.strip():
            logger.debug("Claim extraction (NPC response): empty LLM response, no claims extracted.")
            return []

        claim_dicts = self.__parse_response(raw_response)
        if claim_dicts is None:
            return []

        statements: list[Statement] = []
        for claim_dict in claim_dicts:
            statement = self.__build_statement(claim_dict, allowed_actors, created_at, fixed_source_type=SourceType.LLM_GENERATED)
            if statement is not None:
                statements.append(statement)
        return statements

    

    def __request(self, player_text: str, game_events_text: str, allowed_actors: set[Actor]) -> str | None:
        system_prompt = self.__build_system_prompt(allowed_actors)
        user_text = self.__build_user_text(player_text, game_events_text)

        thread = message_thread(self.__config, system_prompt)
        thread.add_message(UserMessage(self.__config, user_text))

        return self.__client.request_call_with_overridden_params(thread, {"max_tokens": self._MAX_TOKENS_OVERRIDE})

    def __request_npc(self, speaker: Actor, npc_text: str, allowed_actors: set[Actor]) -> str | None:
        system_prompt = self.__build_npc_system_prompt(speaker, allowed_actors)

        thread = message_thread(self.__config, system_prompt)
        thread.add_message(UserMessage(self.__config, npc_text.strip()))

        return self.__client.request_call_with_overridden_params(thread, {"max_tokens": self._MAX_TOKENS_OVERRIDE})

    @staticmethod
    def __build_user_text(player_text: str, game_events_text: str) -> str:
        sections = []
        if player_text.strip():
            sections.append(f"Player said:\n{player_text.strip()}")
        if game_events_text.strip():
            sections.append(f"Game events:\n{game_events_text.strip()}")
        return "\n\n".join(sections)

    @staticmethod
    def __build_system_prompt(allowed_actors: set[Actor]) -> str:
        predicate_lines = "\n".join(
            ClaimExtractor.__describe_predicate(p) for p in Predicate
        )
        actor_names = ", ".join(f'"{a.value}"' for a in sorted(allowed_actors, key=lambda a: a.value))
        item_names = ", ".join(f'"{i.value}"' for i in Item)
        npc_names = ", ".join(f'"{a.value}"' for a in sorted(allowed_actors - {Actor.PLAYER}, key=lambda a: a.value))

        return f"""You are extracting factual claims from Skyrim dialogue and game events into a strict, closed vocabulary. You have no creative freedom: only emit a claim if it is one of the predicates below, using only the listed actor/item names, and only if the text clearly states or strongly implies it. If nothing in the text maps to this vocabulary, that is the expected outcome, not a failure.

Predicates (subject domain / object domain / value domain):
{predicate_lines}

Allowed actor names for this turn: {actor_names}
Allowed item names: {item_names}
Copy every predicate, actor, item, and value name exactly as written above, including case - never translate, paraphrase, or re-case them.

The "Player said" text, if present, is spoken BY "Player" TO {npc_names}. Resolve pronouns from the player's point of view: "you"/"your" refers to {npc_names}, "I"/"me"/"my" refers to "Player".

{_QUESTION_GUARD}

{_DIRECTION_GUARD}

{_BETRAYAL_DIRECTION_GUARD}

Output ONLY a JSON array, no prose, no markdown code fences. Each element must have this shape:
{{"predicate": "<one of the predicate names above>", "subject": "<actor name>", "object": "<actor or item name, matching the predicate's object domain>", "value": <matching the predicate's value domain>, "source": "player" or "game_event"}}

{_COMMITMENT_FIELD_GUARD}

Tag "source" as "player" for claims coming from what the player said, and "game_event" for claims coming from the game event lines. Do not guess a source for a claim that doesn't clearly come from one of these two inputs.

If nothing applies, output exactly: []"""

    @staticmethod
    def __build_npc_system_prompt(speaker: Actor, allowed_actors: set[Actor]) -> str:
        predicate_lines = "\n".join(
            ClaimExtractor.__describe_predicate(p) for p in Predicate
        )
        actor_names = ", ".join(f'"{a.value}"' for a in sorted(allowed_actors, key=lambda a: a.value))
        item_names = ", ".join(f'"{i.value}"' for i in Item)

        return f"""You are extracting factual claims from something {speaker.value} (a Skyrim NPC) just said out loud, into a strict, closed vocabulary. You have no creative freedom: only emit a claim if it is one of the predicates below, using only the listed actor/item names, and only if the text clearly states or strongly implies it. If nothing in the text maps to this vocabulary, that is the expected outcome, not a failure.

Predicates (subject domain / object domain / value domain):
{predicate_lines}

Allowed actor names for this turn: {actor_names}
Allowed item names: {item_names}
Copy every predicate, actor, item, and value name exactly as written above, including case - never translate, paraphrase, or re-case them.

The text below is spoken BY "{speaker.value}". Resolve pronouns from {speaker.value}'s point of view: "I"/"me"/"my" refers to "{speaker.value}", "you"/"your" refers to "Player".

{_QUESTION_GUARD}

{_DIRECTION_GUARD}

{_BETRAYAL_DIRECTION_GUARD}

Output ONLY a JSON array, no prose, no markdown code fences. Each element must have this shape:
{{"predicate": "<one of the predicate names above>", "subject": "<actor name>", "object": "<actor or item name, matching the predicate's object domain>", "value": <matching the predicate's value domain>}}

{_COMMITMENT_FIELD_GUARD}

If nothing applies, output exactly: []"""

    @staticmethod
    def __describe_predicate(predicate: Predicate) -> str:
        subject_type, object_type = PREDICATE_ENTITY_DOMAIN[predicate]
        value_domain = PREDICATE_VALUE_DOMAIN[predicate]
        if value_domain is bool:
            value_desc = "true or false"
        else:
            value_desc = " | ".join(f'"{member.value}"' for member in value_domain)
        return (
            f'- "{predicate.value}": subject is an {subject_type.__name__} name, '
            f'object is an {object_type.__name__} name, value is {value_desc}.'
        )

    

    @staticmethod
    def __parse_response(raw_response: str) -> list[dict] | None:
        cleaned = _CODE_FENCE_RE.sub("", raw_response.strip()).strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.debug(f"Claim extraction: malformed JSON response, dropping all claims: {e}")
            return None

        if not isinstance(parsed, list):
            logger.debug("Claim extraction: response was not a JSON array, dropping all claims.")
            return None

        return [item for item in parsed if isinstance(item, dict)]

    def __build_statement(
        self,
        claim: dict[str, Any],
        allowed_actors: set[Actor],
        created_at: float,
        fixed_source_type: SourceType | None = None,
    ) -> Statement | None:
        
        predicate_raw = claim.get("predicate")
        try:
            predicate = Predicate(predicate_raw)
        except ValueError:
            logger.debug(f"Claim extraction: unknown predicate {predicate_raw!r}, dropping claim.")
            return None

        subject_type, object_type = PREDICATE_ENTITY_DOMAIN[predicate]

        subject = self.__parse_entity(claim.get("subject"), subject_type)
        if subject is None:
            logger.debug(f"Claim extraction: invalid subject {claim.get('subject')!r} for {predicate.value}, dropping claim.")
            return None
        if isinstance(subject, Actor) and subject not in allowed_actors:
            logger.debug(f"Claim extraction: subject {subject.value!r} not in allowed actors for this turn, dropping claim.")
            return None

        object_ = self.__parse_entity(claim.get("object"), object_type)
        if object_ is None:
            logger.debug(f"Claim extraction: invalid object {claim.get('object')!r} for {predicate.value}, dropping claim.")
            return None
        if isinstance(object_, Actor) and object_ not in allowed_actors:
            logger.debug(f"Claim extraction: object {object_.value!r} not in allowed actors for this turn, dropping claim.")
            return None

        
        value = self.__parse_value(predicate, claim.get("value"))
        if value is None:
            logger.debug(f"Claim extraction: invalid value {claim.get('value')!r} for {predicate.value}, dropping claim.")
            return None

        if fixed_source_type is not None:
            source_type = fixed_source_type
        else:
            source_type = self.__parse_source_type(claim.get("source"))
            if source_type is None:
                logger.debug(f"Claim extraction: missing/unknown source tag {claim.get('source')!r}, dropping claim.")
                return None

        commitment_status = None
        if predicate == Predicate.COMMITMENT:
            commitment_status = self.__parse_commitment_status(claim.get("commitment_status"))
            if commitment_status is None:
                logger.debug(f"Claim extraction: invalid commitment_status {claim.get('commitment_status')!r}, dropping claim.")
                return None

        try:
            return create_statement(
                predicate=predicate,
                subject=subject,
                object=object_,
                value=value,
                source_type=source_type,
                created_at=created_at,
                commitment_status=commitment_status,
            )
        except ValueError as e:
            logger.debug(f"Claim extraction: {predicate.value} claim failed validation, dropping: {e}")
            return None

    @staticmethod
    def __parse_entity(raw: Any, entity_type: type) -> object | None:
        if not isinstance(raw, str):
            return None
        try:
            return entity_type(raw)
        except ValueError:
            return None

    @staticmethod
    def __parse_value(predicate: Predicate, raw: Any) -> object | None:
        value_domain = PREDICATE_VALUE_DOMAIN[predicate]
        if value_domain is bool:
            return raw if isinstance(raw, bool) else None
        if not isinstance(raw, str):
            return None
        try:
            return value_domain(raw)
        except ValueError:
            return None

    @staticmethod
    def __parse_source_type(raw: Any) -> SourceType | None:
        if not isinstance(raw, str):
            return None
        return _SOURCE_TAG_TO_SOURCE_TYPE.get(raw.lower())

    @staticmethod
    def __parse_commitment_status(raw: Any) -> CommitmentStatus | None:
        if raw is None:
            return CommitmentStatus.PENDING
        if not isinstance(raw, str):
            return None
        try:
            return CommitmentStatus(raw)
        except ValueError:
            return None
