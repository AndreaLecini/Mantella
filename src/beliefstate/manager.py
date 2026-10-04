"""
BeliefStateManager, the connection point between the beliefstate module
(independent of Mantella) and Context/game_manager.

Same shape as src/remember/remembering.py:Remembering.get_prompt_text(), so it
plugs into Context.generate_system_message() with the same pattern already
used for conversation_summaries, no new concept for someone reading the
code, just a second text source for the prompt.

Persistence: same folder convention as Summaries
(src/remember/summaries.py:__get_latest_conversation_summary_file_path),
"{base_name} - {ref_id}" inside conversation_folder_path/{world_id}/ — one
belief state per NPC per world, next to the native summary files.
"""

import os

from src import utils
from src.character_manager import Character
from src.games.gameable import Gameable

from src.beliefstate import Actor, BeliefStateDAG, load_from_file, save_to_file
from src.beliefstate.prompt import generate_belief_state_text

logger = utils.get_logger()

_FILE_NAME = "beliefstate.json"


def actor_for_name(name: str) -> Actor | None:
    base_name = utils.remove_trailing_number(name)
    try:
        return Actor(base_name)
    except ValueError:
        return None


class BeliefStateManager:

    def __init__(self, game: Gameable) -> None:
        self.__game = game
        self.__cache: dict[tuple[str, str], BeliefStateDAG] = {}

    @utils.time_it
    def get_prompt_text(self, characters: list[Character], world_id: str) -> str:
        
        sections = []
        for character in characters:
            actor = actor_for_name(character.name)
            if actor is None:
                continue
            dag = self.__load(character, world_id)
            text = generate_belief_state_text(dag)
            if not text:
                continue
            if len(characters) > 1:
                sections.append(f"{character.name}: {text}")
            else:
                sections.append(text)
        if not sections:
            return ""
        return "\n\n".join(sections)

    def get_dag(self, character: Character, world_id: str) -> BeliefStateDAG:
        
        return self.__load(character, world_id)

    def save(self, character: Character, world_id: str) -> None:
        dag = self.__cache.get((world_id, character.name))
        if dag is not None:
            save_to_file(dag, self.__file_path(character, world_id))

    

    def __file_path(self, character: Character, world_id: str) -> str:
        base_name = utils.remove_trailing_number(character.name)
        folder_name = f"{base_name} - {character.ref_id}"
        folder_path = os.path.join(
            self.__game.conversation_folder_path, world_id, folder_name
        ).replace(os.sep, "/")
        os.makedirs(folder_path, exist_ok=True)
        return os.path.join(folder_path, _FILE_NAME).replace(os.sep, "/")

    def __load(self, character: Character, world_id: str) -> BeliefStateDAG:
        cache_key = (world_id, character.name)
        if cache_key in self.__cache:
            return self.__cache[cache_key]

        path = self.__file_path(character, world_id)
        if os.path.exists(path):
            try:
                dag = load_from_file(path)
            except (ValueError, KeyError) as e:
                
                logger.warning(f"Invalid belief state for {character.name}, restarting empty: {e}")
                dag = BeliefStateDAG(npc=character.name)
        else:
            dag = BeliefStateDAG(npc=character.name)

        self.__cache[cache_key] = dag
        return dag
