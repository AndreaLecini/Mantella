"""
Regression test for a cross-process id collision: dag.py's id counter used to
be a fresh itertools.count(1) every process launch, with no knowledge of ids
already handed out by a *previous* process and persisted to a beliefstate.json
file. The first statement created live after loading such a file could then
be assigned an id already active in the loaded DAG - silently overwriting
BeliefStateDAG._by_id[id] while a stale _active_by_key entry kept pointing at
it. Symptom: one proposition (whichever id got reused) vanishes from
active(), and another appears twice (once under its own key, once as a ghost
under the collided key).

ensure_id_counter_past() fixes this by advancing the id counter past every id
in a DAG's log the moment it's deserialized - the only place ids minted by
another process enter this one's memory.
"""

import pytest

from src.beliefstate import Actor, Item, Predicate, SourceType, TrustLevel, create_statement
from src.beliefstate.engine import insert
from src.beliefstate.serialization import deserialize, serialize
import src.beliefstate.dag as dag_module


@pytest.fixture(autouse=True)
def reset_id_counter():
    """The id counter is process-global state (src.beliefstate.dag._next_id_value)
    - reset it around each test so tests don't leak ids into one another,
    regardless of run order."""
    original = dag_module._next_id_value
    dag_module._next_id_value = 1
    yield
    dag_module._next_id_value = original


def test_deserializing_advances_the_counter_past_loaded_ids():
    dag = dag_module.BeliefStateDAG(npc="Agnis")
    insert(dag, create_statement(Predicate.POSSESSION, Actor.AGNIS, Item.ANCESTRAL_SWORD, True, SourceType.ENGINE, created_at=1.0))
    insert(dag, create_statement(Predicate.TRUST, Actor.AGNIS, Actor.PLAYER, TrustLevel.HIGH, SourceType.PLAYER_DIALOGUE, created_at=2.0))
    saved_ids = {s.id for s in dag.active()}
    assert saved_ids == {"S00001", "S00002"}

    data = serialize(dag)

    # Simulate a brand-new process: reset the counter as if this were a
    # fresh launch, unaware that ids up to S00002 already exist in `data`.
    dag_module._next_id_value = 1
    deserialize(data)

    fresh_statement = create_statement(Predicate.BETRAYAL, Actor.PLAYER, Actor.AGNIS, True, SourceType.ENGINE, created_at=3.0)
    assert fresh_statement.id not in saved_ids


def test_no_id_collision_end_to_end_after_reload():
    """The exact bug: after loading a DAG, a newly created statement's id
    used to collide with an already-loaded one, causing that old statement
    to silently vanish from active() and the new one to appear as a ghost
    duplicate under the collided key."""
    dag = dag_module.BeliefStateDAG(npc="Agnis")
    insert(dag, create_statement(Predicate.POSSESSION, Actor.AGNIS, Item.ANCESTRAL_SWORD, True, SourceType.ENGINE, created_at=1.0))
    insert(dag, create_statement(Predicate.TRUST, Actor.AGNIS, Actor.PLAYER, TrustLevel.HIGH, SourceType.PLAYER_DIALOGUE, created_at=2.0))
    data = serialize(dag)

    dag_module._next_id_value = 1  # simulate a fresh process
    reloaded = deserialize(data)

    betrayal = create_statement(Predicate.BETRAYAL, Actor.PLAYER, Actor.AGNIS, True, SourceType.ENGINE, created_at=3.0)
    insert(reloaded, betrayal)

    active_ids = [s.id for s in reloaded.active()]
    assert len(active_ids) == len(set(active_ids)), "no two active statements should ever share an id"

    predicates_present = {s.predicate for s in reloaded.active()}
    assert Predicate.POSSESSION in predicates_present, "the loaded POSSESSION claim must not have been clobbered"
    assert Predicate.BETRAYAL in predicates_present
    # R1 cascade: betrayal drops trust
    assert reloaded.active_for_key((dag_module.StatementType.RELATIONAL_STATE, Predicate.TRUST, Actor.AGNIS, Actor.PLAYER)).value == TrustLevel.LOW

    betrayal_lines = [
        s for s in reloaded.active()
        if s.predicate == Predicate.BETRAYAL and s.subject == Actor.PLAYER and s.object == Actor.AGNIS
    ]
    assert len(betrayal_lines) == 1, "betrayal must appear exactly once, not duplicated via a collided id"
