"""
Serialization for BeliefStateDAG, persisted alongside Summaries' files.

In Mantella, text summaries live in:
    data/conversations/{world_id}/{npc_name}_{ref_id}/{npc_name}_summary_{n}.txt
This module produces/reads a JSON file in the same folder, e.g.:
    data/conversations/{world_id}/{npc_name}_{ref_id}/{npc_name}_beliefstate.json
so the DAG follows the same lifecycle as the native summaries (per-world,
per-NPC).
"""

from __future__ import annotations

import json
import os
from typing import Any

from .dag import BeliefStateDAG, Statement, ensure_id_counter_past
from .entities import (
    CommitmentClause,
    CommitmentStatus,
    TrustLevel,
    PREDICATE_ENTITY_DOMAIN,
    Predicate,
    SourceType,
    Status,
    StatementType,
)


_VALUE_ENUM_PER_PREDICATE: dict[Predicate, type | None] = {
    Predicate.POSSESSION: None,  # bool, no Enum
    Predicate.BETRAYAL: None,
    Predicate.INTENTION_TO_FOLLOW: None,
    Predicate.TRUST: TrustLevel,
    Predicate.COMMITMENT: CommitmentClause,
}


def _value_to_json(predicate: Predicate, value: Any) -> Any:
    enum_cls = _VALUE_ENUM_PER_PREDICATE[predicate]
    return value.value if enum_cls is not None else value


def _value_from_json(predicate: Predicate, value_json: Any) -> Any:
    enum_cls = _VALUE_ENUM_PER_PREDICATE[predicate]
    return enum_cls(value_json) if enum_cls is not None else value_json


def statement_to_dict(stmt: Statement) -> dict:
    return {
        "id": stmt.id,
        "type": stmt.type.value,
        "predicate": stmt.predicate.value,
        "subject": stmt.subject.value,
        "object": stmt.object.value,
        "value": _value_to_json(stmt.predicate, stmt.value),
        "status": stmt.status.value,
        "commitment_status": stmt.commitment_status.value if stmt.commitment_status else None,
        "source_type": stmt.source_type.value,
        "created_at": stmt.created_at,
    }


def statement_from_dict(d: dict) -> Statement:
    predicate = Predicate(d["predicate"])
    subject_type, object_type = PREDICATE_ENTITY_DOMAIN[predicate]
    return Statement(
        id=d["id"],
        type=StatementType(d["type"]),
        predicate=predicate,
        subject=subject_type(d["subject"]),
        object=object_type(d["object"]),
        value=_value_from_json(predicate, d["value"]),
        status=Status(d["status"]),
        commitment_status=CommitmentStatus(d["commitment_status"]) if d["commitment_status"] else None,
        source_type=SourceType(d["source_type"]),
        created_at=d["created_at"],
    )


def serialize(dag: BeliefStateDAG) -> dict:
    
    return {
        "npc": dag.npc,
        "log": [statement_to_dict(s) for s in dag.log()],
        "depends_on_edges": [list(pair) for pair in dag.depends_on_edges()],
    }


def save_to_file(dag: BeliefStateDAG, path: str) -> None:
    
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(serialize(dag), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)  # atomic on the same filesystem


def deserialize(data: dict) -> BeliefStateDAG:
    
    ensure_id_counter_past(d["id"] for d in data["log"])
    dag = BeliefStateDAG(npc=data["npc"])
    for d in data["log"]:
        stmt = statement_from_dict(d)
        dag.validate_entities(stmt)
        dag.register(stmt)
    for cascade_id, trigger_id in data["depends_on_edges"]:
        dag.register_depends_on_edge(cascade_id, trigger_id)
    return dag


def load_from_file(path: str) -> BeliefStateDAG:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return deserialize(data)
