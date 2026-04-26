from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


NODE_TYPES: List[str] = ["herb", "ingredient", "protein", "pathway", "disease"]

RELATIONS: List[Tuple[str, str, str, bool]] = [
    ("HI", "herb", "ingredient", False),
    ("IP", "ingredient", "protein", True),
    ("PPi", "protein", "protein", True),
    ("PPath", "protein", "pathway", False),
    ("PD", "protein", "disease", True),
    ("PathD", "pathway", "disease", True),
]


@dataclass(frozen=True)
class RelationSpec:
    name: str
    head_type: str
    tail_type: str
    has_evidence: bool


REL_SPECS: List[RelationSpec] = [RelationSpec(*r) for r in RELATIONS]

REL_NAME_TO_ID: Dict[str, int] = {r.name: i for i, r in enumerate(REL_SPECS)}
