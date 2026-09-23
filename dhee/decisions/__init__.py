"""Typed decisions for Dhee: the closed choices a memory makes, made by a
decision model (TypeSafe's Jev) instead of parsed out of LLM prose.

* :mod:`dhee.decisions.jev` — the client, and the ``Decider`` protocol.
* :mod:`dhee.decisions.facts` — the fact gate on the write path: about the
  person?, which label?, restates a stored fact?
* :mod:`dhee.decisions.select` — the read gate: which candidate memories would
  change the next reply?
* :mod:`dhee.decisions.regate` — put a store that grew before the gate through
  it once: retire, relabel, merge. Nothing is deleted.

Everything here is off unless ``MemoryConfig.decisions.enabled`` is set, and
every failure falls back to the behaviour Dhee had without it.
"""

from dhee.decisions.facts import FactGate, GateReport, fact_gate_from_config
from dhee.decisions.jev import (
    Decider,
    JevDecider,
    decider_from_config,
    decisions_switched_off,
    noul,
    ranked,
)
from dhee.decisions.regate import RegateReport, regate_facts
from dhee.decisions.select import Selected, select_relevant

__all__ = [
    "Decider",
    "FactGate",
    "GateReport",
    "JevDecider",
    "RegateReport",
    "Selected",
    "decider_from_config",
    "decisions_switched_off",
    "fact_gate_from_config",
    "noul",
    "ranked",
    "regate_facts",
    "select_relevant",
]
