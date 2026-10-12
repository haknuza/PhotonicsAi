"""Shared state of the graph: the dialogue, and nothing else.

Every turn carries the name of the role that produced it and the text that role
said. Extra, machine-written facts (the orchestrator's parsed decision, the
RMSE that a script really printed) ride along in optional keys of the turn, so
the whole run can be reconstructed from ``messages`` alone.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Mapping, Optional, TypedDict

SPECIALISTS = ("scientist", "general", "executor")
NEXT_ROLES = SPECIALISTS + ("human", "finish")


class Turn(TypedDict, total=False):
    role: str                      # user | orchestrator | scientist | general | executor | human
    text: str                      # what that role said
    decision: dict[str, Any]       # orchestrator turns only: parsed orchestrator decision
    runs: list[dict[str, Any]]     # executor turns only: one record per script that was run
    test_rmse: Optional[float]     # executor turns only: set by code from real stdout, never by the model
    train_rmse: Optional[float]
    answers: str                   # kind of human question answered: ambiguity | hard


class DialogueState(TypedDict):
    # operator.add: a node returns {"messages": [turn]} and the turn is appended.
    messages: Annotated[list[Turn], operator.add]


# ----------------------------------------------------------------------------
# Helpers that read the dialogue
# ----------------------------------------------------------------------------
MAX_TURN_CHARS = 6_000


def _clip(text: str, limit: int = MAX_TURN_CHARS) -> str:
    if len(text) <= limit:
        return text
    head, tail = limit // 3, limit - limit // 3
    return f"{text[:head]}\n...[{len(text) - limit} characters omitted]...\n{text[-tail:]}"


def render_dialogue(messages: list[Turn], limit: int = MAX_TURN_CHARS,
                    limits: Optional[Mapping[str, int]] = None) -> str:
    """Render the dialogue with an optional per-role character limit."""
    role_limits = limits or {}
    return "\n\n".join(
        f"[turn {i}] {m['role'].upper()}:\n{_clip(m['text'], role_limits.get(m['role'], limit))}"
        for i, m in enumerate(messages, 1)
    )


def count_decisions(messages: list[Turn]) -> int:
    return sum(
        1 for m in messages
        if m["role"] == "orchestrator" and m["decision"]["next_role"] != "human"
    )


def count_pauses(messages: list[Turn]) -> int:
    return sum(
        1 for m in messages
        if m["role"] == "orchestrator" and m["decision"]["next_role"] == "human"
    )


def has_approved_executor_plan(messages: list[Turn]) -> bool:
    """Whether the latest scientist plan has an explicit human approval to run."""
    latest_plan = max(
        (i for i, message in enumerate(messages) if message["role"] == "scientist"),
        default=-1,
    )
    latest_approval: Optional[tuple[int, bool]] = None
    for i, message in enumerate(messages):
        if i <= latest_plan or message["role"] != "orchestrator":
            continue
        decision = message.get("decision", {})
        if (
            decision.get("next_role") != "human"
            or decision.get("kind") != "hard"
            or decision.get("then") != "executor"
        ):
            continue
        response = next(
            (turn for turn in messages[i + 1:] if turn["role"] == "human"),
            None,
        )
        if response is None:
            continue
        approved = response.get("answers") == "hard" and response["text"].strip().lower() in ("y", "yes")
        latest_approval = (i, approved)
    return latest_approval[1] if latest_approval else False


def last_decision(messages: list[Turn]) -> Optional[dict[str, Any]]:
    for m in reversed(messages):
        if m["role"] == "orchestrator" and "decision" in m:
            return m["decision"]
    return None


def verified_test_rmse(messages: list[Turn]) -> Optional[float]:
    """Latest test RMSE that an executor turn obtained from a script that ran and exited 0."""
    for m in reversed(messages):
        if m["role"] == "executor" and m.get("test_rmse") is not None:
            return m["test_rmse"]
    return None
