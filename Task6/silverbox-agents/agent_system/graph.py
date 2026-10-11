"""The LangGraph StateGraph: four nodes sharing one dialogue.

    START -> orchestrator --(model decides)--> scientist | general | executor | END
                  ^                                  |         |         |
                  +----------------------------------+---------+---------+   (edge returns the turn)

Each node reads the messages already in the state, including what the other roles wrote, and appends
exactly one turn of its own.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

from . import prompts
from .coding_tool import DEFAULT_TIMEOUT_S, CodingTool
from .state import (NEXT_ROLES, SPECIALISTS, DialogueState, count_decisions,
                    last_decision, render_dialogue, verified_test_rmse)

MAX_DECISIONS = 8
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def content_text(resp: Any) -> str:
    """Text of a chat-model reply, whatever shape the provider returns."""
    c = getattr(resp, "content", resp)
    if isinstance(c, str):
        return c
    return "".join(b if isinstance(b, str) else b.get("text", "") for b in c)


def clean(text: str) -> str:
    """Drop <think> blocks that reasoning models emit; they are not part of the dialogue."""
    return _THINK.sub("", text).strip()


class OrchestratorError(RuntimeError):
    pass


class Decision(BaseModel):
    next_role: str
    instruction: str = ""
    reason: str = ""

    @field_validator("next_role")
    @classmethod
    def _known_role(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in NEXT_ROLES:
            raise ValueError(f"next_role must be one of {NEXT_ROLES}, got {v!r}")
        return v


def parse_decision(text: str) -> Decision:
    """First JSON object in the reply that has a next_role (or next) key."""
    text = clean(text)
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("next_role" in obj or "next" in obj):
            obj.setdefault("next_role", obj.get("next"))
            return Decision(**{k: obj[k] for k in ("next_role", "instruction", "reason") if k in obj})
    raise ValueError("no JSON object with a next_role field found")


def ask_for_decision(llm, system: str, user: str, can_finish: bool, attempts: int = 4) -> Decision:
    """A model call that must return a valid Decision. Rejected answers are re-asked with feedback.

    There is deliberately no keyword fallback: if the model cannot produce a valid decision, the run
    fails loudly instead of being steered by a script.
    """
    feedback = ""
    for _ in range(attempts):
        resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user + feedback)])
        try:
            d = parse_decision(content_text(resp))
        except ValueError as e:
            feedback = f"\n\nYour previous answer was rejected: {e}. Answer with the JSON object only."
            continue
        if d.next_role == "finish" and not can_finish:
            feedback = ("\n\nYour previous answer was rejected: you may finish only after an executor turn "
                        "contains a test RMSE printed by a script that ran and exited 0, and the STATUS "
                        "line says none exists yet. Choose scientist, general or executor.")
            continue
        if d.next_role != "finish" and not d.instruction.strip():
            feedback = "\n\nYour previous answer was rejected: the instruction is empty."
            continue
        if "```" in d.instruction:
            feedback = "\n\nYour previous answer was rejected: the instruction contains code. You write no code."
            continue
        return d
    raise OrchestratorError(f"orchestrator produced no valid decision in {attempts} attempts")


# ----------------------------------------------------------------------------
# the coding tool as the model sees it
# ----------------------------------------------------------------------------
class _RunArgs(BaseModel):
    filename: str = Field(description="Short descriptive file name, e.g. 'mlp_rk4_fit.py'. A new numbered file is created on every call.")
    code: str = Field(description="The complete Python source of the script to write and run.")


def _tool_schema(tool: CodingTool) -> StructuredTool:
    def write_and_run_python(filename: str, code: str) -> str:
        return tool.write_and_run(filename, code).report()

    return StructuredTool.from_function(
        func=write_and_run_python, name="write_and_run_python", args_schema=_RunArgs,
        description=("Write the given Python source to a new file in the run's working directory and run it "
                     f"with a time limit of {int(tool.timeout_s)} seconds. Returns exit code, stdout and stderr."),
    )


# ----------------------------------------------------------------------------
# graph
# ----------------------------------------------------------------------------
def build_graph(llm, workdir: str | Path, *, max_decisions: int = MAX_DECISIONS,
                timeout_s: float = DEFAULT_TIMEOUT_S, env: Optional[dict[str, str]] = None,
                max_tool_calls_per_turn: int = 6):
    """Compile the four-node StateGraph.

    `llm` is any chat model with .invoke(messages) and .bind_tools(tools). Only the executor ever calls
    bind_tools; the other three nodes use the bare model.
    """
    coding_tool = CodingTool(workdir, timeout_s=timeout_s, env=env)
    tool_schema = _tool_schema(coding_tool)

    # ---- orchestrator ------------------------------------------------------
    def orchestrator(state: DialogueState):
        msgs = state["messages"]
        k = count_decisions(msgs)
        if k >= max_decisions:
            raise OrchestratorError("decision budget already spent")
        verified = verified_test_rmse(msgs)
        status = (f"STATUS: this will be decision {k + 1} of at most {max_decisions}. "
                  + (f"A script that exited 0 printed test RMSE = {verified:.6g}; you may finish."
                     if verified is not None else
                     "No executor turn has a verified test RMSE yet; you may NOT finish."))
        user = f"{render_dialogue(msgs)}\n\n---\n{status}\nReturn your decision as the JSON object."
        d = ask_for_decision(llm, prompts.ORCHESTRATOR, user, can_finish=verified is not None)
        text = f"NEXT: {d.next_role}\nINSTRUCTION: {d.instruction}\nREASON: {d.reason}"
        return {"messages": [{"role": "orchestrator", "text": text, "decision": d.model_dump()}]}

    # ---- scientist, general: plain model calls, no tools -------------------
    def specialist(role: str, system: str):
        def node(state: DialogueState):
            msgs = state["messages"]
            d = last_decision(msgs)
            instr = d["instruction"] if d else "(none)"
            user = (f"{render_dialogue(msgs)}\n\n---\nYou are the {role.upper()}. "
                    f"The orchestrator's instruction to you: {instr}")
            reply = clean(content_text(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])))
            return {"messages": [{"role": role, "text": reply or "(empty reply)"}]}
        return node

    # ---- executor: the only node with the coding tool ----------------------
    def executor(state: DialogueState):
        msgs = state["messages"]
        d = last_decision(msgs)
        instr = d["instruction"] if d else "(none)"
        user = (f"{render_dialogue(msgs)}\n\n---\nYou are the EXECUTOR. The orchestrator's instruction to you: {instr}\n"
                f"Working directory for your scripts: {coding_tool.workdir}")
        convo = [SystemMessage(content=prompts.EXECUTOR), HumanMessage(content=user)]
        bound = llm.bind_tools([tool_schema])

        runs, n_calls, summary = [], 0, ""
        for _ in range(2 * max_tool_calls_per_turn + 2):
            try:
                resp = bound.invoke(convo)
            except Exception as e:
                if "tool_use_failed" not in str(e):
                    raise
                convo.append(HumanMessage(content="Your tool call was cut off and could not be parsed. "
                                                  "Write a much shorter script and call the tool again."))
                continue
            convo.append(resp)
            calls = list(getattr(resp, "tool_calls", None) or [])
            if not calls:
                summary = clean(content_text(resp))
                break
            for call in calls:
                args = call.get("args") or {}
                if n_calls >= max_tool_calls_per_turn:
                    out = "Tool-call limit for this turn reached; nothing was run."
                elif not isinstance(args.get("code"), str):
                    out = "Error: the argument 'code' (the full Python source as a string) is required; nothing was run."
                else:
                    n_calls += 1
                    result = coding_tool.write_and_run(str(args.get("filename") or "script.py"), args["code"])
                    runs.append(result)
                    out = result.report()
                convo.append(ToolMessage(content=out, tool_call_id=call["id"], name=call["name"]))
            if n_calls >= max_tool_calls_per_turn:
                convo.append(HumanMessage(content="The tool-call limit for this turn is reached. Reply in plain text "
                                                  "with what happened. Do not call the tool again."))
                try:
                    summary = clean(content_text(bound.invoke(convo)))
                except Exception as e:
                    if "tool_use_failed" not in str(e):
                        raise
                    convo.append(HumanMessage(content="Your tool call was cut off and could not be parsed. "
                                                      "Write a much shorter script and call the tool again."))
                    continue
                break
        summary = summary or "(the executor wrote no summary)"

        good = [r for r in runs if r.test_rmse is not None]       # exit 0 AND a TEST_RMSE line
        record = "\n\n".join(f"=== RUN {i} - tool output recorded by the harness (authoritative) ===\n{r.report()}"
                             for i, r in enumerate(runs, 1)) or "(no script was run in this turn)"
        turn = {
            "role": "executor",
            "text": f"EXECUTOR'S SUMMARY (written by the model):\n{summary}\n\n{record}",
            "runs": [r.as_record() for r in runs],
            "test_rmse": good[-1].test_rmse if good else None,
            "train_rmse": good[-1].train_rmse if good else None,
        }
        return {"messages": [turn]}

    # ---- routing -------------------------------------------------------------
    def after_orchestrator(state: DialogueState) -> str:
        nxt = last_decision(state["messages"])["next_role"]
        return END if nxt == "finish" else nxt

    def after_specialist(state: DialogueState) -> str:
        # the edge that returns the turn; the run also stops once the decision budget is spent
        return END if count_decisions(state["messages"]) >= max_decisions else "orchestrator"

    g = StateGraph(DialogueState)
    g.add_node("orchestrator", orchestrator)
    g.add_node("scientist", specialist("scientist", prompts.SCIENTIST))
    g.add_node("general", specialist("general", prompts.GENERAL))
    g.add_node("executor", executor)

    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", after_orchestrator,
                            {"scientist": "scientist", "general": "general", "executor": "executor", END: END})
    for role in SPECIALISTS:
        g.add_conditional_edges(role, after_specialist, {"orchestrator": "orchestrator", END: END})
    return g.compile()
