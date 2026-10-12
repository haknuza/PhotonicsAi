"""A scripted stand-in for the chat model, so graph wiring and guards can be tested without an LLM.

It answers according to which role's system prompt it receives, popping one scripted reply per call.
It records, for every call, the role and whether the model had tools bound.
"""
import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROLE_MARKERS = {"ORCHESTRATOR": "orchestrator", "SCIENTIST": "scientist", "GENERAL": "general", "EXECUTOR": "executor"}


class ScriptedLLM:
    def __init__(self, script, bound=False, log=None):
        self.script, self.bound = script, bound
        self.log = log if log is not None else []

    def bind_tools(self, tools, **kw):
        assert [t.name for t in tools] == ["write_and_run_python"]
        return ScriptedLLM(self.script, True, self.log)

    def invoke(self, messages, **kw):
        first = messages[0].content
        role = next(r for marker, r in ROLE_MARKERS.items() if first.startswith(f"You are the {marker}"))
        self.log.append((role, self.bound))
        item = self.script[role].pop(0)
        if callable(item):
            item = item(messages)
        return item if isinstance(item, AIMessage) else AIMessage(content=item)


def decision(next_role, instruction="do it", reason="because"):
    return json.dumps({"next_role": next_role, "instruction": instruction, "reason": reason})


def tool_call(code, filename="fit.py", call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": "write_and_run_python", "id": call_id,
                                              "args": {"filename": filename, "code": code}}])


OK_SCRIPT = "print('TRAIN_RMSE=0.0123')\nprint('TEST_RMSE=0.0456')\n"


@pytest.fixture
def workdir(tmp_path):
    return tmp_path / "work"
