import pytest
from langchain_core.messages import AIMessage

from agent_system import OrchestratorError, build_graph, summarize
from agent_system.coding_tool import CodingTool
from agent_system.graph import parse_decision
from agent_system.prompts import ORCHESTRATOR
from agent_system.report import to_markdown
from agent_system.state import (count_decisions, count_pauses, has_approved_executor_plan,
                                render_dialogue)
from conftest import OK_SCRIPT, ScriptedLLM, decision, tool_call
from langgraph.types import Command

USER = [{"role": "user", "text": "Do Task 5."}]


def run(script, workdir, **kw):
    human_responses = iter(kw.pop("human_responses", []))
    llm = ScriptedLLM(script)
    graph = build_graph(llm, workdir, **kw)
    config = {"configurable": {"thread_id": str(workdir)}, "recursion_limit": 80}
    graph_input = {"messages": list(USER)}
    while True:
        for _state in graph.stream(graph_input, stream_mode="values", config=config):
            pass
        snapshot = graph.get_state(config)
        pending = [item for task in snapshot.tasks for item in task.interrupts]
        if not pending:
            break
        graph_input = Command(resume=next(human_responses))
    return snapshot.values["messages"], llm.log


# ------------------------------------------------------------------ coding tool
def test_tool_writes_new_numbered_file_each_call_and_returns_outputs(workdir):
    tool = CodingTool(workdir)
    a = tool.write_and_run("../../evil.py", "print('hi')\nimport sys; sys.stderr.write('warn')")
    b = tool.write_and_run("evil.py", "print('TEST_RMSE=0.5')")
    assert a.file == "01_evil.py" and b.file == "02_evil.py"       # no path traversal, never overwritten
    assert (workdir / "01_evil.py").exists() and (workdir / "02_evil.py").exists()
    assert (a.exit_code, a.stdout.strip(), a.stderr.strip()) == (0, "hi", "warn")
    assert b.test_rmse == 0.5


def test_tool_truncates_dialogue_output_but_saves_full_output(workdir):
    code = "import sys\nsys.stdout.write('x' * 3500)\nsys.stderr.write('y' * 3500)"
    result = CodingTool(workdir).write_and_run("verbose.py", code)

    assert len(result.stdout) == 3500 and len(result.stderr) == 3500
    assert (workdir / "01_verbose.stdout.txt").read_text() == result.stdout
    assert (workdir / "01_verbose.stderr.txt").read_text() == result.stderr
    report = result.report()
    assert "500 earlier characters omitted" in report
    assert "x" * 3000 in report and "x" * 3001 not in report
    assert "y" * 3000 in report and "y" * 3001 not in report


def test_tool_returns_traceback_and_no_rmse_when_script_fails(workdir):
    r = CodingTool(workdir).write_and_run("x.py", "print('TEST_RMSE=0.1')\nraise ValueError('boom')")
    assert r.exit_code == 1 and "ValueError: boom" in r.stderr
    assert r.test_rmse is None                                      # printed, but the script did not succeed


def test_tool_time_limit_kills_the_script(workdir):
    r = CodingTool(workdir, timeout_s=1).write_and_run("slow.py", "import time\nprint('start', flush=True)\ntime.sleep(30)")
    assert r.timed_out and r.exit_code == 124 and r.duration_s < 10 and r.test_rmse is None


@pytest.mark.parametrize("line", ["TEST_RMSE=nan", "TEST_RMSE=-1", "TEST_RMSE=inf", "test rmse is 0.1"])
def test_tool_rejects_non_finite_negative_or_malformed_rmse(workdir, line):
    assert CodingTool(workdir).write_and_run("x.py", f"print({line!r})").test_rmse is None


# ------------------------------------------------------------------ parsing
def test_orchestrator_checks_ambiguity_before_routing_rules():
    check = "FIRST, before applying any routing rule below:"
    assert ORCHESTRATOR.index(check) < ORCHESTRATOR.index("Routing rules:")
    assert "7. AMBIGUITY. Apply the FIRST check above." in ORCHESTRATOR
    assert "Always list the plausible readings of the latest user request first." in ORCHESTRATOR
    contract = '{"readings": ["<each plausible reading of the latest user request, 1 to 3 short items>"],'
    assert contract in ORCHESTRATOR


def test_parse_decision_handles_think_blocks_and_fences():
    text = '<think>hmm {"next_role": "finish"}</think>\n```json\n{"next_role": "Scientist", "instruction": "plan", "reason": "r"}\n```'
    d = parse_decision(text)
    assert d.next_role == "scientist" and d.instruction == "plan" and d.readings == []
    with_readings = parse_decision(
        '{"readings": ["explain the result", "run the evaluation"], "next_role": "general"}'
    )
    assert with_readings.readings == ["explain the result", "run the evaluation"]
    with pytest.raises(ValueError):
        parse_decision('{"next_role": "banana"}')


def test_parse_human_decision_fields_and_required_fields():
    d = parse_decision(decision(
        "human", instruction="run it", kind="hard", question="May I run it?", then="executor"
    ))
    assert (d.next_role, d.kind, d.question, d.then, d.instruction) == (
        "human", "hard", "May I run it?", "executor", "run it"
    )

    for invalid in (
        decision("human", kind="hard", then="executor", instruction="run it"),
        decision("human", kind="hard", question="May I?", instruction="run it"),
        decision("human", kind="unknown", question="May I?", then="executor", instruction="run it"),
        decision("human", kind="hard", question="May I?", then="human", instruction="run it"),
        decision("human", kind="hard", question="May I?", then="general", instruction="run it"),
    ):
        with pytest.raises(ValueError):
            parse_decision(invalid)


def test_decision_and_pause_counts_are_separate():
    messages = [
        {"role": "orchestrator", "decision": {"next_role": "human"}},
        {"role": "human", "answers": "ambiguity"},
        {"role": "orchestrator", "decision": {"next_role": "executor"}},
    ]

    assert count_decisions(messages) == 1
    assert count_pauses(messages) == 1


def test_executor_approval_status_tracks_latest_scientist_plan():
    messages = [
        {"role": "scientist", "text": "Plan A"},
        {"role": "orchestrator", "decision": {
            "next_role": "human", "kind": "hard", "then": "executor",
        }},
        {"role": "human", "text": "yes", "answers": "hard"},
    ]
    assert has_approved_executor_plan(messages)

    messages.append({"role": "scientist", "text": "Revised plan B"})
    assert not has_approved_executor_plan(messages)

    messages.extend([
        {"role": "orchestrator", "decision": {
            "next_role": "human", "kind": "hard", "then": "executor",
        }},
        {"role": "human", "text": "no", "answers": "hard"},
    ])
    assert not has_approved_executor_plan(messages)


def test_render_dialogue_clips_each_turn_at_six_thousand_characters():
    rendered = render_dialogue([{"role": "scientist", "text": "x" * 7000}])

    assert "[1000 characters omitted]" in rendered
    assert len(rendered) < 6100


def test_render_dialogue_supports_per_role_limits_without_clipping_the_task():
    task = "task " * 850
    messages = [
        {"role": "user", "text": task},
        {"role": "orchestrator", "text": "o" * 1000},
        {"role": "human", "text": "h" * 1000},
        {"role": "scientist", "text": "s" * 7000},
    ]

    rendered = render_dialogue(messages, limits={"orchestrator": 500, "human": 300})

    assert task in rendered
    assert "[500 characters omitted]" in rendered
    assert "[700 characters omitted]" in rendered
    assert "[1000 characters omitted]" in rendered
    assert "[turn 2] ORCHESTRATOR:" in rendered
    assert "[turn 3] HUMAN:" in rendered
    assert "[turn 4] SCIENTIST:" in rendered


def test_transcript_handles_human_harness_and_unknown_roles():
    messages = [
        {"role": "user", "text": "Task"},
        {"role": "human", "text": "yes"},
        {"role": "harness", "text": "REFUSED: no action was run."},
        {"role": "reviewer", "text": "Review note"},
    ]

    transcript = to_markdown(messages, {
        "status": "stopped_early",
        "decisions_used": 0,
        "route": [],
        "verified_test_rmse": None,
        "verified_train_rmse": None,
    })

    assert "| 2 | human | yes |" in transcript
    assert "| 3 | harness | REFUSED: no action was run. |" in transcript
    assert "| 4 | reviewer | spoke |" in transcript


# ------------------------------------------------------------------ graph behaviour
def happy_script():
    return {
        "orchestrator": [decision("scientist", "write the plan"), decision("executor", "implement the plan"),
                         decision("finish", "Test RMSE 0.0456 was printed.")],
        "scientist": ["1. equation ... 2. network ... 6. steps"],
        "executor": [tool_call(OK_SCRIPT), "Ran 01_fit.py, exit 0."],
    }


def test_executor_retries_tool_use_failed_with_shorter_script(workdir):
    def tool_use_failed(_messages):
        raise RuntimeError("400 tool_use_failed: could not parse tool call arguments")

    script = {
        "orchestrator": [decision("executor"), decision("finish")],
        "executor": [tool_use_failed, tool_call(OK_SCRIPT), "Ran the shorter script."],
    }
    msgs, _ = run(script, workdir)
    executor_turn = next(m for m in msgs if m["role"] == "executor")

    assert executor_turn["test_rmse"] == 0.0456
    assert [r["file"] for r in executor_turn["runs"]] == ["01_fit.py"]
    assert "shorter script" in executor_turn["text"]


def test_executor_retries_tool_use_failed_during_final_summary(workdir):
    def tool_use_failed(_messages):
        raise RuntimeError("400 tool_use_failed: could not parse tool call arguments")

    script = {
        "orchestrator": [decision("executor"), decision("finish")],
        "executor": [tool_call(OK_SCRIPT), tool_use_failed, "The script ran successfully."],
    }
    msgs, _ = run(script, workdir, max_tool_calls_per_turn=1)
    executor_turn = next(m for m in msgs if m["role"] == "executor")

    assert executor_turn["test_rmse"] == 0.0456
    assert "The script ran successfully." in executor_turn["text"]


def test_full_run_route_roles_and_verified_rmse(workdir):
    msgs, log = run(happy_script(), workdir)
    assert [m["role"] for m in msgs] == ["user", "orchestrator", "scientist", "orchestrator",
                                         "executor", "orchestrator"]
    s = summarize(msgs, 8)
    assert s["status"] == "finished" and s["route"] == ["scientist", "executor", "finish"]
    assert s["verified_test_rmse"] == 0.0456 and s["verified_train_rmse"] == 0.0123
    assert s["scripts_run"] == ["01_fit.py"]


def test_only_the_executor_gets_the_coding_tool(workdir):
    _, log = run(happy_script(), workdir)
    with_tools = {role for role, bound in log if bound}
    assert with_tools == {"executor"}
    assert {role for role, _ in log} == {"orchestrator", "scientist", "executor"}


def test_ambiguity_answer_returns_to_orchestrator(workdir):
    script = {
        "orchestrator": [
            decision("human", kind="ambiguity", question="Which split should I use?"),
            decision("general", instruction="Explain the split."),
        ],
        "general": ["The task specifies the split."],
    }
    msgs, _ = run(script, workdir, human_responses=["Use the specified split."], max_decisions=1)

    assert [m["role"] for m in msgs] == ["user", "orchestrator", "human", "orchestrator", "general"]
    assert msgs[2]["text"] == "Use the specified split."
    assert msgs[2]["answers"] == "ambiguity"


def test_hard_approval_hands_off_to_requested_role(workdir):
    orchestrator_prompts = []

    def request_approval(messages):
        orchestrator_prompts.append(messages[1].content)
        return decision(
            "human", instruction="run the planned script", kind="hard",
            question="May I run the planned script?", then="executor",
        )

    def capture_prompt(messages):
        orchestrator_prompts.append(messages[1].content)
        return decision("finish", instruction="Test RMSE 0.0456 was printed.")

    script = {
        "orchestrator": [
            request_approval,
            capture_prompt,
        ],
        "executor": [tool_call(OK_SCRIPT), "Ran the planned script."],
    }
    msgs, _ = run(script, workdir, human_responses=["yes"])

    assert [m["role"] for m in msgs] == ["user", "orchestrator", "human", "executor", "orchestrator"]
    assert msgs[2]["text"] == "yes"
    assert "Person has approved running the coding tool for the latest plan: no." in orchestrator_prompts[0]
    assert "Person has approved running the coding tool for the latest plan: yes." in orchestrator_prompts[1]
    assert summarize(msgs, 8)["verified_test_rmse"] == 0.0456


def test_hard_approval_reprompts_for_invalid_input(workdir):
    script = {
        "orchestrator": [
            decision("human", instruction="run the planned script", kind="hard",
                     question="May I run the planned script?", then="executor"),
            decision("general", instruction="Explain the result."),
        ],
        "executor": [tool_call(OK_SCRIPT), "The planned script ran."],
        "general": ["The result was recorded."],
    }
    msgs, _ = run(script, workdir, human_responses=["maybe"], max_decisions=1)

    assert msgs[2]["text"] == "maybe"
    assert msgs[2]["answers"] == "hard"
    assert msgs[3]["role"] == "harness"
    assert "REFUSED:" in msgs[3]["text"] and "NOT run" in msgs[3]["text"]
    assert "executor" not in [m["role"] for m in msgs]


def test_rejected_hard_approval_returns_to_orchestrator(workdir):
    script = {
        "orchestrator": [
            decision("human", instruction="run the planned script", kind="hard",
                     question="May I run the planned script?", then="executor"),
            decision("general", instruction="Explain the plan."),
        ],
        "general": ["The plan is ready."],
    }
    msgs, _ = run(script, workdir, human_responses=["no"], max_decisions=1)

    assert [m["role"] for m in msgs] == ["user", "orchestrator", "human", "harness", "orchestrator", "general"]
    assert msgs[2]["text"] == "no"
    assert "REFUSED:" in msgs[3]["text"] and "NOT run" in msgs[3]["text"]


def test_human_pause_cap_stops_repeated_questions_without_spending_decisions(workdir):
    script = {
        "orchestrator": [
            decision("human", kind="ambiguity", question=f"Clarify item {i}?")
            for i in range(4)
        ],
    }
    script["orchestrator"].extend([
        decision("human", kind="ambiguity", question="Clarify one more item?"),
        decision("general", instruction="Continue without asking another question."),
    ])
    script["general"] = ["Continue."]
    msgs, _ = run(script, workdir, human_responses=["continue"] * 4, max_decisions=1)
    summary = summarize(msgs, 8)

    assert count_decisions(msgs) == 1
    assert count_pauses(msgs) == 4
    assert summary["human_pauses_used"] == 4
    assert len([m for m in msgs if m["role"] == "human"]) == 4


def test_human_pause_does_not_spend_decision_budget(workdir):
    script = {
        "orchestrator": [
            decision("human", kind="ambiguity", question="Which option?"),
            decision("executor", instruction="Run the planned script."),
        ],
        "executor": [tool_call(OK_SCRIPT), "Ran successfully."],
    }
    msgs, _ = run(script, workdir, human_responses=["option A"], max_decisions=1)

    assert [m["role"] for m in msgs] == ["user", "orchestrator", "human", "orchestrator", "executor"]
    assert count_decisions(msgs) == 1
    assert count_pauses(msgs) == 1


def test_hard_approval_cannot_finish_without_verified_rmse(workdir):
    script = {
        "orchestrator": [
            decision("human", instruction="finish", kind="hard",
                     question="May I finish?", then="finish"),
            decision("general", instruction="Explain the current status."),
        ],
        "general": ["There is no verified result yet."],
    }
    msgs, _ = run(script, workdir, human_responses=["yes"], max_decisions=1)

    assert [m["role"] for m in msgs] == ["user", "orchestrator", "general"]
    assert summarize(msgs, 8)["status"] == "stopped_early"


def test_approved_hard_finish_still_requires_verified_rmse(workdir):
    script = {
        "orchestrator": [
            decision("executor", instruction="run the script"),
            decision("human", instruction="report the verified result", kind="hard",
                     question="May I finish?", then="finish"),
        ],
        "executor": [tool_call(OK_SCRIPT), "Ran successfully."],
    }
    msgs, _ = run(script, workdir, human_responses=["yes"])
    summary = summarize(msgs, 8)

    assert summary["status"] == "finished"
    assert summary["route"] == ["executor", "human", "finish"]
    assert summary["verified_test_rmse"] == 0.0456


def test_each_role_reads_what_the_others_wrote(workdir):
    seen = {}
    script = happy_script()
    script["executor"][0] = lambda m: (seen.update(executor=m[1].content), tool_call(OK_SCRIPT))[1]
    run(script, workdir)
    assert "1. equation ... 2. network ... 6. steps" in seen["executor"]       # scientist's plan is in the state
    assert "SCIENTIST:" in seen["executor"] and "ORCHESTRATOR:" in seen["executor"]


def test_premature_finish_is_rejected_and_reasked(workdir):
    script = happy_script()
    script["orchestrator"].insert(0, decision("finish", "I am done"))      # before anything ran
    msgs, _ = run(script, workdir)
    assert summarize(msgs, 8)["route"] == ["scientist", "executor", "finish"]   # the early finish never became a turn
    first_finish = next(i for i, m in enumerate(msgs) if m.get("decision", {}).get("next_role") == "finish")
    assert any(m["role"] == "executor" and m["test_rmse"] is not None for m in msgs[:first_finish])


def test_orchestrator_that_only_says_finish_fails_loudly_not_by_a_fallback(workdir):
    script = {"orchestrator": [decision("finish")] * 4}
    with pytest.raises(OrchestratorError):
        run(script, workdir)


def test_fabricated_rmse_in_model_text_does_not_unlock_finish(workdir):
    script = {
        "orchestrator": [decision("executor", "run it"), decision("finish"), decision("finish"),
                         decision("finish"), decision("finish")],
        "executor": ["I ran it. TEST_RMSE=0.001, great!"],                    # no tool call at all
    }
    with pytest.raises(OrchestratorError):
        run(script, workdir)


def test_failed_script_gives_no_verified_rmse_then_repair_succeeds(workdir):
    script = {
        "orchestrator": [decision("scientist"), decision("executor"), decision("executor", "repair"),
                         decision("finish", "done")],
        "scientist": ["plan"],
        "executor": [tool_call("print('TEST_RMSE=0.9')\nraise RuntimeError('bad')"), "It crashed.",
                     tool_call(OK_SCRIPT, call_id="c2"), "Repaired."],
    }
    msgs, _ = run(script, workdir)
    execs = [m for m in msgs if m["role"] == "executor"]
    assert execs[0]["test_rmse"] is None and "RuntimeError: bad" in execs[0]["text"]
    assert execs[1]["test_rmse"] == 0.0456
    assert [r["file"] for m in execs for r in m["runs"]] == ["01_fit.py", "02_fit.py"]   # repair is a new file


def test_stops_after_eight_decisions_without_finish(workdir):
    script = {"orchestrator": [decision("general", "explain") for _ in range(20)],
              "general": ["an explanation"] * 20}
    msgs, _ = run(script, workdir, max_decisions=8)
    assert count_decisions(msgs) == 8
    s = summarize(msgs, 8)
    assert s["status"] == "decision_budget_spent" and s["verified_test_rmse"] is None


def test_orchestrator_instruction_may_not_contain_code(workdir):
    script = happy_script()
    script["orchestrator"].insert(0, decision("scientist", "```python\nprint(1)\n```"))
    msgs, _ = run(script, workdir)
    assert all("```" not in m["decision"]["instruction"] for m in msgs if m["role"] == "orchestrator")


def test_executor_tool_call_limit_per_turn(workdir):
    # limit of 2: once it is reached the model is asked for a plain-text summary instead of more runs
    script = {
        "orchestrator": [decision("executor"), decision("finish")],
        "executor": [tool_call(OK_SCRIPT, call_id="c1"), tool_call(OK_SCRIPT, call_id="c2"), "stopped"],
    }
    msgs, _ = run(script, workdir, max_tool_calls_per_turn=2)
    ex = next(m for m in msgs if m["role"] == "executor")
    assert len(ex["runs"]) == 2 and "stopped" in ex["text"]


def test_parallel_tool_calls_beyond_the_limit_are_refused(workdir):
    three = AIMessage(content="", tool_calls=[
        {"name": "write_and_run_python", "id": f"c{i}", "args": {"filename": "f.py", "code": OK_SCRIPT}}
        for i in range(3)])
    script = {"orchestrator": [decision("executor"), decision("finish")], "executor": [three, "summary text"]}
    msgs, _ = run(script, workdir, max_tool_calls_per_turn=2)
    ex = next(m for m in msgs if m["role"] == "executor")
    assert len(ex["runs"]) == 2 and "summary text" in ex["text"]


def test_executor_keeps_harness_record_when_summary_hits_rate_limit(workdir):
    def fail_summary(_messages):
        raise RuntimeError("413 Request too large")

    script = {
        "orchestrator": [decision("executor"), decision("finish")],
        "executor": [tool_call(OK_SCRIPT), fail_summary],
    }
    msgs, _ = run(script, workdir, max_tool_calls_per_turn=1)
    ex = next(m for m in msgs if m["role"] == "executor")

    assert ex["test_rmse"] == 0.0456
    assert "(summary unavailable: provider rate limit)" in ex["text"]
    assert "TRAIN_RMSE=0.0123" in ex["text"]
    assert len(ex["runs"]) == 1


def test_executor_clips_model_tool_result_but_keeps_harness_record(workdir):
    def check_tool_result(messages):
        tool_message = next(message for message in messages if message.type == "tool")
        assert len(tool_message.content) == 1500
        return "Ran the script."

    code = "print('x' * 3500)\nprint('TEST_RMSE=0.2')\n"
    script = {
        "orchestrator": [decision("executor"), decision("finish")],
        "executor": [tool_call(code), check_tool_result],
    }
    msgs, _ = run(script, workdir, max_tool_calls_per_turn=1)
    ex = next(m for m in msgs if m["role"] == "executor")

    assert len(ex["runs"][0]["stdout"].splitlines()[0]) == 3500
    assert "x" * 2000 in ex["text"]


def test_tool_call_without_code_runs_nothing(workdir):
    bad = AIMessage(content="", tool_calls=[{"name": "write_and_run_python", "id": "c1", "args": {"filename": "f.py"}}])
    script = {"orchestrator": [decision("executor"), decision("finish"), decision("finish"),
                               decision("finish"), decision("finish")],
              "executor": [bad, "could not run"]}
    with pytest.raises(OrchestratorError):                 # nothing ran, so finishing stays blocked
        run(script, workdir)
