import pytest
from langchain_core.messages import AIMessage

from agent_system import OrchestratorError, build_graph, summarize
from agent_system.coding_tool import CodingTool
from agent_system.graph import parse_decision
from agent_system.state import count_decisions
from conftest import OK_SCRIPT, ScriptedLLM, decision, tool_call

USER = [{"role": "user", "text": "Do Task 5."}]


def run(script, workdir, **kw):
    llm = ScriptedLLM(script)
    graph = build_graph(llm, workdir, **kw)
    state = graph.invoke({"messages": list(USER)}, config={"recursion_limit": 80})
    return state["messages"], llm.log


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
def test_parse_decision_handles_think_blocks_and_fences():
    text = '<think>hmm {"next_role": "finish"}</think>\n```json\n{"next_role": "Scientist", "instruction": "plan", "reason": "r"}\n```'
    d = parse_decision(text)
    assert d.next_role == "scientist" and d.instruction == "plan"
    with pytest.raises(ValueError):
        parse_decision('{"next_role": "banana"}')


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


def test_tool_call_without_code_runs_nothing(workdir):
    bad = AIMessage(content="", tool_calls=[{"name": "write_and_run_python", "id": "c1", "args": {"filename": "f.py"}}])
    script = {"orchestrator": [decision("executor"), decision("finish"), decision("finish"),
                               decision("finish"), decision("finish")],
              "executor": [bad, "could not run"]}
    with pytest.raises(OrchestratorError):                 # nothing ran, so finishing stays blocked
        run(script, workdir)
