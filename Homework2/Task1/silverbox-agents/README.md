# Four-role agentic system (LangGraph) - Homework 1, Task 6

A `StateGraph` of four nodes - **orchestrator, scientist, general, executor** - that share one state: the
dialogue. Each turn is `{role, text}`. A node reads every message already in the state (including what the
other roles wrote) and appends exactly one turn of its own. The system is validated by giving it Task 5
(Silverbox PINN) and letting it try to solve it.

```
START -> orchestrator --(model decision)--> scientist | general | executor | END
              ^                                  |          |         |
              +----------------------------------+----------+---------+   (edge returns the turn)
```

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q                                   # 20 tests, no LLM needed (scripted model)

# any OpenAI-compatible server for an open-source model, e.g. Ollama:
export LLM_MODEL=qwen2.5-coder:14b                    # must support native tool calling
export LLM_BASE_URL=http://localhost:11434/v1         # default
python run_task5.py --mat /path/to/SNLS80mV.mat       # data: https://www.nonlinearbenchmark.org/
```

Ollama note: its default context window is small. Transcripts include plans, code and stdout, so start the
server with a context of at least 16k tokens (e.g. `OLLAMA_CONTEXT_LENGTH=16384`).

Each run writes `runs/<timestamp>/`: `work/` (every script the executor wrote and ran, plus its
`.stdout.txt`/`.stderr.txt`), `transcript.md` (which agent did what, turn by turn - use it in the PDF),
`transcript.json`, `summary.json`.

## How the spec is met

| Requirement | Where |
|---|---|
| StateGraph, four nodes, state = dialogue, each turn has role name + text | `agent_system/state.py`, `graph.py` |
| Orchestrator decision = {next role, instruction, one-sentence reason}, made by a model call; no keyword table or fixed script | `ask_for_decision` in `graph.py` (invalid answers are re-asked; after 4 failures the run raises instead of falling back to rules) |
| Orchestrator writes no code, has no coding tool | it only ever gets the bare model; an instruction containing a code fence is rejected and re-asked |
| Stops on `finish` or after 8 decisions | `after_orchestrator` / `after_specialist` edges. The 8th decision's specialist still runs, then the graph ends. |
| May finish only after an executor turn holds a test RMSE printed by a script that ran | `can_finish` guard. The flag lives in the turn and is set by harness code only when the script **exited 0** and its stdout has a line `TEST_RMSE=<finite, non-negative float>`. Text the model writes cannot set it. |
| Scientist: theory + plan, no coding tool, no unmeasured numbers | `prompts.SCIENTIST`; plain model call |
| General: high-level questions only, no tool, no plan | `prompts.GENERAL`; plain model call |
| Executor is the only role with the tool; tool writes a file and runs it (5-min limit) and returns exit code, stdout, stderr | `coding_tool.py`; `bind_tools` is called only inside the executor node |
| Executor implements the latest plan in a new file; repairs keep the same science | every call writes a new numbered file (`01_`, `02_`, ...); prompt forbids changing the equation or the free-run test |

## Things you should know before you rely on it

- **Decision counting.** A rejected answer (premature `finish`, invalid JSON, code in the instruction) is
  re-asked and is not counted; only the accepted decision is a turn and counts toward the 8.
- **The RMSE guard checks that a script ran and printed the number - not that the script computes the right
  free-run RMSE.** A model could write a script that prints a made-up value. The audit trail (`work/*.py`)
  is there so you can read the kept script and check the free-run protocol yourself (initial state,
  no reading of `y` on the test segment, same state carried into the test segment). Do this before putting
  any number in the PDF.
- **Code execution is not sandboxed.** Model-written scripts run as your user. Use a container/VM if the
  model is untrusted.
- **Executor needs tool calling.** If your open-source model/server does not support it, the executor node
  will see no tool calls and nothing will run, so the orchestrator will never be allowed to finish.
- **Compute.** Five minutes of CPU per script is tight for neural-ODE training; the scientist prompt says so
  and asks for batched short rollout windows. A weak model may need several repair rounds, and 8 decisions
  is a small budget - a run that ends as `decision_budget_spent` is a legitimate outcome to report honestly.
