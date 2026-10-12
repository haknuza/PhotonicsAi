#!/usr/bin/env python
"""Give the four-role system Task 5 and let it try to solve it.

    export LLM_MODEL=qwen2.5-coder:14b          # any OpenAI-compatible server, see agent_system/llm.py
    python run_task5.py

Everything of the run lands in runs/<timestamp>/ :
    work/            the scripts the executor wrote and ran, with their .stdout.txt / .stderr.txt
    transcript.md    who said what, turn by turn (put this in the PDF)
    transcript.json  the same, machine-readable
    summary.json     status, route, scripts run, and the RMSE a script really printed
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

from agent_system import MAX_DECISIONS, build_graph, write_transcript
from agent_system.llm import build_llm

HERE = Path(__file__).parent


def default_mat_path() -> Path:
    candidates = (
        HERE / "data" / "SNLS80mV.mat",
        HERE.parent.parent / "Task6" / "silverbox-agents" / "data" / "SNLS80mV.mat",
    )
    return next((path.resolve() for path in candidates if path.is_file()), candidates[0].resolve())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mat", help="path to SNLS80mV.mat (default: local data/ or Task 6 data/)")
    ap.add_argument("--model", help="model name (default: $LLM_MODEL)")
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint (default: $LLM_BASE_URL or local Ollama)")
    ap.add_argument("--runs-dir", default=str(HERE / "runs"))
    ap.add_argument("--task", default=str(HERE / "task5_problem.md"))
    ap.add_argument("--max-decisions", type=int, default=MAX_DECISIONS)
    ap.add_argument("--script-timeout", type=float, default=300, help="seconds per executed script (spec: 300)")
    args = ap.parse_args()

    mat = Path(args.mat).expanduser().resolve() if args.mat else default_mat_path()
    if not mat.is_file():
        sys.exit(f"data file not found: {mat}\nDownload SNLS80mV.mat from https://www.nonlinearbenchmark.org/")

    run_dir = Path(args.runs_dir) / time.strftime("%Y%m%d-%H%M%S")
    work_dir = run_dir / "work"
    work_dir.mkdir(parents=True)

    task = Path(args.task).read_text(encoding="utf-8")
    task += ("\n\nEnvironment note: the data file is at the path in the environment variable DATA_FILE "
             "when a script runs. Each script has a hard limit of five minutes on CPU.\n")

    graph = build_graph(build_llm(args.model, args.base_url), work_dir,
                        max_decisions=args.max_decisions, timeout_s=args.script_timeout,
                        env={"DATA_FILE": str(mat)})

    # The user speaks only to the orchestrator: the dialogue starts with one user turn, and the first edge
    # out of START goes to the orchestrator node.
    messages = [{"role": "user", "text": task}]
    shown, error = 0, None
    try:
        for state in graph.stream({"messages": messages}, stream_mode="values",
                                  config={"recursion_limit": 6 * args.max_decisions + 10}):
            messages = state["messages"]
            for m in messages[shown:]:
                if m["role"] == "orchestrator":
                    print(f"[orchestrator] -> {m['decision']['next_role']}: {m['decision']['reason']}", flush=True)
                elif m["role"] == "executor":
                    runs = "; ".join(f"{r['file']} exit {r['exit_code']}" for r in m.get("runs", [])) or "no run"
                    print(f"[executor] {runs}  test_rmse={m.get('test_rmse')}", flush=True)
                elif m["role"] != "user":
                    print(f"[{m['role']}] {len(m['text'])} characters", flush=True)
            shown = len(messages)
    except Exception:                                    # keep the partial dialogue; it is the evidence
        error = traceback.format_exc()
        print(error, file=sys.stderr)

    summary = write_transcript(messages, run_dir, args.max_decisions, extra={"error": error})
    print("\n" + "\n".join(f"{k}: {v}" for k, v in summary.items() if k != "error"))
    print(f"\ntranscript: {run_dir / 'transcript.md'}")
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
