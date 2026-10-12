"""Summary and transcript of a run: the evidence for 'which agent did what, and what error it printed'."""
from __future__ import annotations

import json
from pathlib import Path

from .state import Turn, count_decisions, last_decision, verified_test_rmse


def summarize(messages: list[Turn], max_decisions: int) -> dict:
    d = last_decision(messages)
    finished = bool(d and d["next_role"] == "finish")
    n = count_decisions(messages)
    last_ok = next((m for m in reversed(messages) if m["role"] == "executor" and m.get("test_rmse") is not None), None)
    return {
        "status": "finished" if finished else ("decision_budget_spent" if n >= max_decisions else "stopped_early"),
        "decisions_used": n,
        "route": [m["decision"]["next_role"] for m in messages if m["role"] == "orchestrator"],
        "verified_test_rmse": verified_test_rmse(messages),
        "verified_train_rmse": last_ok.get("train_rmse") if last_ok else None,
        "scripts_run": [r["file"] for m in messages if m["role"] == "executor" for r in m.get("runs", [])],
    }


def to_markdown(messages: list[Turn], summary: dict) -> str:
    out = ["# Run transcript", "",
           f"- status: **{summary['status']}**, decisions used: {summary['decisions_used']}",
           f"- route chosen by the orchestrator: {' -> '.join(summary['route']) or '(none)'}",
           f"- verified free-run test RMSE (printed by a script that exited 0): "
           f"**{summary['verified_test_rmse']}**, train RMSE: {summary['verified_train_rmse']}", "",
           "| turn | role | what it did |", "|---|---|---|"]
    for i, m in enumerate(messages, 1):
        if m["role"] == "orchestrator":
            what = f"decided next = {m['decision']['next_role']}"
        elif m["role"] == "executor":
            what = "; ".join(f"{r['file']} exit {r['exit_code']}" for r in m.get("runs", [])) or "ran nothing"
        else:
            what = "spoke"
        out.append(f"| {i} | {m['role']} | {what} |")
    for i, m in enumerate(messages, 1):
        out += ["", f"## Turn {i} - {m['role']}", "", m["text"]]
    return "\n".join(out) + "\n"


def write_transcript(messages: list[Turn], run_dir: str | Path, max_decisions: int, extra: dict | None = None) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {**summarize(messages, max_decisions), **(extra or {})}
    (run_dir / "transcript.json").write_text(json.dumps(messages, indent=2), encoding="utf-8")
    (run_dir / "transcript.md").write_text(to_markdown(messages, summary), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
