"""The coding tool. Only the executor is given it.

One call does both required things:
  1. write a Python file into the working directory chosen for this run, and
  2. run that file with a time limit (default five minutes), returning the exit
     code, the standard output and the standard error.

Every call writes a *new* numbered file (01_..., 02_...), so a repair never
overwrites the script it repairs and the directory is an audit trail.

WARNING: this runs model-written code with your user's privileges. It is not a
sandbox. Run the whole project inside a container or VM if you need isolation.
"""
from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

DEFAULT_TIMEOUT_S = 300          # five minutes
TAIL_CHARS = 3_000               # how much of stdout/stderr goes into the dialogue

# Output contract: a line that is exactly NAME=<number> (":" also accepted).
_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"
_TEST = re.compile(rf"^\s*TEST_RMSE\s*[=:]\s*{_NUM}\s*$", re.I | re.M)
_TRAIN = re.compile(rf"^\s*TRAIN_RMSE\s*[=:]\s*{_NUM}\s*$", re.I | re.M)


def parse_metric(pattern: re.Pattern, stdout: str) -> Optional[float]:
    """Last matching line of stdout, as a finite non-negative float, else None."""
    matches = pattern.findall(stdout)
    if not matches:
        return None
    value = float(matches[-1])
    return value if math.isfinite(value) and value >= 0 else None


def _tail(text: str, limit: int = TAIL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"...[{len(text) - limit} earlier characters omitted; full text is in the .stdout/.stderr file]...\n{text[-limit:]}"


@dataclass
class RunResult:
    file: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    test_rmse: Optional[float]     # only set if the script exited 0 and printed TEST_RMSE=<x>
    train_rmse: Optional[float]

    def as_record(self) -> dict:
        return asdict(self)

    def report(self) -> str:
        status = "TIMED OUT" if self.timed_out else f"exit code {self.exit_code}"
        parts = [
            f"file: {self.file}",
            f"result: {status} after {self.duration_s:.1f} s",
            f"--- stdout ---\n{_tail(self.stdout) or '(empty)'}",
            f"--- stderr ---\n{_tail(self.stderr) or '(empty)'}",
        ]
        if self.test_rmse is not None:
            parts.append(f"printed TEST_RMSE = {self.test_rmse:.6g}"
                         + (f", TRAIN_RMSE = {self.train_rmse:.6g}" if self.train_rmse is not None else ""))
        return "\n".join(parts)


class CodingTool:
    def __init__(self, workdir: str | Path, timeout_s: float = DEFAULT_TIMEOUT_S,
                 env: Optional[dict[str, str]] = None):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.env = {**os.environ, **(env or {})}

    # -- step 1: write ---------------------------------------------------------
    def _next_path(self, filename: str) -> Path:
        stem = Path(filename).name                       # no directories, no "../"
        stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem) or "script.py"
        if not stem.endswith(".py"):
            stem += ".py"
        used = [int(m.group(1)) for p in self.workdir.glob("*.py")
                if (m := re.match(r"(\d+)_", p.name))]
        return self.workdir / f"{(max(used) + 1 if used else 1):02d}_{stem}"

    # -- steps 1 + 2 -----------------------------------------------------------
    def write_and_run(self, filename: str, code: str) -> RunResult:
        path = self._next_path(filename)
        path.write_text(code, encoding="utf-8")

        t0 = time.monotonic()
        proc = subprocess.Popen(
            [sys.executable, str(path)], cwd=self.workdir, env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True,                      # so we can kill child processes too
        )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
        duration = time.monotonic() - t0

        exit_code = 124 if timed_out else proc.returncode
        path.with_suffix(".stdout.txt").write_text(stdout or "", encoding="utf-8")
        path.with_suffix(".stderr.txt").write_text(stderr or "", encoding="utf-8")

        ok = exit_code == 0 and not timed_out
        return RunResult(
            file=path.name, exit_code=exit_code, stdout=stdout or "", stderr=stderr or "",
            timed_out=timed_out, duration_s=duration,
            test_rmse=parse_metric(_TEST, stdout or "") if ok else None,
            train_rmse=parse_metric(_TRAIN, stdout or "") if ok else None,
        )
