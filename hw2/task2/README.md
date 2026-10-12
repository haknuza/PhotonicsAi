# Homework 2, Task 2

## Prepared artifacts

- `runs/recorded-agent/task2_report.pdf`: report for the live recorded run,
  with measured settings, camera images and the lab-resource comparison.
- `runs/recorded-agent.webm`: genuine 7-minute recording of the dedicated
  window displaying the live model run; no desktop-wide capture or transcript replay.
- `runs/recorded-agent.log` and `runs/recorded-agent/transcript.json`: run evidence.
- `runs/final/task2_report.pdf`: earlier three-grid run, with best local enhancement
  4.979; its changed measurement regions limit comparisons.
- `runs/controlled-agent/task2_report.pdf`: additional controlled-geometry run.

The recorded run's best measured enhancement is 3.115. The model attempted an
invalid 8-pixel/16-flip configuration and repeated the 2-pixel configuration after
validation rejected it. Its narrative mislabels that repeated run; the report table
and summary use actual tool results. These are exploratory outcomes, not optimality
claims. All 45 simulator/agent tests passed. GitHub upload and email submission
remain separate steps.

Uses the Homework 1 four-role architecture and the shared `hw2/agent` model factory,
dialogue state, routing parser and coding tool. A task-specific LangGraph exposes
`focus_beam` only to the executor, and requires a scientist plan before experiments.
Task 1 remains unchanged. There are no human interrupts in this autonomous task.

```bash
python -m pip install -r hw2/agent/requirements.txt reportlab
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-task2 python hw2/task2/run_focus.py --model qwen2.5:3b
python hw2/task2/make_report.py hw2/task2/runs/<run-directory>
```

The only initial request is "Focus the beam ...". The model chooses all experiment
parameters through native tool calls. Up to eight model routing decisions are allowed.
A finish decision requires three completed experiments. The tool stores exact settings,
measured results, raw camera arrays and camera figures. `transcript.json` records native
calls and their results; `experiments.csv` is the report table. Model claims do not replace
measurements. `summary.json` keeps the best measured experiment.

The DMD is 16x16, the camera 32x32. Fixed Gaussian complex T uses seed 2026.
Input amplitudes are 0 or 1 (constant laser amplitude, no phase modulation). Larger
macropixels sum physical T columns exactly. Shot noise and read noise are added after
squaring the field magnitude. The search sees only noisy camera images. Regional
background means are averaged equally. Final enhancement uses 20 independent exposures.
The seed controls the random initial pattern, proposals and camera noise. Changing
macro size changes how random values are consumed, so seeds do not imply identical starts.

## Recording and submission

Screen-record a **fresh actual agent run**, showing the terminal's `[orchestrator]`,
`[scientist]`, `[tool]` and `[measurement]` messages. On Ubuntu use the desktop recording
control (Print Screen, then video mode), select the terminal and start recording before
the command above. Stop after the experiments finish; copy the recording into this folder.
No replay or generated animation is represented as a screen recording.

Alternatively, on an X11/XWayland desktop with tkinter and GStreamer installed:
```bash
python hw2/task2/record_run.py --output hw2/task2/runs/my-recorded-run
```
This opens a dedicated window, runs the real agent and records only that window
to `runs/my-recorded-run.webm`. Its text log is retained alongside the video.

The report compares the simulation with Figure 1 and the method in the supplied
[lab resource](https://photonicsai.com/wp-content/uploads/2026/08/Lab_work_2_focusing.pdf).
That resource uses a phase-only SLM, whereas the homework uses binary amplitude control.
Its results table is empty, so no numerical lecture enhancement can be inferred.
If a measured lecture enhancement becomes available, use
`make_report.py ... --lecture-enhancement VALUE` with a matching metric.
Combine the Task 2 report with the Task 1 report into the single requested homework PDF.
Push the source, run artifacts and recording to GitHub, and check the report's branch link.

Verification:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-task2 python -m pytest hw2/task2/tests hw2/agent/tests -q
```
