# LLM Agent Running a Fabry-Perot Simulation

Homework for the *AI-Driven Science and Autonomous Laboratories* course (Yerevan State University).

**Task:** build an LLM system that runs a scientific workflow on a simulation.

This project connects a small open-source language model (running locally with Ollama) to a Fabry-Perot cavity simulation. The model acts as the "scientist": it chooses parameters, runs the simulation through tools, reads the results and decides the next step. An independent checker, which does not use the LLM, verifies the final answer.

## The scientific problem

A Fabry-Perot cavity is two partially reflecting mirrors facing each other. Light bounces between them. At some wavelengths the bounced waves add up (constructive interference) and the light passes through; at other wavelengths they cancel. The transmission spectrum therefore has sharp peaks (resonances).

Transmission (Airy function, lossless cavity):

```
T(lambda) = 1 / (1 + F * sin^2(delta / 2))
delta = 4 * pi * n * L * cos(theta) / lambda
F = 4R / (1 - R)^2
```

- `L` is the cavity length, `n` the refractive index inside, `theta` the angle, `R` the mirror (intensity) reflectivity.
- Resonances: `lambda_m = 2 * n * L * cos(theta) / m`, with `m` an integer.
- Free spectral range (distance between peaks): `FSR ~ lambda^2 / (2 n L)`.
- Finesse: `FSR / FWHM`. For a lossless cavity, `finesse ~ pi * sqrt(R) / (1 - R)`.

**Goal given to the LLM:** find a cavity length `L` between 5 and 15 micrometers, with `R = 0.9`, that has a resonance within 2 nm of 1550 nm. Report `L` and the finesse.

(Correct answers for reference: `L = m * 0.775 um`, for example `L = 10.075 um` with `m = 13`.)

## Workflow

```
Goal -> LLM chooses a tool call -> Python runs the tool -> result goes back to the LLM -> repeat
                                                                                         |
                                           final answer -> independent check (PASS / FAIL)
```

The LLM does not calculate physics. It only decides which tool to call and with which arguments. All numbers come from the tools.

## Project structure

| File | Purpose |
|---|---|
| `fabry_perot.py` | Pure physics: the Airy transmission function. Plot code only runs with `python3 fabry_perot.py`. |
| `run_spectrum.py` | The three tools the LLM can call: `run_spectrum`, `find_resonances`, `measure_finesse`. Includes input validation and automatic choice of the wavelength step. |
| `check_claim.py` | Independent checker: uses the analytic formula `lambda_m = 2*L*1000/m` to test whether a given `L` has a resonance within the tolerance. |
| `agent.py` | The agent loop (Ollama + tool calling), retry loop, and JSON logging. |
| `logs/` | One JSON file per attempt (model, goal, messages, last `L`, PASS/FAIL). |

### Design decisions

- **The spectrum is stored in the program, not sent to the LLM.** `run_spectrum` saves the arrays and returns a short text summary. The model never has to read thousands of numbers.
- **The wavelength step is chosen automatically** as `FWHM / 10`, because a step larger than the peak width gives a wrong finesse.
- **Tools return error strings instead of crashing**, so the model can read the error and correct itself. Limits on the number of points protect memory.
- **Only one tool call per model reply is executed.** The prompt asks for this, but the 3B model ignored the request, so it is enforced in code.
- **The checker is independent of the LLM and of the simulation** (it uses the formula), so it can catch confident but wrong answers.

## Setup

Requirements: Python 3, [Ollama](https://ollama.com).

```bash
pip install numpy scipy matplotlib ollama
ollama pull qwen2.5:3b
```

The model used here is `qwen2.5:3b` (chosen because the development machine has 7.6 GB RAM and no dedicated GPU). Change `MODEL` in `agent.py` to try another model.

## Usage

Test the physics and tools by hand (no LLM):

```bash
python3 fabry_perot.py     # plot of the spectrum
python3 run_spectrum.py    # runs the three tools on a test case
```

Run the agent (with up to `MAX_ATTEMPTS` retries, each retry gets feedback from the checker):

```bash
python3 agent.py
```

Output: one line per tool call, the model's final report, and a line such as

```
CHECK: L=14.0 um, nearest resonance 1555.56 nm, distance 5.56 nm -> FAIL
```

## Validation

The tools were checked against theory before connecting the LLM.

Test case: `L = 10 um`, `R = 0.9`, range 1400 to 1700 nm.

| Quantity | Theory | Measured |
|---|---|---|
| Resonances (nm) | 1428.57, 1538.46, 1666.67 | 1428.62, 1538.65, 1666.82 |
| FWHM | about 4.03 nm | 4.04 nm |
| Finesse | 29.8 (at the central wavelength) | 29.5 |

The measured finesse is about 1% lower because the FSR is not constant: it grows with `lambda^2`, so the mean FSR over a wide range is an approximation.

Known limitation: the finesse for a peak very close to the edge of the wavelength range may be inaccurate (one run gave a finesse about 12% above theory for a peak 3 nm from the edge). This has not been fully tested yet.

## Results so far

With `qwen2.5:3b` the system works as a pipeline, but the model does not reach the goal:

- The tool calls are correct after improving the tool descriptions.
- The model recovers from error messages (for example, by widening the wavelength range).
- The model satisfies a sub-goal (measuring a finesse) but loses the main goal: it tries lengths such as 10, 12 and 14 um, which are far from the resonance condition.
- In one run the final report claimed `L = 10 um` with a finesse that had been measured on a different cavity (`L = 14 um`). The independent checker reported FAIL.
- Retries with feedback did not fix this in the attempts made.

Probable reasons: the tolerance is narrow (`L` must be correct within about 0.013 um), the feedback has no direction, and small models are unreliable at the arithmetic `L = m * 1550 / 2000`.

| Version | Change | Passes / runs |
|---|---|---|
| A | baseline, `qwen2.5:3b` | TODO |
| B | add helper tool `length_for_resonance(target_nm, m)` | TODO |
| C | baseline with `qwen2.5:7b` | TODO |

Report separately the pass rate on the first attempt (model alone) and after retries (model plus checker feedback).

## Limitations

- Lossless cavity, normal incidence by default, no material dispersion.
- Results of a small local model are not deterministic even with `temperature = 0`; one run is not enough evidence.
- The checker assumes `n = 1` (vacuum).

## Next steps

- Run each version several times and fill in the table above.
- Test the edge effect on the finesse measurement.
- Try a larger model and a helper tool for the arithmetic, and say clearly that the helper tool makes the task easier.