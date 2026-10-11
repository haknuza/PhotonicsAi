# Project context: Silverbox PINN (Homework 1, Task 5 and Task 6)

This file gives an AI assistant the background of the project. Read it before suggesting code.

## About me and how to help

- I am a university student (radio-physics, programming, math). I am an early-to-intermediate programmer and I am learning English, so use simple, clear wording.
- I want to understand the code  Explain the idea first.
- If I make a mistake, say so directly and explain why.
- Scientific correctness comes first, then understanding, reproducibility, efficiency, presentation.

## The task

Build a model that predicts the output voltage `y(t)` of the Silverbox circuit from its input voltage `u(t)`.
The Silverbox is an electronic Duffing oscillator (a mass on a spring that gets stiffer when stretched).

Equation of the circuit (equation (1) in the homework):

```
y'' + a*y' + b*y + c*y^3 = g*u(t)
```

- `a`: damping (multiplies velocity), a few tens of 1/s
- `b`: linear stiffness (multiplies y), in 1/s^2
- `c`: cubic stiffness (multiplies y^3), makes the spring stiffer at large amplitude
- `g`: input gain (u and y are voltages on opposite sides of the circuit)

These are circuit properties, NOT the weights of the network.

### Task 5 (PINN)

The acceleration is produced by a fully connected PyTorch network. With state `(y, v)`, `v = y'`:

```
y' = v
v' = alpha_theta(y, v, u)
```

The state is integrated with Runge-Kutta 4 (RK4) at step `DT = 1/610.35 s`, driven by the measured `u`.
I choose depth, widths, activation, input/output scaling. I must try more than one architecture and tune, for each one: optimizer, learning rate, number of steps, and the length of the rollout used for training. Keep the version with the best test RMSE.

### Task 6

A LangGraph system with 4 agents that solves Task 5 (separate task, built after Task 5 works).

## Data

- File: `data/SNLS80mV.mat` (Silverbox benchmark, Wigren and Schoukens, ECC 2013). Channel `V1` = input u, channel `V2` = output y. 131,072 samples, `fs = 610.35 Hz`.
- Excerpt: `[42650 : 46746]`, 4096 samples. Subtract the mean of u and the mean of y of THIS excerpt.
- First 3072 samples = training record (about 5.03 s). Last 1024 samples = test record (about 1.68 s).
- Checked values of the excerpt: peak |y| = 0.205 V, test RMS of y = 0.0605 V, dominant frequency of y = 70.04 Hz.
- Acceleration is of order 10^4 V/s^2, so input and output scaling of the network matters.

## The metric (important rules)

1. After the weights are fixed, do ONE free run over the whole excerpt: start at sample 0 with the measured position and a finite-difference velocity, integrate through the training samples and on into the test samples, and keep the same state across the boundary.
2. On the test segment the rollout may read `u`. It must NEVER read `y`.
3. Score = RMSE of the free run on the 1024 test samples. Also report the same RMSE on the 3072 training samples.
4. Reference: predicting zero gives about 0.06 V on the test part.

## Code structure

```
silverbox-pinn/
├── CONTEXT.md
├── data/SNLS80mV.mat
├── src/
│   ├── data.py              # load_excerpt() -> centred u, y; constants FS, DT, START, N, N_TRAIN
│   ├── explore.py           # sanity checks, spectra, acceleration plot
│   ├── baseline.py          # fit_baseline(): least-squares fit of a, b, c, g
│   ├── rk4.py               # rk4_step(f, s, u0, u_mid, u1, dt) and rollout(f, s0, u, dt)
│   ├── baseline_rollout.py  # free run with the baseline numbers
│   ├── model.py             # (planned) the network alpha_theta
│   ├── train.py             # (planned) windowed training
│   └── evaluate.py          # (planned) free run, train/test RMSE, plot
├── results/                 # one record per experiment (settings + RMSEs)
└── figures/
```

Conventions:

- Constants live only in `data.py`. All scripts import from it.
- No code runs on import. Scripts use `if __name__ == "__main__":` for prints and plots.
- RK4 code is written with plain `+` and `*` on the state, so the same code must later work on PyTorch tensors (gradients must flow through the integrator).
- The input between samples: `u_mid = 0.5 * (u[k] + u[k+1])`.
- Fix random seeds. Log every experiment (settings, train RMSE, test RMSE) in `results/`.
- Units: time in seconds, voltages in volts.

## Current status

Next:

1. Replace the known force by the network `alpha_theta(y, v, u)` in PyTorch (careful scaling of inputs and output).
2. Train on short rollout windows from many start points, then longer windows (long rollouts are hard to optimize).
3. Final free run, train and test RMSE.
4. Try several architectures and training settings, log all test RMSEs.
5. Write the PDF and push the code to GitHub.

## Deliverable (due 2026-10-11)

One PDF, emailed to the instructor, with: the state equation, the architecture kept and its hyperparameters, how it was trained, which other settings were tried and their test RMSE, both free-run RMSE numbers for the kept version, a plot of measured against predicted output on the test segment (from the free run), and the link to the GitHub repository with the code.

## Known pitfalls

- The function `f(s, u)` given to `rk4_step` must return a numpy array (or tensor), not a list or tuple.
- Differentiating `y` twice amplifies noise, so the baseline coefficients are approximate.
- Long free-run rollouts accumulate error: this is why training uses short windows first.
- Do not read `y` of the test segment inside the test rollout.