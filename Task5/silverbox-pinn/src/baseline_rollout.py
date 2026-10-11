from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from baseline import fit_baseline
from data import DT, N, N_TRAIN, load_excerpt
from rk4 import rollout


def rmse(pred, true):
    """Return the root mean squared error between predictions and targets."""
    return np.sqrt(np.mean((pred - true) ** 2))


def main():
    u, y = load_excerpt()
    a, b, c, g = map(float, fit_baseline())

    def f(s, u_now):
        """Time derivative of the state s = (y, v)."""
        position, velocity = s
        acceleration = -a * velocity - b * position - c * position**3 + g * u_now
        return torch.stack((velocity, acceleration))

    u_tensor = torch.as_tensor(u, dtype=torch.float64)
    s0 = torch.tensor([y[0], (y[1] - y[0]) / DT], dtype=torch.float64)
    y_hat = rollout(f, s0, u_tensor, DT).detach().cpu().numpy()

    print("free-run RMSE, train:", rmse(y_hat[:N_TRAIN], y[:N_TRAIN]))
    print("free-run RMSE, test: ", rmse(y_hat[N_TRAIN:], y[N_TRAIN:]))
    print("zero-prediction baseline on test:", rmse(np.zeros_like(y[N_TRAIN:]), y[N_TRAIN:]))

    n_start = min(int(0.2 / DT) + 1, N)
    t_start = np.arange(n_start) * DT
    print("free-run RMSE, first 0.2 s:", rmse(y_hat[:n_start], y[:n_start]))

    Path("figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(t_start, y[:n_start], label="Measured")
    ax.plot(t_start, y_hat[:n_start], label="Predicted")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Voltage (volts)")
    ax.set_title("Measured vs. predicted output, first 0.2 seconds")
    ax.legend()
    fig.savefig("figures/baseline_start.png", dpi=150)
    plt.show()

    t_test = np.arange(N_TRAIN, N) * DT
    fig, ax = plt.subplots()
    ax.plot(t_test, y[N_TRAIN:], label="Measured")
    ax.plot(t_test, y_hat[N_TRAIN:], label="Predicted")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Voltage (volts)")
    ax.set_title("Measured vs. predicted output on test segment")
    ax.legend()
    fig.savefig("figures/baseline_test.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
