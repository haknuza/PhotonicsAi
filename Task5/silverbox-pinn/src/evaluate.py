import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from data import DT, N_TRAIN, load_excerpt
from rk4 import rollout
from train import AccelerationModel


def rmse(predicted: np.ndarray, measured: np.ndarray) -> float:
    """Compute root mean squared error in volts."""
    return float(np.sqrt(np.mean((predicted - measured) ** 2)))


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Evaluate a trained model by free run.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "results" / "acceleration_model.pt",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=project_dir / "figures" / "model_test.png",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    checkpoint_dtype = next(iter(checkpoint["model_state_dict"].values())).dtype
    model = AccelerationModel(width=checkpoint["model_width"]).to(
        device=device,
        dtype=checkpoint_dtype,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dtype = next(model.parameters()).dtype

    u, y = load_excerpt()
    u_tensor = torch.as_tensor(u, dtype=dtype, device=device)
    y_tensor = torch.as_tensor(y, dtype=dtype, device=device)
    state0 = torch.stack(
        (
            y_tensor[0],
            (y_tensor[1] - y_tensor[0]) / DT,
        )
    )

    def state_derivative(state: Tensor, u_now: Tensor) -> Tensor:
        position, velocity = state
        acceleration = model(position, velocity, u_now)
        return torch.stack((velocity, acceleration))

    with torch.no_grad():
        predicted = rollout(state_derivative, state0, u_tensor, DT)
    y_hat = predicted.cpu().numpy()

    print("free-run RMSE, train:", rmse(y_hat[:N_TRAIN], y[:N_TRAIN]))
    print("free-run RMSE, test: ", rmse(y_hat[N_TRAIN:], y[N_TRAIN:]))

    t_test = np.arange(N_TRAIN, len(y)) * DT
    fig, ax = plt.subplots()
    ax.plot(t_test, y[N_TRAIN:], label="Measured")
    ax.plot(t_test, y_hat[N_TRAIN:], label="Predicted")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Voltage (volts)")
    ax.set_title("Measured vs. predicted output on test segment")
    ax.legend()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=150)
    print(f"Saved test plot to {args.plot}")
    plt.show()


if __name__ == "__main__":
    main()
