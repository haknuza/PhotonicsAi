import argparse
from pathlib import Path

import torch
from torch import Tensor, nn

from baseline import finite_difference_derivatives
from data import DT, N_TRAIN, load_excerpt
from rk4 import rollout


ACCELERATION_SCALE = 1e4


class AccelerationModel(nn.Module):
    """Small scaled MLP that predicts acceleration from ``y``, ``v``, and ``u``."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, y: Tensor, v: Tensor, u_now: Tensor) -> Tensor:
        inputs = torch.stack((y / 0.2, v / 20.0, u_now / 0.2), dim=-1)
        return ACCELERATION_SCALE * self.network(inputs).squeeze(-1)


def pretrain(
    model: nn.Module,
    y_train: Tensor,
    v_train: Tensor,
    u_train: Tensor,
    acceleration_train: Tensor,
    *,
    steps: int = 2000,
    lr: float = 1e-3,
    log_every: int = 200,
) -> tuple[list[float], float]:
    """Regress predicted acceleration directly onto training-only targets."""
    tensors = (y_train, v_train, u_train, acceleration_train)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("y, v, u, and acceleration must be one-dimensional")
    if len({len(tensor) for tensor in tensors}) != 1:
        raise ValueError("y, v, u, and acceleration must have equal lengths")
    if steps < 1 or lr <= 0 or log_every < 1:
        raise ValueError("steps, lr, and log_every must be positive")

    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("model must have trainable parameters")
    y_train, v_train, u_train, acceleration_train = (
        torch.as_tensor(tensor, dtype=parameter.dtype, device=parameter.device)
        for tensor in tensors
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []

    model.train()
    for step in range(steps):
        predicted_acceleration = model(y_train, v_train, u_train)
        loss = torch.mean(
            (
                predicted_acceleration / ACCELERATION_SCALE
                - acceleration_train / ACCELERATION_SCALE
            ) ** 2
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = loss.detach().item()
        history.append(loss_value)
        if step % log_every == 0 or step + 1 == steps:
            print(f"pretrain step {step}: scaled acceleration MSE = {loss_value:.6g}")

    model.eval()
    with torch.no_grad():
        predicted_acceleration = model(y_train, v_train, u_train)
        residual_variance = torch.var(
            predicted_acceleration - acceleration_train,
            unbiased=False,
        )
        target_variance = torch.var(acceleration_train, unbiased=False)
        explained_variance = 1.0 - residual_variance / target_variance
    model.train()
    print(f"supervised acceleration explained variance: {explained_variance.item():.2%}")
    return history, explained_variance.item()


def train_model(
    model: nn.Module,
    u_train: Tensor,
    y_train: Tensor,
    dt: float,
    *,
    epochs: int = 100,
    windows_per_batch: int = 16,
    min_window: int = 64,
    max_window: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
    log_every: int = 10,
) -> list[float]:
    """Train ``model`` on random short free-run windows from training data.

    The model is called as ``model(y, v, u_now)`` and must return acceleration.
    Pass only the training portion of the record as ``u_train`` and ``y_train``.
    """
    if u_train.ndim != 1 or y_train.ndim != 1:
        raise ValueError("u_train and y_train must be one-dimensional")
    if len(u_train) != len(y_train):
        raise ValueError("u_train and y_train must have equal lengths")
    if min_window < 2 or max_window < min_window:
        raise ValueError("window lengths must satisfy 2 <= min_window <= max_window")
    if len(u_train) < max_window:
        raise ValueError("training data must contain at least max_window samples")
    if epochs < 1 or windows_per_batch < 1:
        raise ValueError("epochs and windows_per_batch must be positive")
    if log_every < 1:
        raise ValueError("log_every must be positive")

    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("model must have trainable parameters")
    u_train = torch.as_tensor(u_train, dtype=parameter.dtype, device=parameter.device)
    y_train = torch.as_tensor(y_train, dtype=parameter.dtype, device=parameter.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    random_generator = torch.Generator(device="cpu").manual_seed(seed)
    history = []

    model.train()
    for step in range(epochs):
        starts = torch.randint(
            0,
            len(u_train) - max_window + 1,
            (windows_per_batch,),
            generator=random_generator,
        ).tolist()
        lengths = torch.randint(
            min_window,
            max_window + 1,
            (windows_per_batch,),
            generator=random_generator,
        ).tolist()
        window_losses = []

        for start, length in zip(starts, lengths):
            stop = start + length
            u_window = u_train[start:stop]
            y_window = y_train[start:stop]
            initial_velocity = (y_train[start + 1] - y_train[start]) / dt
            state0 = torch.stack((y_train[start], initial_velocity))

            def state_derivative(state: Tensor, u_now: Tensor) -> Tensor:
                position, velocity = state
                acceleration = model(position, velocity, u_now)
                return torch.stack((velocity, acceleration))

            y_predicted = rollout(state_derivative, state0, u_window, dt)
            window_losses.append(torch.mean((y_predicted - y_window) ** 2))

        batch_loss = torch.stack(window_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        batch_loss.backward()
        optimizer.step()
        loss_value = batch_loss.detach().item()
        history.append(loss_value)
        if (step + 1) % log_every == 0 or step + 1 == epochs:
            print(f"rollout step {step + 1}/{epochs}: window MSE = {loss_value:.6g}")

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Silverbox acceleration model.")
    parser.add_argument("--pretrain-steps", type=int, default=2000)
    parser.add_argument("--pretrain-learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--steps",
        "--rollout-steps",
        "--epochs",
        "--rollout-epochs",
        dest="rollout_steps",
        type=int,
        default=2000,
        help="number of RK4 window optimizer updates after supervised pretraining",
    )
    parser.add_argument("--windows-per-batch", type=int, default=8)
    parser.add_argument("--min-window", type=int, default=16)
    parser.add_argument("--max-window", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "acceleration_model.pt",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AccelerationModel().to(device=device, dtype=torch.float64)
    u, y = load_excerpt()
    u_train_np = u[:N_TRAIN]
    y_train_np = y[:N_TRAIN]
    velocity_train_np, acceleration_train_np = finite_difference_derivatives(
        y_train_np,
        DT,
    )
    valid_train = slice(2, -2)
    u_supervised = torch.as_tensor(u_train_np[valid_train])
    y_supervised = torch.as_tensor(y_train_np[valid_train])
    v_supervised = torch.as_tensor(velocity_train_np[valid_train])
    acceleration_supervised = torch.as_tensor(acceleration_train_np[valid_train])
    pretrain_history, supervised_explained_variance = pretrain(
        model,
        y_supervised,
        v_supervised,
        u_supervised,
        acceleration_supervised,
        steps=args.pretrain_steps,
        lr=args.pretrain_learning_rate,
    )
    rollout_history = []
    if args.rollout_steps:
        rollout_history = train_model(
            model,
            torch.as_tensor(u_train_np),
            torch.as_tensor(y_train_np),
            DT,
            epochs=args.rollout_steps,
            windows_per_batch=args.windows_per_batch,
            min_window=args.min_window,
            max_window=args.max_window,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_width": 64,
            "input_scales": {"y": 0.2, "v": 20.0, "u": 0.2},
            "output_scale": 1e4,
            "dt": DT,
            "pretrain_steps": args.pretrain_steps,
            "pretrain_learning_rate": args.pretrain_learning_rate,
            "pretrain_final_scaled_mse": pretrain_history[-1],
            "supervised_explained_variance": supervised_explained_variance,
            "rollout_steps": args.rollout_steps,
            "windows_per_batch": args.windows_per_batch,
            "min_window": args.min_window,
            "max_window": args.max_window,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "final_window_mse": rollout_history[-1] if rollout_history else None,
        },
        args.output,
    )
    print(f"Saved model checkpoint to {args.output}")


if __name__ == "__main__":
    main()
