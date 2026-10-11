from collections.abc import Callable

import torch
from torch import Tensor


def rk4_step(
    f: Callable[[Tensor, Tensor], Tensor],
    s: Tensor,
    u0: Tensor,
    u_mid: Tensor,
    u1: Tensor,
    dt: float,
) -> Tensor:
    """Advance state ``s`` by one RK4 step with sampled input values."""
    k1 = f(s, u0)
    k2 = f(s + 0.5 * dt * k1, u_mid)
    k3 = f(s + 0.5 * dt * k2, u_mid)
    k4 = f(s + dt * k3, u1)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def rollout(
    f: Callable[[Tensor, Tensor], Tensor],
    s0: Tensor,
    u: Tensor,
    dt: float,
) -> Tensor:
    """Simulate len(u) samples from the state s0; return the predicted y."""
    states = [s0]
    for k in range(len(u) - 1):
        u_mid = 0.5 * (u[k] + u[k + 1])
        states.append(rk4_step(f, states[-1], u[k], u_mid, u[k + 1], dt))
    return torch.stack(states)[:, 0]