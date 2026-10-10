import numpy as np

def rk4_step(f, s, u0, u_mid, u1, dt):
    """Advance state ``s`` by one RK4 step with sampled input values."""
    k1 = f(s, u0)
    k2 = f(s + 0.5 * dt * k1, u_mid)
    k3 = f(s + 0.5 * dt * k2, u_mid)
    k4 = f(s + dt * k3, u1)
    return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def rollout(f, s0, u, dt):
    """Simulate len(u) samples from the state s0; return the predicted y."""
    s = np.zeros((len(u), 2))
    s[0] = s0
    for k in range(len(u) - 1):
        u_mid = 0.5 * (u[k] + u[k + 1])
        s[k + 1] = rk4_step(f, s[k], u[k], u_mid, u[k + 1], dt)
    return s[:, 0]