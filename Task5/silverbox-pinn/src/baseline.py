import numpy as np
from data import DT, N, N_TRAIN, load_excerpt


def finite_difference_derivatives(y, dt=DT):
    """Return velocity and acceleration estimated from measured position."""
    v = np.gradient(y, dt)
    acc = np.gradient(v, dt)
    return v, acc


def features(sel, u, y, v):
    """Matrix with one row per sample in `sel` and 4 columns: -v, -y, -y**3, u."""    
    col1 = -v[sel]
    col2 = -y[sel]
    col3 = -y[sel]**3
    col4 = u[sel]
    return np.column_stack((col1, col2, col3, col4))


def fit(X, target):
    """Fit the linear model parameters to a feature matrix and target."""
    theta, *_ = np.linalg.lstsq(X, target, rcond=None)
    return theta


def fit_baseline():
    """Fit and return the baseline parameters ``a, b, c, g``."""
    u, y = load_excerpt()
    v, acc = finite_difference_derivatives(y)

    train = np.arange(2, min(N_TRAIN, N - 2))
    X_train = features(train, u, y, v)
    return fit(X_train, acc[train])


def explained_variance(X, target, theta):
    """Fraction of the variance of `target` that the model X @ theta explains."""
    residual = target - X @ theta
    return 1 - np.var(residual) / np.var(target)


def main():
    u, y = load_excerpt()

    # velocity and acceleration from finite differences
    v, acc = finite_difference_derivatives(y)

    # drop the 2 edge samples on each side (np.gradient is less accurate there)
    idx = np.arange(2, N - 2)
    train = idx[idx < N_TRAIN]       # samples used for the fit
    test = idx[idx >= N_TRAIN]       # samples NOT used for the fit

    X_train, X_test = features(train, u, y, v), features(test, u, y, v)
    theta = fit(X_train, acc[train])
    a, b, c, g = theta

    print(f"a = {a:.2f} 1/s")
    print(f"b = {b:.3e} 1/s^2   (sqrt(b)/2pi = {np.sqrt(abs(b)) / (2 * np.pi):.1f} Hz)")
    print(f"c = {c:.3e}")
    print(f"g = {g:.3e}")
    print("condition number of X_train:", np.linalg.cond(X_train))
    print("explained variance, train:", explained_variance(X_train, acc[train], theta))
    print("explained variance, test: ", explained_variance(X_test, acc[test], theta))


if __name__ == "__main__":
    main()