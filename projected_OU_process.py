import numpy as np

def ornstein_uhlenbeck(T=1, dt=1e-3, fc=1., mean=1., std=1., x0=None, nonnegative=True, rng=None):
    """
    Simulate a single path Ornstein-Uhlenbeck process.
        dX(t) = theta * (mu - X(t)) * dt + sigma * dW(t)
    Power spectrum:
        S(f) = sigma**2 / (theta**2 + (2 * pi * f)**2)
    Variance:
        Var(X) = sigma**2 / (2 * theta)

    Parameters
    ----------
    T : float
        Total time (sec)
    dt : float
        Time step size (sec)
    fc : float
        Cutoff frequency (Hz) where power spectrum drops to half its maximum value
        theta = 2 * pi * fc
    mean : float
        Long-term mean mu.
    std : float
        Standard deviation of OU process
        sigma = std * sqrt(2 * theta)
    x0 : float
        Initial value. If not provided, defaults to long-term mean mu
    nonnegative : bool
        Whether to enforce non-negativity
    rng : Generator or int
        Random number generator or seed

    Returns
    -------
    t : array of time points (sec)
    X : array of OU process values
    """
    nonnegative = bool(nonnegative)

    theta = 2 * np.pi * fc
    sigma = std * np.sqrt(2 * theta)
    theta_dt = theta * dt
    sigma_sqrt_dt = sigma * np.sqrt(dt)

    n = max(int(T / dt), 0) + 1
    t = np.arange(n) * dt
    X = np.empty(n)
    X[0] = mean if x0 is None else x0

    rng = np.random.default_rng(rng)
    xi = rng.normal(0, 1, size=n - 1)
    for i in range(n - 1):
        dX = theta_dt * (mean - X[i]) + sigma_sqrt_dt * xi[i]
        x = X[i] + dX
        X[i + 1] = max(x, 0) if nonnegative else x
    return t, X

def proj_to_simplex(v, z=1):
    """Projects vector v onto the simplex {x: sum(x) = z, x >= 0}"""
    n = len(v)
    v_sorted = np.sort(v)[::-1]
    cssv = np.cumsum(v_sorted)
    for i in range(n, 0, -1):
        rho = i - 1
        if v_sorted[rho] + (z - cssv[rho]) / i > 0:
            break
    theta = (z - cssv[rho]) / (rho + 1)
    w = np.fmax(v + theta, 0)
    return w


def projected_ou_magnitude(T, dt=1e-3, fc=1., mean=[1.], std=1., x0=None, nonnegative=True, rng=None):
    """
    Simulate N coupled Ornstein-Uhlenbeck processes with constant sum and non-negativity constraints
        dX_i(t) = theta * (mu - X_i(t)) * dt + sigma_i * dW_i(t) + mu_i / C * lambda(t) * dt + dL_i(t)
    where
        lambda(t)dt = - sum(theta * (mu_j - X_i(t)))dt - sum(sigma_i * dW_i(t))
    is the drift correction term to enforce sum constraint:
        sum(X_i) = C a constant at all times
    The term dL_i(t) enforces non-negativity by projection onto the simplex {X: sum(X_i) = C, X_i >= 0}
    where L_i(t) is a non-decreasing process that only increases when X_i(t)=0.

    Parameters
    ----------
    T : float
        Total time (sec)
    dt : float
        Time step size (sec)
    fc : float
        Cutoff frequency (Hz) where power spectrum drops to half its maximum value
        theta = 2 * pi * fc
    mean : array of float
        Long-term mean of each variable. Must be non-negative mu_i >= 0
        Number of variables is determined by the length of this array
        Sum of the variables = C = sum(mu_i)
    std : float
        Sum of standard deviation of each variable
        std_i = mu_i / C * std  (hence std = sum(std_i))
        sigma_i = std_i * sqrt(2 * theta)
    x0 : array of float
        Initial value of each variable. If not provided, defaults to long-term mean
    nonnegative : bool
        Whether to enforce non-negativity
    rng : Generator or int
        Random number generator or seed

    Returns
    -------
    t : array of time points (sec)
    X : (steps, N) array of OU processes
    """
    nonnegative = bool(nonnegative)

    mean = np.asarray(mean)
    N = len(mean)
    C = np.sum(mean)
    if x0 is None:
        x0 = mean
    else:
        x0 = np.asarray(x0)
        if np.allclose(C, 0., atol=1e-3 * (x0.max() - x0.min())):
            x0 = x0 - np.mean(x0)  # center
        else:
            x0 = C / np.sum(x0) * x0  # normalize
    if nonnegative:
        assert all(mean >= 0) and all(x0 >= 0) and C > 0

    weights = mean / C
    theta = 2 * np.pi * fc
    sigma_i = std * np.sqrt(2 * theta) * weights
    theta_dt = theta * dt
    sigma_sqrt_dt = sigma_i * np.sqrt(dt)

    n = max(int(T / dt), 0) + 1
    t = np.arange(n) * dt
    X = np.empty((n, N))
    X[0] = x0

    rng = np.random.default_rng(rng)
    xi = rng.normal(0, 1, size=(n - 1, N))
    for i in range(n - 1):
        drift = theta_dt * (mean - X[i])
        noise = sigma_sqrt_dt * xi[i]
        correction = weights * (np.sum(drift) + np.sum(noise))
        x = X[i] + drift + noise - correction
        X[i + 1] = proj_to_simplex(x, C) if nonnegative else x + (C - np.sum(x)) / N
    return t, X
