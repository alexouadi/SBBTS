import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import seaborn as sns
from scipy.optimize import minimize

@nb.jit(nopython=True, cache=True)
def MLE_OU_robust(params, X, dt):
    """MLE OU robust.

    Args:
        params: Parameter vector optimized by maximum-likelihood estimation.
        X: Input time-series samples.
        dt: Time discretization step.

    Returns:
        Computed output(s) produced by the function.
    """
    theta, mu, sigma = params
    N = len(X)
    logL = 0

    exp_neg_theta_dt = np.exp(-theta * dt)
    one_minus_exp_neg_theta_dt = 1 - exp_neg_theta_dt
    sigma_eta2 = (sigma ** 2 / (2 * theta)) * (1 - np.exp(-2 * theta * dt))

    for t in range(N - 1):
        mu_t = X[t] * exp_neg_theta_dt + mu * one_minus_exp_neg_theta_dt
        residual = X[t + 1] - mu_t
        logL += -0.5 * np.log(2 * np.pi * sigma_eta2) - (residual ** 2) / (2 * sigma_eta2)

    return -logL

def plot_params_distrib_OU(X_data, X_sbts, dt=1 / 252, fix=False):
    """Plot params distrib OU.

    Args:
        X_data: Collection of real trajectories used for parameter estimation.
        X_sbts: Collection of SBTS synthetic trajectories used for comparison.
        dt: Time discretization step.
        fix: If True, compare against fixed reference parameters instead of sampled priors.

    Returns:
        None.
    """
    params_data = np.zeros((len(X_data), 3))
    for m in range(len(X_data)):
        params_init_data = [1, np.mean(X_data[m]), np.std(X_data[m])]
        result_data = minimize(MLE_OU_robust, np.array(params_init_data), args=(X_data[m], dt),
                               bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                               method='L-BFGS-B')
        params_data[m] = result_data.x

    params_sbts = np.zeros((len(X_sbts), 3))
    for m in range(len(X_sbts)):
        params_init_sbts = [1, np.mean(X_sbts[m]), np.std(X_sbts[m])]
        result_sbts = minimize(MLE_OU_robust, np.array(params_init_sbts), args=(X_sbts[m], dt),
                               bounds=[(1e-5, np.inf), (-np.inf, np.inf), (1e-5, np.inf)],
                               method='L-BFGS-B')
        params_sbts[m] = result_sbts.x

    theta = np.random.uniform(.5, 2.5, 100000)
    mu = np.random.uniform(.5, 1.5, 100000)
    sigma = np.random.uniform(.1, .5, 100000)
    lines = [1.5, 1., 0.1]

    lower_bounds = np.percentile(params_data, 5, axis=0)
    upper_bounds = np.percentile(params_data, 95, axis=0)
    filtered_params_data = params_data[
        (params_data >= lower_bounds).all(axis=1) & (params_data <= upper_bounds).all(axis=1)]

    lower_bounds = np.percentile(params_sbts, 5, axis=0)
    upper_bounds = np.percentile(params_sbts, 95, axis=0)
    filtered_params_sbts = params_sbts[
        (params_sbts >= lower_bounds).all(axis=1) & (params_sbts <= upper_bounds).all(axis=1)]

    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    sns.kdeplot(ax=axs[0], data=params_data[:, 0], shade=True, label='Data')
    sns.kdeplot(ax=axs[0], data=params_sbts[:, 0], shade=True, label='SBTS')
    if not fix:
        sns.kdeplot(ax=axs[0], data=theta, shade=True, label='Real')
    else:
        line_obj = axs[0].axvline(x=lines[0], color='black', linestyle='--',
                                  label='Real')
        axs[0].legend(handles=[line_obj])
    axs[0].set_title(r'Distribution of params $\theta$')
    axs[0].legend()

    sns.kdeplot(ax=axs[1], data=filtered_params_data[:, 1], shade=True, label='Data')
    sns.kdeplot(ax=axs[1], data=filtered_params_sbts[:, 1], shade=True, label='SBTS')
    if not fix:
        sns.kdeplot(ax=axs[1], data=mu, shade=True, label='Real')
    else:
        line_obj = axs[1].axvline(x=lines[1], color='black', linestyle='--',
                                  label='Real')
        axs[1].legend(handles=[line_obj])
    axs[1].set_title(r'Distribution of params $\mu$')
    axs[1].legend()

    sns.kdeplot(ax=axs[2], data=params_data[:, 2], shade=True, label='Data')
    sns.kdeplot(ax=axs[2], data=params_sbts[:, 2], shade=True, label='SBTS')
    if not fix:
        sns.kdeplot(ax=axs[2], data=sigma, shade=True, label='Real')
    else:
        line_obj = axs[2].axvline(x=lines[2], color='black', linestyle='--',
                                  label='Real')
        axs[2].legend(handles=[line_obj])
    axs[2].set_title(r'Distribution of params $\sigma$')
    axs[2].legend()

    fig.tight_layout()
    plt.show()

@nb.jit(nopython=True, cache=True)
def MLE_Heston_robust(params, X, dt):
    """MLE Heston robust.

    Args:
        params: Parameter vector optimized by maximum-likelihood estimation.
        X: Input time-series samples.
        dt: Time discretization step.

    Returns:
        Computed output(s) produced by the function.
    """
    kappa, theta, xi, rho, r = params
    N = len(X)
    logL = 0.0

    S = X[:, 0]
    v = X[:, 1]
    for t in range(N - 1):
        S_t, S_t_next = S[t], S[t + 1]
        v_t = v[t]

        mu_S = np.log(S_t) + (r - 0.5 * v_t) * dt
        mu_v = v_t + kappa * (theta - v_t) * dt

        var_S = v_t * dt
        var_v = xi ** 2 * v_t * dt

        cov_Sv = rho * xi * v_t * dt
        cov_matrix = np.array([[var_S, cov_Sv], [cov_Sv, var_v]])

        if np.linalg.det(cov_matrix) <= 0:
            return 1e10

        inv_cov = np.linalg.inv(cov_matrix)
        det_cov = np.linalg.det(cov_matrix)

        joint_observation = np.array([
            np.log(S_t_next) - mu_S,
            v[t + 1] - mu_v
        ])
        joint_log_pdf = -0.5 * (
                2 * np.log(2 * np.pi) + np.log(det_cov) + joint_observation.T @ inv_cov @ joint_observation
        )
        logL -= joint_log_pdf

    return logL


def get_params_estimation(X, dt=1 / 252):
    """Get params estimation.

    Args:
        X: Input time-series samples.
        dt: Time discretization step.

    Returns:
        Computed output(s) produced by the function.
    """
    params_data = np.zeros((len(X), 5))

    bounds = [
        (1e-6, None),  # kappa > 0
        (1e-6, None),  # theta > 0
        (1e-6, None),  # xi > 0
        (-1, 1),  # rho in [-1, 1]
        (0, 1),  # r [0, 1]
    ]

    for m in range(len(X)):
        result_data = minimize(
            # mle_heston_vec,
            MLE_Heston_robust,
            x0=np.array([2.0, 0.02, 0.1, -0.5, 0.03]),
            args=(X[m], dt),
            bounds=bounds,
            method='L-BFGS-B'
        )
        params_data[m] = result_data.x
    return params_data

def plot_params_distrib_Heston(params_data, params_sbts, params_sbbts=None, q1=5, q2=95, fix=False, robust=False):
    """Plot params distrib Heston.

    Args:
        params_data: Estimated parameters from real data.
        params_sbts: Estimated parameters from SBTS samples.
        params_sbbts: Estimated parameters from SBBTS samples (if provided).
        q1: Lower percentile used to trim outliers.
        q2: Upper percentile used to trim outliers.
        fix: If True, compare against fixed reference parameters instead of sampled priors.
        robust: Whether to use robust clipping in the visualization.

    Returns:
        None.
    """
    kappa = np.random.uniform(0.5, 4., 100000)
    theta = np.random.uniform(0.5, 1.5, 100000)
    xi = np.random.uniform(0.1, 0.9, 100000)
    rho = np.random.uniform(-0.9, 0.9, 100000)
    r = np.random.uniform(0.01, 0.1, 100000)

    params_real = [kappa, theta, xi, rho, r]
    labels = [r'$\kappa$', r'$\theta$', r'$\xi$', r'$\rho$', r'$r$']
    lines = [3., 1., .7, .7, .02]

    lower_bounds = np.percentile(params_data, q1, axis=0)
    upper_bounds = np.percentile(params_data, q2, axis=0)
    filtered_params_data = params_data[
        (params_data >= lower_bounds).all(axis=1) & (params_data <= upper_bounds).all(axis=1)]

    lower_bounds_sbts = np.percentile(params_sbts, q1, axis=0)
    upper_bounds_sbts = np.percentile(params_sbts, q2, axis=0)
    filtered_params_sbts = params_sbts[
        (params_sbts >= lower_bounds_sbts).all(axis=1) & (params_sbts <= upper_bounds_sbts).all(axis=1)]

    lower_bounds_sbbts = np.percentile(params_sbbts, q1, axis=0)
    upper_bounds_sbbts = np.percentile(params_sbbts, q2, axis=0)
    filtered_params_sbbts = params_sbbts[
        (params_sbbts >= lower_bounds_sbbts).all(axis=1) & (params_sbbts <= upper_bounds_sbbts).all(axis=1)]

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    for i, (param, label, line) in enumerate(zip(params_real, labels, lines)):
        sns.kdeplot(ax=axs[i // 3, i % 3], data=filtered_params_data[:, i], shade=True, label='Data')
        sns.kdeplot(ax=axs[i // 3, i % 3], data=filtered_params_sbts[:, i], shade=True, label='SBTS')
        if params_sbbts is not None:
            sns.kdeplot(ax=axs[i // 3, i % 3], data=filtered_params_sbbts[:, i], shade=True, label='SBBTS')
        if robust:
            sns.kdeplot(ax=axs[i // 3, i % 3], data=param, shade=True, label='Real')
        if fix:
            line_obj = axs[i // 3, i % 3].axvline(x=line, color='black', linestyle='--',
                                                  label='Real')
            axs[i // 3, i % 3].legend(handles=[line_obj])
        axs[i // 3, i % 3].set_title(f'Distribution of param {label}')
        axs[i // 3, i % 3].legend()
        axs[i // 3, i % 3].grid()

    axs[1, 2].axis('off')
    fig.tight_layout()

    plt.show()
