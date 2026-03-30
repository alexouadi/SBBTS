import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def my_acf(my_arr, lag_len, lev=False):
    """Compute empirical autocorrelation up to a fixed lag.

    Args:
        my_arr: One-dimensional return series.
        lag_len: Maximum lag to evaluate.
        lev: If True, compute cross-correlation between squared and raw returns;
            otherwise compute standard return autocorrelation.

    Returns:
        Array of length `lag_len + 1` containing autocorrelation values.
    """
    x = my_arr
    x = x - x.mean()
    acorr = np.empty(lag_len + 1)
    if lev:
        x_squared = my_arr ** 2
        x_squared = x_squared - x_squared.mean()
        acorr[0] = x_squared.dot(x) / np.sqrt(x_squared.dot(x_squared) * x.dot(x))
        for i in range(lag_len):
            acorr[i + 1] = x_squared[i + 1:].dot(x[: -(i + 1)]) / np.sqrt(
                x_squared[i + 1:].dot(x_squared[i + 1:]) * x[: -(i + 1)].dot(x[: -(i + 1)]))
    else:
        acorr[0] = 1
        for i in range(lag_len):
            acorr[i + 1] = x[i + 1:].dot(x[: -(i + 1)]) / np.sqrt(
                x[i + 1:].dot(x[i + 1:]) * x[: -(i + 1)].dot(x[: -(i + 1)]))
    return acorr

def plot_acf(x_data, x_sbbts, path=None, figsize=(8, 2), legend='SBBTS'):
    """Plot mean ACF of returns and squared returns for real vs synthetic data.

    Args:
        x_data: Iterable of real return series.
        x_sbbts: Iterable of synthetic return series.
        path: Optional output path to save the figure.
        figsize: Matplotlib figure size.
        legend: Legend label for the synthetic curve.

    Returns:
        None.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    lag = 60

    acf_real = []
    acf_synth = []
    for s in x_data:
        acf_real.append(my_acf((s) ** 1, lag_len=lag))
    for s in x_sbbts:
        acf_synth.append(my_acf((s) ** 1, lag_len=lag))

    mean_acf = np.array(acf_real).mean(axis=0)
    mean_acf3 = np.array(acf_synth).mean(axis=0)

    ax[0].plot(mean_acf[1:], color='red', alpha=.5, label='Real')
    ax[0].plot(mean_acf3[1:], color='blue', alpha=.5, label=legend)
    ax[0].legend()
    ax[0].set_title('Autocorrelation of returns')
    ax[0].set_xlabel('Time (days)')

    acf_real = []
    acf_synth = []
    for s in x_data:
        acf_real.append(my_acf((s) ** 2, lag_len=lag))
    for s in x_sbbts:
        acf_synth.append(my_acf((s) ** 2, lag_len=lag))

    mean_acf = np.array(acf_real).mean(axis=0)
    mean_acf3 = np.array(acf_synth).mean(axis=0)

    ax[1].plot(mean_acf[1:], color='red', alpha=.5, label='Real')
    ax[1].plot(mean_acf3[1:], color='blue', alpha=.5, label=legend)
    ax[1].legend()
    ax[1].set_title('Autocorrelation of squared returns')
    ax[1].set_xlabel('Time (days)')

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()

def plot_return_dist(x_data, x_sbbts, bins=100, figsize=(8, 2), path=None):
    """Plot return distribution.

    Args:
        x_data: Real return arrays grouped by cluster.
        x_sbbts: Synthetic return arrays grouped by cluster.
        bins: Number of histogram bins.
        figsize: Matplotlib figure size.
        path: Optional output path to save the figure.

    Returns:
        None.
    """
    if len(x_data) != len(x_sbbts):
        raise ValueError('diff_real and diff_fake must have the same length')

    fig, ax = plt.subplots(1, len(x_data), figsize=figsize)

    for i in range(len(x_data)):
        len_ = min(len(x_data), len(x_sbbts))
        idx = np.random.permutation(len_)
        a = ax[i]

        a.hist(x_data[i][idx].flatten(),
               bins=bins,
               density=True,
               alpha=0.3,
               color='red',
               label='Real')

        a.hist(x_sbbts[i][idx].flatten(),
               bins=bins,
               density=True,
               alpha=0.3,
               color='blue',
               label='Synth')

        a.legend(loc='upper right')
        a.set_xlabel('Log return')
        a.set_ylabel('Density')
        a.set_title(f'Cluster {i + 1}')
        a.grid(alpha=0.3)

    if path:
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_corr_matrix(x_data, x_sbbts, annot=False, figsize=(11, 4), path=None):
    """Plot average cross-sectional correlation matrices for real and synthetic data.

    Args:
        x_data: Real data tensor shaped (n_samples, n_steps, n_assets).
        x_sbbts: Synthetic data tensor shaped (n_samples, n_steps, n_assets).
        annot: Whether to annotate each heatmap cell with its value.
        figsize: Matplotlib figure size.
        path: Optional output path to save the figure.

    Returns:
        None.
    """
    data_set1 = x_data
    data_set2 = x_sbbts

    def calculate_correlation_matrix(data):
        """Compute asset-by-asset correlation matrix for one sample.

        Args:
            data: 2D array (time, assets).

        Returns:
            Correlation matrix of shape (assets, assets).
        """
        return np.corrcoef(data, rowvar=False)

    correlation_matrices_set1 = np.array(
        [calculate_correlation_matrix(data_set1[i]) for i in range(data_set1.shape[0])])
    correlation_matrices_set2 = np.array(
        [calculate_correlation_matrix(data_set2[i]) for i in range(data_set2.shape[0])])

    corr_mat_1 = np.mean(correlation_matrices_set1, axis=0)
    corr_mat_2 = np.mean(correlation_matrices_set2, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    sns.heatmap(corr_mat_1, annot=annot, fmt=".2f", ax=axes[0])
    axes[0].set_title('Data')

    sns.heatmap(corr_mat_2, annot=annot, fmt=".2f", ax=axes[1])
    axes[1].set_title('SBBTS')
    if path:
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
