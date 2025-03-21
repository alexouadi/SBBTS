import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot_sample(X_data, X_sbts, x0=0):
    """
    Plot 4 random univariates samples.
    :params X_data: original data; [np.array]
    :params X_sbts: generated data; [np.array]
    :params x0: initial value to plot; [float]
    """
    N = X_data.shape[-1]
    x_d, x_s = np.zeros((X_data.shape[0], N + 1)), np.zeros((X_sbts.shape[0], N + 1))
    x_d[:, 0], x_s[:, 0] = x0, x0
    x_d[:, 1:], x_s[:, 1:] = X_data, X_sbts
    X_data, X_sbts = x_d, x_s

    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for i in range(4):
        j1 = np.random.randint(len(X_data))
        j2 = np.random.randint(len(X_sbts))
        ax[0].plot(X_data[j1], linewidth=1.5)
        ax[1].plot(X_sbts[j2], linewidth=1.5)

    ax[0].set_xlabel('time')
    ax[0].set_ylabel('Data')
    ax[0].tick_params(axis='both', which='major', labelsize=13)

    ax[1].set_xlabel('time')
    ax[1].set_ylabel('SBTS')
    ax[1].tick_params(axis='both', which='major', labelsize=13)
    plt.show()


def plot_sample_multi(X_data, X_sbts, col=None, x0=1):
    """
    Plot 1 random multivariates samples.
    :params X_data: original data; [np.array]
    :params X_sbts: generated data; [np.array]
    :params col: name of each features; [list]
    :params x0: initial value to plot; [float]
    """
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    B, N, d = X_sbts.shape
    x_d, x_s = np.zeros((X_data.shape[0], N + 1, d)), np.zeros((X_sbts.shape[0], N + 1, d))
    x_d[:, 0], x_s[:, 0] = x0, x0
    x_d[:, 1:], x_s[:, 1:] = X_data, X_sbts
    X_data, X_sbts = x_d, x_s

    ind_data = np.random.randint(len(X_data))
    ind_sbts = np.random.randint(len(X_sbts))

    for i in range(X_data.shape[-1]):
        if col is not None:
            ax[0].plot(X_data[:, :, i][ind_data], linewidth=1.5, label=col[i])
            ax[1].plot(X_sbts[:, :, i][ind_sbts], linewidth=1.5, label=col[i])
        else:
            ax[0].plot(X_data[:, :, i][ind_data], linewidth=1.5)
            ax[1].plot(X_sbts[:, :, i][ind_sbts], linewidth=1.5)

    ax[0].set_xlabel('time')
    ax[0].set_ylabel('Data')
    ax[0].legend()
    ax[0].tick_params(axis='both', which='major', labelsize=13)

    ax[1].set_xlabel('time')
    ax[1].set_ylabel('SBTS')
    ax[1].legend()
    ax[1].tick_params(axis='both', which='major', labelsize=13)


def get_stats(X_data, X_sbts, col=None):
    """
    Plot 1% and 99% percentiles, mean, and standard deviation for two input arrays.
    :params X_data: original data; [np.array]
    :params X_sbts: generated data; [np.array]
    :params col: name of each features; [list]
    """

    X_data, X_sbts = fix_dim(X_data), fix_dim(X_sbts)

    # calculate 1% and 99% percentiles for both arrays
    percentiles1 = np.percentile(X_data, [1, 99], axis=(0, 1))  # shape (2, D)
    lower_percentile1 = percentiles1[0, :]  # shape (D,)
    upper_percentile1 = percentiles1[1, :]  # shape (D,)

    percentiles2 = np.percentile(X_sbts, [1, 99], axis=(0, 1))  # shape (2, D)
    lower_percentile2 = percentiles2[0, :]  # shape (D,)
    upper_percentile2 = percentiles2[1, :]  # shape (D,)

    # calculate mean and standard deviation for both arrays
    mean1 = np.mean(X_data, axis=(0, 1))  # shape (D,)
    std1 = np.std(X_data, axis=(0, 1))  # shape (D,)

    mean2 = np.mean(X_sbts, axis=(0, 1))  # shape (D,)
    std2 = np.std(X_sbts, axis=(0, 1))  # shape (D,)

    min_data = X_data.min(axis=(0, 1))
    min_sbts = X_sbts.min(axis=(0, 1))

    max_data = X_data.max(axis=(0, 1))
    max_sbts = X_sbts.max(axis=(0, 1))

    if col is None:
        col = range(len(lower_percentile1))

    df = pd.DataFrame({
        'Feature': col,
        '1% Data': lower_percentile1,
        '1% SBTS': lower_percentile2,
        '99% Data': upper_percentile1,
        '99% SBTS': upper_percentile2,
        'Mean Data': mean1,
        'Mean SBTS': mean2,
        'Std Data': std1,
        'Std SBTS': std2,
        'Min Data': min_data,
        'Min SBTS': min_sbts,
        'Max Data': max_data,
        'Max SBTS': max_sbts
    })

    df.set_index('Feature', inplace=True)
    return df.round(3)
