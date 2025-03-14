import numpy as np
import matplotlib.pyplot as plt


def plot_sample(X_data, X_sbts, x0=0):
    """
    Plot 4 random samples of both X_data and X_sbts
    """
    N = X_data.shape[-1]
    x_d, x_s = np.zeros((X_data.shape[0], N + 1)), np.zeros((X_sbts.shape[0], N + 1))
    x_d[:, 0], x_s[:, 0] = x0, x0
    x_d[:, 1:], x_s[:, 1:] = X_data, X_sbts

    plt.rcParams.update({'font.size':13})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    for j in range(4):
        j1 = np.random.randint(len(X_data))
        j2 = np.random.randint(len(X_sbts))
        ax[0].plot(X_data[j1], linewidth=1.5)
        ax[1].plot(X_sbts[j1], linewidth=1.5)

    ax[0].set_xlabel('time')
    ax[0].set_ylabel('Data')
    ax[0].tick_params(axis='both', which='major', labelsize=13)

    ax[1].set_xlabel('time')
    ax[1].set_ylabel('SBTS')
    ax[1].tick_params(axis='both', which='major', labelsize=13)
    plt.show()


