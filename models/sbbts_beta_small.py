import numpy as np
import datetime
import time
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF

def kernel(x,h):
    """
    Kernel function used for kernel regression.
    :params x:; [float]
    :params h: kernel bandwidth; [float]
    return: kernel function of shape (len(x),); [np.array]
    """
    return np.where(np.abs(x) < h, (1 - (x / h) ** 2) ** 2, 0)


def first_derivate(f, y, h=1e-6):
    return (f(y + h) - f(y - h)) / (2 * h)


def second_derivate(f, y, h=1e-6):
    return (f(y + h) - 2 * f(y) + f(y - h)) / (h ** 2)


def simulate_sbbts_sb(N, M, X, N_pi, deltati, grid, K, beta, eps=1e-6):
    """
    Simulate 1 univariate time series via the SBBTS kernel.
    :params N: number of time steps to generate, must be equal to (X.shape[1] - 1); [int]
    :params M: number of samples; [int]
    :params X: sample sof shape (M, N+1); [np.array]
    :params N_pi: number of time steps in Euler scheme; [int]
    :params h: kernel bandwidth, must be > 0; [float]
    :params deltati: time step between two consecutive observations in the time series; [float]
    :params grid: uniform spatial grid; [np.array]
    :params K: number of iteration to compute the potential phi*; [int]
    :params beta: cost parameter to control the volatility, must be > 0; [float]
    :params eps: threshold to stop the convergence if |phi^{k+1} - phi^k| < eps, must be > 0; [float]
    return: 1 time series of shape (N+1,); [np.array]
    """
    # Diffusion calendar
    time_step_Euler = deltati / N_pi
    v_time_step_Euler = np.arange(0, deltati + 1e-9, time_step_Euler)

    # Generate Brownian increments
    num_brownian = N * (len(v_time_step_Euler) - 1)
    Brownian = np.random.normal(0, 1, num_brownian)

    # Simulation initialization
    X_ = X[0, 0]
    timeSeries = np.zeros(N + 1)
    timeSeries[0] = X_
    weights = np.ones(M)
    index_ = 0

    n = len(grid)
    grid_vect = grid[np.newaxis, :]
    dx = grid[1] - grid[0]

    grid_expanded = grid[:, np.newaxis] - grid[np.newaxis, :]
    gaussian_kernel = np.exp(-(grid_expanded ** 2) / (2 * deltati))
    gaussian_kernel /= np.sum(gaussian_kernel, axis=0) * dx
    
    delta_t = (deltati - v_time_step_Euler)[:-1]
    gaussian_kernel_star_ = np.exp(-(grid_expanded[:, :, np.newaxis] ** 2) / (2 * deltat[np.newaxis,: ]))
    gaussian_kernel_star_ /= np.sum(gaussian_kernel_star_, axis=0) * dx

    for i in range(N):

        # Initialization
        y_0 = X_
        h_T = np.ones(n)

        # Iterate until convergence towards phi*
        for k in range(K):
            # Compute h_T * nu_T
            h_nu_T = h_T * np.exp(-(grid - y_0) ** 2 / (2 * deltati))

            # Compute the CDF and X_T
            F_h_nu_T = np.cumsum(h_nu_T) * dx
            F_sample = ECDF(X[:, i + 1])
            sort_samples = np.sort(np.unique(X[:, i + 1]))
            F_sample_inv = interp1d(F_sample(sort_samples), sort_samples, fill_value='extrapolate')
            X_T = F_sample_inv(F_h_nu_T)

            # Update h_T and h_0
            h_T_new = np.exp(beta * np.cumsum(X_T - grid) * dx)
            h_0 = gaussian_kernel @ h_T_new * dx            

            # Update y_0
            y_0_index = np.argmin(np.log(h_0) + 0.5 * beta * (X_ - grid) ** 2)
            y_0 = grid[y_0_index]
            
            if np.max(np.abs(h_T - h_T_new)) < eps:  # convergence is reached
                h_T = h_T_new
                break
                
            h_T = h_T_new

        for k in range(len(v_time_step_Euler) - 1):
            timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]

            # Compute h*
            gaussian_kernel_star = gaussian_kernel_star_[:, :, k]
            h_star = gaussian_kernel_star @ h_T * dx

            # Compute msY* at time t via grid search
            msY_star_index = np.argmin(np.log(h_star) + 0.5 * beta * (X_ - grid) ** 2)
            msY_star = grid[msY_star_index]

            # Compute the drift and volatility
            log_h_star = interp1d(grid, no.log(h_star), fill_value='extrapolate')
            drift = first_derivate(log_h_star, msY_star)
            vol = 1 + 1 / beta *  second_derivate(log_h_star, msY_star)

            X_ += drift * timestep + Brownian[index_] * np.sqrt(timestep) * np.abs(vol)
            index_ += 1
                
        timeSeries[i + 1] = X_

    return timeSeries


def simusbbts_sb(N, M, X, N_pi, deltati, grid, K, beta, M_simu, eps=1e-6):
    """
    Simulate M_simu univariate time series via the SBBTS kernel.
    :params N: number of time steps to generate, must be equal to (X.shape[1] - 1); [int]
    :params M: number of samples; [int]
    :params X: sample sof shape (M, N+1); [np.array]
    :params N_pi: number of time steps in Euler scheme; [int]
    :params h: kernel bandwidth, must be > 0; [float]
    :params deltati: time step between two consecutive observations in the time series; [float]
    :params grid: uniform spatial grid; [np.array]
    :params K: number of iteration to compute the potential phi*; [int]
    :params beta: cost parameter to control the volatility, must be > 0; [float]
    :params M_simu: number of time series to generate; [int]
    :params eps: threshold to stop the convergence if |phi^{k+1} - phi^k| < eps, must be > 0; [float]
    return: M_simu time series of shape (M_simu,N); [np.array]
    """
    data_sb = np.zeros((M_simu, X.shape[-1]))
    st = datetime.datetime.now()
    print(f'Start time: {st.strftime("%H:%M:%S")}', flush=True)
    time1 = time.perf_counter()
    
    for k in range(M_simu):
        data_sb[k] = simulate_sbbts_sb(N, M, X, N_pi, deltati, grid, K, beta, eps)
        if k == 0:
            mm = (time.perf_counter() - time1) * (M_simu - 1) / 60
            st += datetime.timedelta(minutes=mm)
            print(f'Expected finish time: {st.strftime("%H:%M:%S")}', flush=True)

    print(f'Finish time: {datetime.datetime.now().strftime("%H:%M:%S")}', flush=True)
    print(f'Time to generate {M_simu} samples with N_pi={N_pi}: {int(time.perf_counter() - time1)} seconds.',
          flush=True)
    return data_sb[:, 1:]
