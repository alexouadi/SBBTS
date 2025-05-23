import numpy as np
import numba as nb
import datetime
import time
from scipy.interpolate import interp1d


@nb.jit(nopython=True, cache=True)
def kernel(x,h):
    """
    Kernel function used for kernel regression.
    :params x:; [float]
    :params h: kernel bandwidth; [float]
    return: kernel function of shape (len(x),); [np.array]
    """
    return np.where(np.abs(x) < h, (h ** 2 - x ** 2) ** 2, 0)


@nb.jit(nopython=True, cache=True)
def grad_kernel(x, h, eps=1e-5):
    return (kernel(x+eps, h) - kernel(x-eps, h)) / (2 * eps)


def get_grid(X, n_points=1000, factor=2.0):
    return np.linspace(X.min() - factor * X.std(), X.max() + factor * X.std())


def simulate_sbbts_kernel(N, M, X, N_pi, h, deltati, grid, K, beta):
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
    grid_vect = grid[np.newaxis, :]

    for i in range(N):
        if i > 0:
            weights[:] *= kernel(X[:, i] - X_, h)
        else:
            weights[:] = 1 / M
            
        weights_vect = weights[:, np.newaxis]
        weights_ratio = weights / np.sum(weights) if np.sum(weights) > 0 else np.zeros_like(weights)

        # Initialization
        y_0 = X_
        msY_T = X[:, i + 1].copy()

        # Iterate until convergence towards phi*
        for k in range(K):
            # Solve y_i^k via grid search
            weights_one = np.exp((msY_T - y_0) ** 2 / (2 * deltati)) * weights_ratio
            weights_h_0 = weights_one[:, np.newaxis] * np.exp(
                (msY_T[:, np.newaxis] - grid_vect) ** 2 / (-2 * deltati))
            
            h_0 = np.mean(weights_h_0, axis=0)
            y_0_index = np.argmin(np.log(h_0) + 0.5 * beta * (X_ - grid) ** 2)
            y_0_new = grid[y_0_index]

            # Update msY_k
            grad_phi_num = np.sum(grad_kernel(msY_T[:, np.newaxis] - grid_vect, h) * weights_vect, axis=0)
            grad_phi_den = np.sum(kernel(msY_T[:, np.newaxis] - grid_vect, h) * weights_vect, axis=0)
            grad_phi = np.where(grad_phi_den != 0, grad_phi_num / grad_phi_den + (grid - y_0) / deltati, 0.0)
            
            msX = grid + 1 / beta * grad_phi
            msY_T_interpolate = interp1d(msX, grid, fill_value='extrapolate')

            # Update y_k and msY_k
            y_0 = y_0_new
            msY_T = msY_T_interpolate(X[:, i + 1])  # msY_k pushforward mu_{i+1|0:i}
            

        weights_one_star = np.exp((msY_T - y_T) ** 2 / (2 * deltati)) * weights_ratio
        coeff_exp = (msY_T[:, np.newaxis] - grid_vect) ** 2
        for k in range(len(v_time_step_Euler) - 1):
            timeprev = v_time_step_Euler[k]
            timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]
            delta_t = deltati - timeprev

            # Compute Y* at time t
            weights_h_star = weights_one_star[:, np.newaxis] * np.exp(coeff_exp / (-2 * delta_t))
            h_star_ = np.mean(weights_h_star, axis=0)
            msY_star_index = np.argmin(np.log(h_star_) + 0.5 * (X_ - grid) ** 2)

            # Compute msY* at time t via grid search
            msY_star_index = np.argmin(np.log(h_star) + 0.5 * beta * (X_ - grid) ** 2)
            msY_star = grid[msY_star_index]

            # Compute the drift and volatility
            weights_den = weights_one_star * np.exp((msY_T - msY_star) ** 2 / (-2 * delta_t))
            h_star = np.sum(weights_den)
            
            grad_h_star = 1 / delta_t * np.sum(weights_den * (msY_T - msY_star))
            hessian_h_star = np.sum(weights_den * (((msY_T - msY_star) / delta_t) ** 2 - 1 / delta_t))

            drift = grad_h_star / h_star if h_star > 0 else 0.0
            vol = 1 + 1 / beta * (hessian_h_star / h_star - (grad_h_star / h_star) ** 2) if h_star > 0 else 0.0

            X_ += drift * timestep + Brownian[index_] * np.sqrt(timestep) * np.abs(vol)
            index_ += 1
                
        timeSeries[i + 1] = X_

    return timeSeries


def simusbbts_kernel(N, M, X, N_pi, h, deltati, grid, K, beta, M_simu):
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
    return: M_simu time series of shape (M_simu,N); [np.array]
    """
    data_sb = np.zeros((M_simu, X.shape[-1]))
    st = datetime.datetime.now()
    print(f'Start time: {st.strftime("%H:%M:%S")}', flush=True)
    time1 = time.perf_counter()
    
    for k in range(M_simu):
        data_sb[k] = simulate_sbbts_kernel(N, M, X, N_pi, h, deltati, grid, K, beta)
        if k == 0:
            mm = (time.perf_counter() - time1) * (M_simu - 1) / 60
            st += datetime.timedelta(minutes=mm)
            print(f'Expected finish time: {st.strftime("%H:%M:%S")}', flush=True)

    print(f'Finish time: {datetime.datetime.now().strftime("%H:%M:%S")}', flush=True)
    print(f'Time to generate {M_simu} samples with N_pi={N_pi}: {int(time.perf_counter() - time1)} seconds.',
          flush=True)
    return data_sb[:, 1:]
