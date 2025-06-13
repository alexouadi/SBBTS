import numpy as np
import datetime
import time
from scipy.interpolate import interp1d
from collections import deque
from scipy.signal import fftconvolve

def kernel(x,h):
    """
    Kernel function used for kernel regression.
    :params x:; [float]
    :params h: kernel bandwidth; [float]
    return: kernel function of shape (len(x),); [np.array]
    """
    return np.exp(- 0.5 * (x / h) ** 2)


def first_derivate(f, y, h=1e-6):
    return (f(y + h) - f(y - h)) / (2 * h)


def simulate_sbbts_sb(N, M, X, N_pi, deltati, grid, beta, K_markov, K, eps=1e-6):
    """
    Simulate 1 univariate time series via the SBBTS kernel.
    :params N: number of time steps to generate, must be equal to (X.shape[1] - 1); [int]
    :params M: number of samples; [int]
    :params X: sample sof shape (M, N+1); [np.array]
    :params N_pi: number of time steps in Euler scheme; [int]
    :params h: kernel bandwidth, must be > 0; [float]
    :params deltati: time step between two consecutive observations in the time series; [float]
    :params grid: uniform spatial grid; [np.array]
    :params beta: cost parameter to control the volatility, must be > 0; [float]
    :params K_markov: markovian order; [int]
    :params K: number of iteration to compute the potential phi*; [int]
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

    n, M = len(grid), len(X)
    grid_vect = grid[np.newaxis, :]
    dx = grid[1] - grid[0]
    weights = np.ones(M)
    last_K = deque(maxlen=K_markov)

    gaussian_kernel = np.exp(-(grid ** 2) / (2 * deltati))
    gaussian_kernel /= np.sum(gaussian_kernel) * dx

    for i in range(N):
        if i > 0:
            if len(last_K) == K_markov:
                X_oldest = last_K[0]
                kernel_oldest = kernel(X[:, i - K_markov] - X_oldest, h)

                if np.any(kernel_oldest == 0):
                    weights = np.ones(M)
                    ind_ref = i - K_markov
                    for j in range(1, K_markov):
                        weights *= kernel(X[:, ind_ref + j] - last_K[j], h)
                else:
                    weights /= kernel_oldest
                    
            last_K.append(X_)
            weights[:] *= kernel(X[:, i] - X_, h)

        else:
            weights[:] = 1 / M

        weights_ratio = weights / np.sum(weights) if np.sum(weights) > 0 else np.zeros_like(weights)
        weights_vect = weights_ratio[:, None]
        
        # Initialization
        y_0 = X_
        h_T = np.ones(n)

        # Iterate until convergence towards phi*
        for k in range(K):
            # Compute h_T * nu_T
            h_nu_T = h_T * np.exp(-(grid - y_0) ** 2 / (2 * deltati))

            # Compute the CDF and X_T
            F_h_nu_T = np.cumsum(h_nu_T) * dx
            F_h_nu_T /= F_h_nu_T[-1]
            
            F_sample = np.mean((X[:, i + 1, None] <= grid[None, :]) * weights_vect, axis=0)
            F = F_sample / F_sample[-1] if F_sample[-1] > 0 else np.zeros(n)
            F_sample_inv = interp1d(F, grid, fill_value='extrapolate')
            X_T = F_sample_inv(F_h_nu_T)

            # Update h_T and h_0
            h_T_new = np.exp(beta * np.cumsum(X_T - grid) * dx)
            h_0 = fftconvolve(gaussian_kernel, h_T_new, mode='same')
            h_0 /= np.sum(h_0) * dx

            # Update y_0
            y_0_index = np.argmin(np.log(h_0) + 0.5 * beta * (X_ - grid) ** 2)
            y_0 = grid[y_0_index]
            
            if np.max(np.abs(h_T - h_T_new)) < eps:  # convergence is reached
                h_T = h_T_new
                break
                
            h_T = h_T_new

        for k in range(len(v_time_step_Euler) - 1):
            timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]
            delta_t = deltati - v_time_step_Euler[k]

            # Compute h*
            gaussian_kernel_star = np.exp(-grid ** 2 / (2 * delta_t))
            gaussian_kernel_star /= np.sum(gaussian_kernel_star) * dx
            
            h_t = fftconvolve(gaussian_kernel_star, h_T, mode='same')
            h_t /= np.sum(h_t) * dx
            
            # Compute msY* at time t via grid search
            msY_star_index = np.argmin(np.log(h_t) + 0.5 * beta * (X_ - grid) ** 2)
            msY_star = grid[msY_star_index]

            # Compute the drift and volatility
            log_h_t = interp1d(grid, np.log(h_t), fill_value='extrapolate')
            grad_log_h_t = first_derivate(log_h_t, grid)
            drift = grad_log_h_t[msY_star_index]

            grad_log_h_t_interp = interp1d(grid[1:-1], drift[1:-1], fill_value='extrapolate')
            vol = 1 + 1 / beta *  first_derivate(grad_log_h_t_interp, msY_star)

            X_ += drift * timestep + Brownian[index_] * np.sqrt(timestep) * np.abs(vol)
            index_ += 1
                
        timeSeries[i + 1] = X_

    return timeSeries


def simusbbts_sb(N, M, X, N_pi, deltati, grid, beta, M_simu, K_markov, K=50, eps=1e-6):
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
        data_sb[k] = simulate_sbbts_sb(N, M, X, N_pi, deltati, grid, beta, K_markov, K, eps)
        if k == 0:
            mm = (time.perf_counter() - time1) * (M_simu - 1) / 60
            st += datetime.timedelta(minutes=mm)
            print(f'Expected finish time: {st.strftime("%H:%M:%S")}', flush=True)

    print(f'Finish time: {datetime.datetime.now().strftime("%H:%M:%S")}', flush=True)
    print(f'Time to generate {M_simu} samples with N_pi={N_pi}: {int(time.perf_counter() - time1)} seconds.',
          flush=True)
    return data_sb[:, 1:]
