import numpy as np
import datetime
import time
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftfreq

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


def simulate_sbbts_fft(N, M, X, N_pi, h, deltati, grid, K, beta, eps=1e-6):
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
    
    gaussian_kernel = np.exp(-(grid ** 2) / (2 * deltati))
    gaussian_kernel /= np.sum(gaussian_kernel) * dx
    fft_gaussian = fft.fft(gaussian_kernel)

    for i in range(N):
        if i > 0:
            weights[:] *= kernel(X[:, i] - X_, h)
        else:
            weights[:] = 1 / M
            
        weights_ratio = weights / np.sum(weights) if np.sum(weights) > 0 else np.zeros_like(weights)

        # Initialization
        phi_values = np.zeros(n)
        phi = interp1d(grid, phi_values, fill_values='extrapolate')

        # Iterate until convergence towards phi*
        for k in range(K):
            fft_exp_phi = fft.fft(np.exp(phi_values))
            fft_h_0 = fft_gaussian * fft_exp_phi
            h_0 = np.real(fft.ifft(fft_h_0)) * dx  # h^k over all the grid
            
            y_0_index = np.argmin(np.log(h_0) + 0.5 * beta * (X_ - grid) ** 2)
            y_0 = grid[y_0_index]

            # Compute the transport map
            grad_phi = first_derivate(phi, grid)

            msX = grid + 1 / beta * grad_phi
            msY_T_interpolate = interp1d(msX, grid, fill_value='extrapolate')
            msY_T = msY_T_interpolate(X[:, i + 1])  # msY pushforward mu_{i+1}

            # Update the potential
            msY_vect = msY_T[:, np.newaxis]
            term_1 = np.log(np.mean(kernel(msY_vect - grid_vect, h) * weights_ratio[:, np.newaxis], axis=0))
            term_2 = (grid - y_0) ** 2 / (2 * deltati)
            phi_values_new = term_1 + term_2 
            
            if np.max(np.abs(phi_values - phi_values_new)) < eps:  # convergence is reached
                phi_values = phi_values_new
                break
                
            phi_values = phi_values_new
            phi = interp1d(grid, phi_values, fill_value='extrapolate')

        fft_exp_phi_star = fft.fft(np.exp(phi_values))
        for k in range(len(v_time_step_Euler) - 1):
            timeprev = v_time_step_Euler[k]
            timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]
            delta_t = deltati - timeprev

            # Compute h*
            gaussian_kernel_star = np.exp(-(grid ** 2) / (2 * delta_t ** 2))
            gaussian_kernel_star /= np.sum(gaussian_kernel_star) * dx
            fft_gaussian_star = fft.fft(gaussian_kernel_star)
            fft_h_star = fft_gaussian_star * fft_exp_phi_star
            h_star = np.real(fft.ifft(fft_h_star)) * dx

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


def simusbbts_fft(N, M, X, N_pi, h, deltati, grid, K, beta, M_simu, eps=1e-6):
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
        data_sb[k] = simulate_sbbts_fft(N, M, X, N_pi, h, deltati, grid, K, beta, eps)
        if k == 0:
            mm = (time.perf_counter() - time1) * (M_simu - 1) / 60
            st += datetime.timedelta(minutes=mm)
            print(f'Expected finish time: {st.strftime("%H:%M:%S")}', flush=True)

    print(f'Finish time: {datetime.datetime.now().strftime("%H:%M:%S")}', flush=True)
    print(f'Time to generate {M_simu} samples with N_pi={N_pi}: {int(time.perf_counter() - time1)} seconds.',
          flush=True)
    return data_sb[:, 1:]
