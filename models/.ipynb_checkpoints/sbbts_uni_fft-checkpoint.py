import numpy as np
import scipy.fft as fft
import datetime
import time
from scipy.interpolate import interp1d


def kernel(x,h):
    """
    Kernel function used for kernel regression.
    :params x:; [float]
    :params h: kernel bandwidth; [float]
    return: kernel function of shape (len(x),); [np.array]
    """
    return np.where(np.abs(x) < h, (h ** 2 - x ** 2) ** 2 / h, 0)


def simulate_kernel_sbts(N, M, X, N_pi, h, deltati, grid, K, beta, eps=1e-6):
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
    L = grid[-1] - grid[0]
    term_2 = np.log(2 * np.pi * deltati) / 2
    grid_vect = grid[np.newaxis, :]

    fvf = fft.fftfreq(n, d=L/n) * 2 * np.pi
    gaussian_kernel = 1 / np.sqrt(2 * np.pi * deltati) * np.exp(-(grid ** 2) / (2 * deltati ** 2))
    fft_gaussian = fft.fft(gaussian_kernel)

    for i in range(N):
        if i > 0:
            weights[:] *= kernel(X[:, i] - X_, h)
        else:
            weights[:] = 1 / M
        weights_vect = weights[:, np.newaxis]

        # Initialization
        phi_values = np.zeros(n)

        # Iterate until convergence towards phi*
        for k in range(K):
            fft_phi = fft.fft(np.exp(phi_values))
            fft_h_k = fft_gaussian * fft_phi
            h_k = fft.ifft(fft_h_k).real  # h^k over all the grid

            y_k_index = np.argmin(np.log(h_k) + 0.5 * beta * (X_ - grid) ** 2)
            y_k = grid[y_k_index]

            # Compute the transport map
            fft_grad_phi = 1j * fvf * fft_phi
            grad_phi = fft.ifft(fft_grad_phi).real

            msX = grid + 1 / beta * grad_phi
            msY_ = msX - 1 / beta * grad_phi  # False, to be corrected
            msY_k = interp1d(msX, msY_, fill_value='extrapolate')
            msY = msY_k(X[:, i + 1])  # msY pushforward mu_{i+1}

            # Update the potential
            msY_vect = msY[:, np.newaxis]
            term_1 = np.log(np.mean(kernel(msY_vect - grid_vect, h) * weights_vect, axis=0))
            term_3 = (grid - y_k) ** 2 / (2 * deltati)
            phi_values_new = term_1 + term_2 + term_3 
            
            if np.max(np.abs(phi_values - phi_values_new)) < eps:  # convergence is reached
                phi_values = phi_values_new
                break
                
            phi_values = phi_values_new

        fft_phi_star = fft.fft(np.exp(phi_values))
        for k in range(len(v_time_step_Euler) - 1):
            timeprev = v_time_step_Euler[k]
            timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]

            # Compute h*
            gaussian_kernel_star = 1 / np.sqrt(2 * np.pi * (deltati - timeprev)) * np.exp(
                -(grid ** 2) / (2 * (deltati - timeprev) ** 2)
            )
            fft_gaussian_star = fft.fft(gaussian_kernel_star)
            fft_h_star = fft_gaussian_star * fft_phi_star
            h_star = fft.ifft(fft_h_star).real

            # Compute msY* at time t via grid search
            msY_star = np.argmin(np.log(h_star) + 0.5 * beta * (X_ - grid) ** 2)

            # Compute the drift and volatility
            fft_log_h_star = fft.fft(np.log(h_star))
            grad_log_h_star = fft.ifft(1j * fvf * fft_log_h_star).real
            fft_hessian_log_h_star = - fvf ** 2 * fft_log_h_star
            hessian_log__hstar = fft.ifft(fft_hessian_log_h_star).real

            drift = grad_log_h_star[msY_star]
            vol = 1 + 1 / beta * hessian_log__hstar[msY_star]

            X_ += drift * timestep + Brownian[index_] * np.sqrt(vol * timestep)
            index_ += 1
                
        timeSeries[i + 1] = X_

    return timeSeries


def simusbbts(N, M, X, N_pi, h, deltati, grid, K, beta, M_simu, eps=1e-6):
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
        data_sb[k] = simulate_kernel_sbts(N, M, X, N_pi, h, deltati, grid, K, beta, eps)
        if k == 0:
            mm = (time.perf_counter() - time1) * (M_simu - 1) / 60
            st += datetime.timedelta(minutes=mm)
            print(f'Expected finish time: {st.strftime("%H:%M:%S")}', flush=True)

    print(f'Finish time: {datetime.datetime.now().strftime("%H:%M:%S")}', flush=True)
    print(f'Time to generate {M_simu} samples with N_pi={N_pi}: {int(time.perf_counter() - time1)} seconds.',
          flush=True)
    return data_sb[:, 1:]
