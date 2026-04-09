import datetime
import time

import numpy as np
import torch

from training.training_sbbts_dsbm import clean_memory

def generate_dsbm_batch(N, X, model, y_0, N_pi, T, beta, M_simu, safe_t=1e-2, ):
    """Simulate one batch of SBBTS trajectories by Euler discretization of the auxiliary process and inverse transport map.

    Args:
        N: Number of time points.
        X: Input time-series tensor or matrix.
        model: SBBTS drift model.
        y_0: Current/past state used as initial condition.
        N_pi: Number of Euler steps per interval.
        T: Final time horizon.
        beta: SBBTS regularization parameter beta.
        M_simu: Number of simulated trajectories.
        safe_t: Small epsilon to avoid evaluating exactly at t=T.

    Returns:
        Tensor of generated trajectories with shape (M_simu, N, d).
    """
    device = X.device
    d = X.shape[-1]

    # Diffusion calendar
    time_step_Euler = T / N_pi
    v_time_step_Euler = torch.arange(0, T + 1e-9, time_step_Euler).to(device)

    # Generate Brownian increments
    num_brownian = (N * (len(v_time_step_Euler) - 1), M_simu, d)
    Brownian = torch.normal(0, 1, num_brownian).to(device)

    # Simulation initialization
    sbbts_sample = torch.zeros((M_simu, N + 1, d)).to(device)
    index_ = 0

    with torch.no_grad():
        T_tensor = torch.ones((M_simu, 1, 1), device=device) * (T - safe_t)
        Y = torch.ones((M_simu, d), device=device) * y_0.clone()
        past = torch.zeros((M_simu, N + 1, d)).to(device)
        past[:, 0] = y_0.clone()

        for i in range(N):
            Y_past = past[:, :i + 1].clone()
            h_n = model.tf_encoder(Y_past)  # M_simu, i+1, d_model

            for k in range(len(v_time_step_Euler) - 1):
                timeprev = v_time_step_Euler[k]
                timestep = v_time_step_Euler[k + 1] - v_time_step_Euler[k]
                drift = model.get_drift(timeprev.repeat(M_simu, 1, 1), Y.unsqueeze(1), h_n)[:, 0]
                Y += drift * timestep + Brownian[index_] * torch.sqrt(timestep).to(device)
                index_ += 1

            X_ = Y + 1 / beta * model.get_drift(T_tensor, Y.unsqueeze(1), h_n)[:, 0]
            sbbts_sample[:, i + 1] = X_
            past[:, i + 1] = Y

            if i % 20 == 0:
                del X_, h_n, drift
                clean_memory(device)

    return sbbts_sample[:, 1:]

def generate_dsbm(N, X, model, y_0, N_pi, T, beta, M_simu, N_batch, scale=1., safe_t=1e-2, exp=False):
    """Generate SBBTS samples by repeatedly calling batch generation and concatenating outputs.

    Args:
        N: Number of time points.
        X: Input time-series tensor or matrix.
        model: SBBTS drift model.
        y_0: Current/past state used as initial condition.
        N_pi: Number of Euler steps per interval.
        T: Final time horizon.
        beta: SBBTS regularization parameter beta.
        M_simu: Number of simulated trajectories.
        N_batch: Number of generation batches.
        scale: Scale factor applied to generated returns.
        safe_t: Small epsilon to avoid evaluating exactly at t=T.
        exp: If True, exponentiate cumulative returns to price levels.

    Returns:
        NumPy array of generated samples aggregated over all batches.
    """
    base = M_simu // N_batch
    extra = M_simu % N_batch
    samples = []

    for j in range(N_batch):
        if j == 0:
            st = datetime.datetime.now()
            print(f'Start time: {st.strftime("%H:%M:%S")}', flush=True)
            time1 = time.perf_counter()

        M_batch = base + (1 if j < extra else 0)
        print(f"Starting inference batch {j + 1} / {N_batch}")
        sample = generate_dsbm_batch(N, X, model, y_0, N_pi, T, beta, M_batch, safe_t=safe_t)
        if exp:
            sample = np.exp(((sample * scale).cpu().detach().numpy()).cumsum(axis=1))
        else:
            sample = (sample * scale).cpu().detach().numpy()
        samples.append(sample)

        if j == 0:
            mm = (time.perf_counter() - time1) * (N_batch - 1) / 60
            st += datetime.timedelta(minutes=mm)
            print(f'Expected finish time: {st.strftime("%H:%M:%S")}', flush=True)

    return np.concatenate(samples, axis=0)
