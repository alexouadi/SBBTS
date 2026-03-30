import gc

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from early_stopping import EarlyStopping

def get_loss(model, y_0, y_T, T, eps=None, t=None, safe_t=1e-2):
    """Get loss.

    Args:
        model: Neural network model used to estimate the SBBTS drift.
        y_0: Initial trajectory values or past observations.
        y_T: Target values at the next time step / horizon T.
        T: Final time horizon.
        eps: Optional Gaussian noise used to sample Brownian-bridge states.
        t: Continuous time variable.
        safe_t: Small epsilon to avoid numerical issues near t=T.

    Returns:
        Computed output(s) produced by the function.
    """
    B, L, d = y_T.shape
    h_n = model.tf_encoder(y_0, training=True)
    if eps is None:
        eps = torch.randn_like(y_T, device=y_0.device)
        t = torch.FloatTensor(B, L, 1).uniform_(0, T - safe_t).to(y_0.device)

    sigma = torch.sqrt((t * (1 - t / T)))
    mu_t = (1 - t / T) * y_0 + t / T * y_T
    y_t = mu_t + sigma * eps

    score_pred = model.get_drift(t, y_t, h_n)  # B, L-1, d
    score_target = (y_T - y_t) / (T - t)
    return ((score_target - score_pred) ** 2).sum(dim=-1).mean()

def training_sbbts_dsbm_inv(X, model, model_inv, T, beta, K, n_epochs=100, batch_size=32, safe_t=1e-2, lr=1e-3):
    """Training sbbts dsbm inv.

    Args:
        X: Input time-series samples.
        model: Neural network model used to estimate the SBBTS drift.
        model_inv: Inverse/auxiliary model trained jointly with the forward model.
        T: Final time horizon.
        beta: Regularization/transport parameter beta from the SBBTS objective.
        K: Number of outer transport-map update iterations.
        n_epochs: Maximum number of epochs per outer iteration.
        batch_size: Mini-batch size for optimization.
        safe_t: Small epsilon to avoid numerical issues near t=T.
        lr: Learning rate for the optimizer.

    Returns:
        Computed output(s) produced by the function.
    """
    device = X.device
    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer_inv = optim.Adam(model_inv.parameters(), lr=lr)
    patience = 20

    M, L, d = X.shape
    x_0 = X[:, :-1]  # M, L-1, d
    x_T = X[:, 1:]  # M, L-1, d

    train_dataset = TensorDataset(x_0[:int(0.5 * len(x_0))], x_T[:int(0.5 * len(x_0))])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_dataset_inv = TensorDataset(x_0[int(0.5 * len(x_0)):], x_T[int(0.5 * len(x_0)):])
    train_loader_inv = DataLoader(train_dataset_inv, batch_size=batch_size, shuffle=True)

    for k in range(K):
        print()
        print(f"Training s^{k + 1}: ")

        curr_epoch = min(n_epochs, max(1000, int(n_epochs * np.exp(-0.2 * k))))

        early_stopping = EarlyStopping(patience=patience, delta=1e-3)
        for epoch in range(curr_epoch):
            total_loss = 0.0
            model.train()

            for batch in train_loader:
                x_0_, x_T_ = batch  # B, L-1, d

                if k == 0:
                    y_0_ = x_0_.clone()
                    y_T_ = x_T_.clone()
                else:
                    t_0 = torch.zeros(len(x_0_), L - 1, 1, device=device)
                    y_0_ = model_inv(t_0, x_0_)  # B, L-1, d

                    t_N = torch.ones(len(x_0_), L - 1, 1, device=device) * (T - safe_t)
                    y_T_ = model_inv(t_N, x_T_)  # B, L-1, d

                loss = get_loss(model, y_0_, y_T_, T, safe_t=safe_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{curr_epoch}] - Training Loss: {total_loss / len(train_loader):.4f}')

            early_stopping(total_loss / len(train_loader), model)
            if early_stopping.early_stop:
                early_stopping.load_best_model(model)
                print(
                    f"Early stopping at epoch {early_stopping.current_epoch},"
                    f" best epochs = {early_stopping.best_epoch},"
                    f" best loss = {np.round(early_stopping.best_score, 4)}")
                break

        early_stopping = EarlyStopping(patience=patience, delta=1e-3)
        for epoch in range(curr_epoch):
            total_loss = 0.0
            model.train()

            for batch in train_loader_inv:
                x_0_, x_T_ = batch  # B, L-1, d

                h = model.tf_encoder(x_0_, training=True)
                t_0 = torch.zeros(len(x_0_), L - 1, 1, device=device)
                Y0 = x_0_ + 1 / beta * model.get_drift(t_0, x_0_, h)  # B, L-1, d
                y_0_ = model_inv(t_0, Y0)

                t_N = torch.ones(len(x_0_), L - 1, 1, device=device) * (T - safe_t)
                YT = x_T_ + 1 / beta * model.get_drift(t_N, x_T_, h)  # B, L-1, d
                y_T_ = model_inv(t_N, YT)

                loss_inv = (F.mse_loss(x_0_, y_0_) + F.mse_loss(x_T_, y_T_)) * 0.5
                optimizer_inv.zero_grad()
                loss_inv.backward()
                optimizer_inv.step()
                total_loss += loss_inv.item()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f'Epoch [{epoch + 1}/{curr_epoch}] - Training Inv Loss: {total_loss / len(train_loader_inv):.4f}')

            early_stopping(total_loss / len(train_loader_inv), model_inv)
            if early_stopping.early_stop:
                early_stopping.load_best_model(model_inv)
                print(
                    f"Early stopping at epoch {early_stopping.current_epoch},"
                    f" best epochs = {early_stopping.best_epoch},"
                    f" best loss = {np.round(early_stopping.best_score, 4)}")
                break

    t_0 = torch.zeros(1, 1, 1, device=device)
    y_0 = model_inv(t_0, x_0[:1, :1])  # 1, 1, d
    return model, y_0.detach()[0, 0]

def clean_memory(device):
    """Clean memory.

    Args:
        device: Torch device used for allocations and cleanup.

    Returns:
        None.
    """
    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
