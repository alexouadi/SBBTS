import gc

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from early_stopping import EarlyStopping


def get_loss(model, y_0, y_T, T, eps=None, t=None, safe_t=1e-2):
    """Get loss.

    Get loss. This routine is part of the SBBTS workflow and related utilities.

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
    loss = ((score_pred - score_target) ** 2).sum(dim=-1)
    return loss.mean()


def training_sbbts_dsbm(X, model, T, beta, K, n_epochs=100, batch_size=32, patience=10, delta=1e-3, safe_t=1e-2,
                        lr=1e-3):
    """Training sbbts dsbm.

    Training sbbts dsbm. This routine is part of the SBBTS workflow and related utilities.

    Args:
        X: Input time-series samples.
        model: Neural network model used to estimate the SBBTS drift.
        T: Final time horizon.
        beta: Regularization/transport parameter beta from the SBBTS objective.
        K: Number of outer transport-map update iterations.
        n_epochs: Maximum number of epochs per outer iteration.
        batch_size: Mini-batch size for optimization.
        patience: Early-stopping patience in epochs.
        delta: Minimum improvement threshold used by early stopping.
        safe_t: Small epsilon to avoid numerical issues near t=T.
        lr: Learning rate for the optimizer.

    Returns:
        Computed output(s) produced by the function.
    """
    device = X.device
    optimizer = optim.Adam(model.parameters(), lr=lr)

    M, L, d = X.shape
    x_0 = X[:, :-1]  # M, L-1, d
    x_T = X[:, 1:]  # M, L-1, d

    size = int(M * 0.8)
    train_dataset = TensorDataset(x_0[:size], x_T[:size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for k in range(K):
        print()
        print(f"Training s^{k + 1}: ")

        early_stopping = EarlyStopping(patience=patience, delta=delta)
        curr_epoch = min(n_epochs, max(1000, int(n_epochs * np.exp(-0.2 * k))))
        train_loss, val_loss = [], []

        eps = torch.randn_like(x_T[size:]).to(device)
        t = torch.FloatTensor(len(x_T[size:]), L - 1, 1).uniform_(0, T - safe_t).to(device)
        val_dataset = TensorDataset(x_0[size:], x_T[size:], eps, t)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(curr_epoch):
            total_loss = 0.0
            model.train()

            for batch in train_loader:
                x_0_, x_T_ = batch  # B, d

                if k == 0:
                    y_0_ = x_0_.clone()
                    y_T_ = x_T_.clone()
                else:
                    h_n = model.tf_encoder(x_0_, training=True)
                    t_0 = torch.zeros(len(x_0_), L - 1, 1, device=device)
                    y_0_ = x_0_ - 1 / beta * model.get_drift(t_0, x_0_, h_n)  # B, L-1, d

                    t_N = torch.ones(len(x_0_), L - 1, 1, device=device) * (T - safe_t)
                    y_T_ = x_T_ - 1 / beta * model.get_drift(t_N, x_T_, h_n)  # B, L-1, d

                loss = get_loss(model, y_0_, y_T_, T, safe_t=safe_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_loss.append(total_loss / len(train_loader))

            model.eval()
            with torch.no_grad():
                total_loss = 0

                for batch in val_loader:
                    x_0_, x_T_, eps, t = batch  # B, d

                    if k == 0:
                        y_0_ = x_0_.clone()
                        y_T_ = x_T_.clone()
                    else:
                        h_n = model.tf_encoder(x_0_, training=True)
                        t_0 = torch.zeros(len(x_0_), L - 1, 1, device=device)
                        y_0_ = x_0_ - 1 / beta * model.get_drift(t_0, x_0_, h_n)  # B, L-1, d

                        t_N = torch.ones(len(x_0_), L - 1, 1, device=device) * (T - safe_t)
                        y_T_ = x_T_ - 1 / beta * model.get_drift(t_N, x_T_, h_n)  # B, L-1, d

                    loss = get_loss(model, y_0_, y_T_, T, eps, t, safe_t=safe_t)
                    total_loss += loss.item()

                val_loss.append(total_loss / len(val_loader))

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f'Epoch [{epoch + 1}/{curr_epoch}] - Training Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}')

            early_stopping(val_loss[-1], model)
            if early_stopping.early_stop:
                early_stopping.load_best_model(model)
                print(
                    f"Early stopping at epoch {early_stopping.current_epoch},"
                    f" best epochs = {early_stopping.best_epoch},"
                    f" best val loss = {np.round(early_stopping.best_score, 4)}")
                break

    h_n = model.tf_encoder(x_0[:1, :1])
    t_0 = torch.zeros(1, 1, 1, device=device)
    y_0 = x_0[:1, :1] - 1 / beta * model.get_drift(t_0, x_0[:1, :1], h_n)  # 1, 1, d
    return model, y_0.detach()[0, 0]


def clean_memory(device):
    """Clean memory.

    Clean memory. This routine is part of the SBBTS workflow and related utilities.

    Args:
        device: Torch device used for allocations and cleanup.

    Returns:
        None.
    """
    gc.collect()
    if device != torch.device('cpu'):
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
