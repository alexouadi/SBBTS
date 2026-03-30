from math import sqrt

import torch

from diffusion_dsbm import generate_dsbm
from gmlab_projects.time_series_generation.utils.data_generation import *
from models.sbbts_model import ScoreNN, InverseMLP
from training.training_sbbts_dsbm import training_sbbts_dsbm
from training.training_sbbts_inv import training_sbbts_dsbm_inv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

M = 5000
N_pi = 60
N = 252
Generator_data = DataGenerator(M)

X_data = Generator_data.generate_heston(r_range=[0.01, 0.1], kappa_range=[.5, 4.], theta_range=[.5, 1.5],
                                        rho_range=[-0.9, 0.9], xi_range=[.1, .9], N=N)

log_returns = np.zeros_like(X_data)
log_returns[:, 1:] = np.diff(np.log(X_data), axis=1)
X_data = X_data[:, 1:]
X = torch.tensor(log_returns).to(torch.float32).to(device)
T = 1
scale = X.std(dim=(0, 1)) / sqrt(T)
exp = True

X /= scale
d = X.shape[-1]

beta = 100
K = 5
safe_t = 1e-2
batch_size = 128
n_epochs = 1000
lr = 1e-3
patience = 15
delta = 1e-3

d_model = 128
hidden_dim = 64
nhead = 32
n_layers = 2

model = ScoreNN(d, d_model, hidden_dim, nhead, n_layers, N, device=device).to(device)
if beta >= 100:
    model, y_0 = training_sbbts_dsbm(X, model, T, beta, K, lr=lr, n_epochs=n_epochs, safe_t=safe_t,
                                     batch_size=batch_size, patience=patience, delta=delta)
else:
    model_inv = InverseMLP(
        input_dim=d,
        t_model=8,
        d_model=32
    )
    model_inv.to(device)
    model, y_0 = training_sbbts_dsbm_inv(X, model, model_inv, T, beta, K=K, n_epochs=n_epochs,
                                         batch_size=batch_size, lr=1e-3, safe_t=1e-2)

model.eval()
X_sbb = generate_dsbm(N, X, model, y_0, N_pi=N_pi, T=T, beta=beta, M_simu=4000, N_batch=2,
                      scale=scale, exp=exp, safe_t=safe_t)
