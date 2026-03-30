import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def marchenko_pastur_lambda_plus(n, d, sigma2=1.0):
    """Compute the Marchenko-Pastur upper edge used for factor selection.

    Args:
        n: Number of time observations in the return matrix.
        d: Number of assets/features.
        sigma2: Reference noise variance in the random-matrix model.

    Returns:
        Scalar threshold λ+ above which eigenvalues are treated as signal.
    """
    return sigma2 * (1.0 + np.sqrt(d / n)) ** 2

def get_decomposition(X):
    """Decompose returns into factor and residual components.

    Args:
        X: Return matrix of shape (n_samples, n_assets).

    Returns:
        Tuple containing:
        (F, F_scaled, P_m, Z, mu_hat, sigma_hat, eigvals_m, eigvals), where
        F are retained factors, Z are residuals, and P_m is the PCA loading matrix.
    """
    n, d = X.shape
    mu_hat = X.mean(axis=0)
    sigma_hat = X.std(axis=0, ddof=1)

    X_bar = (X - mu_hat) / sigma_hat

    Sigma_hat = (X_bar.T @ X_bar) / (n - 1)

    eigvals, eigvecs = np.linalg.eigh(Sigma_hat)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    P = eigvecs[:, idx]

    lambda_plus = marchenko_pastur_lambda_plus(n, d)
    m = max(int(np.sum(eigvals > lambda_plus)), 1)

    P_m = P[:, :m]  # (d, m)
    F = X_bar @ P_m  # (n, m)

    eigvals_m = eigvals[:m]
    sqrt_lambdas = np.sqrt(np.where(eigvals_m <= 0, 1e-12, eigvals_m))
    F_scaled = F / sqrt_lambdas.reshape((1, -1))

    Z = X_bar - F @ P_m.T  # (n, d)
    return F, F_scaled, P_m, Z, mu_hat, sigma_hat, eigvals_m, eigvals

def get_cluster(F_scaled, eigvals_m, window, nc=3):
    """Cluster extracted factors using distributional and spectral features.

    Args:
        F_scaled: Normalized factor matrix of shape (n_samples, n_factors).
        eigvals_m: Eigenvalues corresponding to retained factors.
        window: Window length used to build sub-series for each factor.
        nc: Number of factor clusters (typically 3 in experiments).

    Returns:
        (cluster_sets, labels) where cluster_sets maps cluster id to stacked
        windows and labels gives the cluster assignment of each factor.
    """
    n, m = F_scaled.shape
    feats = []

    for j in range(m):
        col = F_scaled[:, j]
        mean = np.mean(col)
        std = np.std(col, ddof=1)
        skew = stats.skew(col)
        kurt = stats.kurtosis(col, fisher=False)  # kurtosis (Pearson) if wanted
        ev = np.log(eigvals_m[j]) if eigvals_m[j] > 0 else 0.0
        feats.append([mean, std, skew, kurt, ev])

    feats = np.array(feats)  # shape (m, 5)
    scaler = StandardScaler()
    feats_s = scaler.fit_transform(feats)
    kmeans = KMeans(n_clusters=nc, random_state=0)
    labels = kmeans.fit_predict(feats_s)  # length m

    cluster_sets = {}
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        idxs = np.where(labels == lab)[0]
        windows_list = []
        for j in idxs:
            col = F_scaled[:, j]
            if n < window:
                raise ValueError("n < window, cannot construct sub-series")
            ws = np.lib.stride_tricks.sliding_window_view(col, window_shape=window)  # shape (n - s + 1, s)
            windows_list.append(ws)
        if len(windows_list) == 0:
            cluster_sets[lab] = np.empty((0, window))
        else:
            cluster_sets[lab] = np.vstack(windows_list)

    return cluster_sets, labels

def get_F_synth(clusters_synth, labels, eigvals_m, window, m, M_f):
    """Reassemble synthetic latent factors from cluster-wise generations.

    Args:
        clusters_synth: Synthetic windows generated independently per cluster.
        labels: Cluster label of each retained factor dimension.
        eigvals_m: Retained eigenvalues used to rescale factors to original variance.
        window: Temporal length of each generated factor trajectory.
        m: Number of retained factors.
        M_f: Number of synthetic trajectories to generate.

    Returns:
        Synthetic factor tensor of shape (M_f, window, m), rescaled by eigvals_m.
    """
    F_synth = np.zeros((M_f, window, m))
    cluster_index = np.zeros(len(clusters_synth), dtype=int)
    for k, ind in enumerate(labels):
        curr_ind = cluster_index[ind]
        F_synth[:, :, k] = clusters_synth[ind][curr_ind * M_f: (curr_ind + 1) * M_f]
        cluster_index[ind] += 1

    rescale = np.sqrt(np.where(eigvals_m <= 0, 1e-12, eigvals_m)).reshape((1, -1))
    return F_synth * rescale

def fit_Z_gmm(Z, n_components=2, random_state=None, max_iter=200, tol=1e-4):
    """Fit one Gaussian Mixture Model per residual dimension.

    Args:
        Z: Residual matrix of shape (n_samples, n_assets).
        n_components: Number of Gaussian components in each univariate mixture.
        random_state: Optional random seed for EM initialization.
        max_iter: Maximum number of EM iterations.
        tol: Convergence tolerance for EM stopping.

    Returns:
        List of fitted per-asset mixture parameters (weights, means, variances).
    """
    n_assets = Z.shape[1]
    fitted = []

    for i in tqdm(range(n_assets), desc='Fitting 2‑Gaussian GMM per asset'):
        x = Z[:, i][:, np.newaxis]

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="diag",
            random_state=random_state,
            max_iter=max_iter,
            tol=tol,
            reg_covar=1e-6
        )
        gmm.fit(x)

        params = {
            "weights": gmm.weights_,
            "means": gmm.means_.ravel(),
            "covs": gmm.covariances_.ravel()
        }

        params["covs"] = np.maximum(params["covs"], 1e-8)
        fitted.append(params)

    return fitted

def get_Z_synth_gmm(Z_fitted, F_synth, d):
    """Sample synthetic residual series from fitted per-asset GMMs.

    Args:
        Z_fitted: List of fitted GMM parameter dictionaries from `fit_Z_gmm`.
        F_synth: Synthetic factor tensor used to infer output path/time shape.
        d: Number of assets/residual dimensions to sample.

    Returns:
        Residual tensor of shape (n_paths, n_steps, d).
    """
    n_paths, n_steps = F_synth.shape[:2]
    Z_synth = np.zeros((n_paths, n_steps, d))

    for j in tqdm(range(d), desc='Sampling from 2‑Gaussian GMM per asset'):
        params = Z_fitted[j]
        w, mu, var = params["weights"], params["means"], params["covs"]
        sigma = np.sqrt(var)

        comp_idx = np.random.choice(
            a=len(w),
            size=(n_paths, n_steps),
            p=w
        )

        draws = norm.rvs(
            loc=mu[comp_idx],
            scale=sigma[comp_idx]
        )
        Z_synth[:, :, j] = draws

    return Z_synth

def reconstruct_X(F_synth, Z_synth, P_m, mu_hat, sigma_hat):
    """Reconstruct synthetic returns from factors and residual components.

    Args:
        F_synth: Synthetic factor tensor of shape (n_paths, n_steps, m).
        Z_synth: Synthetic residual tensor of shape (n_paths, n_steps, d).
        P_m: PCA loading matrix of shape (d, m).
        mu_hat: Per-asset empirical mean used during standardization.
        sigma_hat: Per-asset empirical standard deviation used during standardization.

    Returns:
        Synthetic return tensor in the original scale, shape (n_paths, n_steps, d).
    """
    X_synth = F_synth @ P_m.T + Z_synth
    X_synth = X_synth * sigma_hat + mu_hat
    return X_synth
