import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def marchenko_pastur_lambda_plus(n, d, sigma2=1.0):
    """Compute the Marchenko-Pastur upper eigenvalue threshold used to retain PCA factors.

    Args:
        n: Number of observations.
        d: Number of assets/features.
        sigma2: Noise variance used in MP threshold.

    Returns:
        Scalar λ+ threshold.
    """
    return sigma2 * (1.0 + np.sqrt(d / n)) ** 2

def get_decomposition(X):
    """Decompose standardized returns into common factors and idiosyncratic residuals.

    Args:
        X: Input time-series tensor or matrix.

    Returns:
        Tuple `(F, F_scaled, P_m, Z, mu_hat, sigma_hat, eigvals_m, eigvals)`.
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
    """Cluster retained PCA factors using moments and eigenvalue-based features.

    Args:
        F_scaled: Scaled PCA factor matrix.
        eigvals_m: Retained eigenvalues.
        window: Sliding window length.
        nc: Number of clusters.

    Returns:
        Tuple `(cluster_sets, labels)` for factor clusters.
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
    """Reconstruct synthetic factor tensor from cluster-wise generated windows.

    Args:
        clusters_synth: Generated windows per cluster.
        labels: Cluster assignment per factor.
        eigvals_m: Retained eigenvalues.
        window: Sliding window length.
        m: Number of retained factors.
        M_f: Number of synthetic factor paths.

    Returns:
        Synthetic factor tensor of shape (M_f, window, m).
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
    """Fit a univariate Gaussian Mixture Model to each residual dimension.

    Args:
        Z: Residual matrix.
        n_components: Number of mixture components.
        random_state: Random seed.
        max_iter: Maximum EM iterations.
        tol: EM convergence tolerance.

    Returns:
        List of fitted mixture parameters for each residual dimension.
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
    """Sample synthetic idiosyncratic residuals from the fitted per-asset GMMs.

    Args:
        Z_fitted: Fitted GMM parameters per residual dimension.
        F_synth: Synthetic factor tensor.
        d: Number of assets/features.

    Returns:
        Synthetic residual tensor of shape (n_paths, n_steps, d).
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
    """Recompose synthetic returns from synthetic factors and residuals, then unscale.

    Args:
        F_synth: Synthetic factor tensor.
        Z_synth: Synthetic residual tensor.
        P_m: PCA loading matrix.
        mu_hat: Empirical mean used for scaling.
        sigma_hat: Empirical std used for scaling.

    Returns:
        Synthetic return tensor in original data scale.
    """
    X_synth = F_synth @ P_m.T + Z_synth
    X_synth = X_synth * sigma_hat + mu_hat
    return X_synth
