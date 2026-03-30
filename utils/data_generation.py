import numpy as np


class DataGenerator:
    def __init__(self, M: int) -> None:
        """
        Parameters
        ----------
        M : int
            Number of time series to generate.
        """
        self.M = M

    def generate_heston(
            self,
            r_range: list[float],
            kappa_range: list[float],
            theta_range: list[float],
            rho_range: list[float],
            xi_range: list[float],
            N: int,
            dt: float = 1 / 252,
            S0: float = 1.0,
            v0: float = 1.0,
    ) -> np.ndarray:
        """
        Return Heston data following the scheme defined in arxiv.org/pdf/2412.11264.

        Parameters
        ----------
        r_range, kappa_range, theta_range, rho_range, xi_range : list of two floats
            Parameter bounds.
        N : int
            Length of each series.
        dt : float, optional
            Time step.
        S0 : float, optional
            Initial asset price.
        v0 : float, optional
            Initial variance.

        Returns
        -------
        np.ndarray
            Array of shape (M, N+1, 2) where the last dimension contains
            ``price`` and ``variance``.
        """

        def simulate_ig(mu: float, lam: float) -> float:
            """Inverse Gaussian sampler (Michael?Schucany?Haas method)."""
            G = np.random.randn()
            Y = G ** 2
            X = mu + (0.5 / lam) * (mu ** 2 * Y - mu * np.sqrt(4 * mu * lam * Y + (mu * Y) ** 2))
            U = np.random.uniform()
            return X if U <= mu / (mu + X) else mu ** 2 / X

        def simulate_vol(kappa: float, theta: float, xi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            V = np.zeros(N + 1)
            U = np.zeros(N)
            Z = np.zeros(N)
            V[0] = v0
            a, b = kappa * theta, -kappa
            for t in range(N):
                alpha_t = V[t] * (np.exp(b * dt) - 1) / b + a * ((np.exp(b * dt) - 1) / b - dt) / b
                sigma_t = xi / b * (np.exp(b * dt) - 1)
                U[t] = simulate_ig(alpha_t, (alpha_t / sigma_t) ** 2)
                Z[t] = (U[t] - alpha_t) / sigma_t
                V[t + 1] = V[t] + a * dt + b * U[t] + xi * Z[t]
            return V, U, Z

        def simulate_h(r: float, kappa: float, theta: float, rho: float, xi: float) -> tuple[np.ndarray, np.ndarray]:
            V, U, Z = simulate_vol(kappa, theta, xi)
            log_S = np.zeros(N + 1)
            log_S[0] = np.log(S0)
            sq_rho = np.sqrt(1 - rho ** 2)
            for t in range(N):
                log_S[t + 1] = (
                        log_S[t]
                        + r * dt
                        - 0.5 * U[t]
                        + rho * Z[t]
                        + sq_rho * np.sqrt(U[t]) * np.random.randn()
                )
            return np.exp(log_S), V

        heston = np.zeros((self.M, N + 1, 2))

        r = np.random.uniform(r_range[0], r_range[1], self.M)
        kappa = np.random.uniform(kappa_range[0], kappa_range[1], self.M)
        theta = np.random.uniform(theta_range[0], theta_range[1], self.M)
        rho = np.random.uniform(rho_range[0], rho_range[1], self.M)
        xi = np.random.uniform(xi_range[0], xi_range[1], self.M)

        for i in range(self.M):
            price, vol = simulate_h(r[i], kappa[i], theta[i], rho[i], xi[i])
            serie = np.concatenate([price[:, np.newaxis], vol[:, np.newaxis]], axis=1)
            heston[i] = serie

        return heston
