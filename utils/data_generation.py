import numpy as np

class DataGenerator:
    def __init__(self, M: int) -> None:
        """Store the number of trajectories to simulate.

        Args:
            M: Number of trajectories to generate.
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
        """Generate multivariate Heston trajectories with stochastic volatility.

        Args:
            r_range: Sampling interval for drift r.
            kappa_range: Sampling interval for mean-reversion speed kappa.
            theta_range: Sampling interval for long-run variance theta.
            rho_range: Sampling interval for correlation rho.
            xi_range: Sampling interval for vol-of-vol xi.
            N: Number of time points.
            dt: Time step.
            S0: Initial asset price.
            v0: Initial variance.

        Returns:
            Array of simulated paths with price and variance channels.
        """

        def simulate_ig(mu: float, lam: float) -> float:
            """Sample one value from an inverse-Gaussian distribution.

            Args:
                mu: Mean parameter.
                lam: Shape parameter.

            Returns:
                One inverse-Gaussian random sample.
            """
            G = np.random.randn()
            Y = G ** 2
            X = mu + (0.5 / lam) * (mu ** 2 * Y - mu * np.sqrt(4 * mu * lam * Y + (mu * Y) ** 2))
            U = np.random.uniform()
            return X if U <= mu / (mu + X) else mu ** 2 / X

        def simulate_vol(kappa: float, theta: float, xi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Simulate the variance process and auxiliary terms for one Heston path.

            Args:
                kappa: Mean-reversion parameter kappa.
                theta: Long-run variance parameter theta.
                xi: Volatility-of-volatility parameter.

            Returns:
                Tuple `(V, U, Z)` for variance path and latent terms.
            """
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
            """Simulate a Heston price/variance path using correlated shocks.

            Args:
                r: Drift parameter r.
                kappa: Mean-reversion parameter kappa.
                theta: Long-run variance parameter theta.
                rho: Correlation between shocks.
                xi: Volatility-of-volatility parameter.

            Returns:
                Tuple `(S, V)` with simulated price and variance paths.
            """
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
