import numpy as np

class DataGenerator:
    def __init__(self, M: int) -> None:
        """Initialize the module/class state.

        Configure internal attributes used by the SBBTS model and utilities.

        Args:
            M: Number of synthetic paths to generate.

        Returns:
            None.
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
        """Generate heston.

    Args:
            r_range: Sampling range for drift parameter r.
            kappa_range: Sampling range for mean-reversion speed kappa.
            theta_range: Sampling range for long-run variance theta.
            rho_range: Sampling range for correlation rho.
            xi_range: Sampling range for volatility-of-volatility xi.
            N: Number of time points (or sequence length minus one, depending on context).
            dt: Time discretization step.
            S0: Initial asset price.
            v0: Initial variance level.

        Returns:
            Computed output(s) produced by the function.
        """

        def simulate_ig(mu: float, lam: float) -> float:
            """Simulate ig.

    Args:
                mu: Mean parameter of the inverse-Gaussian proposal.
                lam: Shape parameter of the inverse-Gaussian proposal.

            Returns:
                Computed output(s) produced by the function.
            """
            G = np.random.randn()
            Y = G ** 2
            X = mu + (0.5 / lam) * (mu ** 2 * Y - mu * np.sqrt(4 * mu * lam * Y + (mu * Y) ** 2))
            U = np.random.uniform()
            return X if U <= mu / (mu + X) else mu ** 2 / X

        def simulate_vol(kappa: float, theta: float, xi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Simulate vol.

    Args:
                kappa: Mean-reversion speed.
                theta: Long-run variance level.
                xi: Volatility-of-volatility parameter.

            Returns:
                Computed output(s) produced by the function.
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
            """Simulate h.

    Args:
                r: Drift coefficient.
                kappa: Mean-reversion speed.
                theta: Long-run variance level.
                rho: Correlation between price and variance shocks.
                xi: Volatility-of-volatility parameter.

            Returns:
                Computed output(s) produced by the function.
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
