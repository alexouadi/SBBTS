import numpy as np


class Generate_Data:
    def __init__(self, M):
        """
        :params M: number of samples to generate; [int]
        """
        self.M = M

    def generate_GARCH(self, N, alpha_0=0.5, alpha_1=0.4, alpha_2=0.1, s=0.1, x0=0):
        """
        Simulate M univariate GARCH processes.
        :params N: number of time steps to generate (without the x0); [int]
        :params alpha_0, alpha_1, alpha_2, s: model parameters; [float]
        :params x0: time series initialization; [float]
        return: M time series of shape (M, N + 1); [np.array]
        """
        def simulate():
            time_series = list()
            x_next, x_prev = 0, 0
    
            for t in range(N + 50):
                if t >= 50:
                    time_series.append(x_next)
                sigma = np.sqrt(alpha_0 + alpha_1 * x_next ** 2 + alpha_2 * x_prev ** 2)
                x_prev = x_next
                x_next = sigma * np.random.normal(scale=s)
            
            return time_series

        X = np.array([simulate() for i in range(self.M)])
        X_garch = np.zeros((self.M, N + 1))
        X_garch[:, 0], X_garch[:, 1:] = x0, X
        return X_garch
