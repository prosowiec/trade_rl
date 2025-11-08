import numpy as np


class DifferentialSharpeRatio:
    def __init__(self, eta=0.01):
        self.eta = eta
        self.A = 0.0
        self.B = 0.0
        self.initialized = False
    
    def update(self, R_t):
        if not self.initialized:
            self.A = R_t
            self.B = R_t**2
            self.initialized = True
            return 0.0
        
        # Increments
        delta_A = R_t - self.A
        delta_B = R_t**2 - self.B

        # Compute numerator and denominator
        numerator = self.B * delta_A - 0.5 * self.A * delta_B
        denominator = (self.B - self.A**2 + 1e-8) ** (1.5)

        D_t = numerator / (denominator + 1e-8)

        # Update A, B (exponentially smoothed estimates)
        self.A += self.eta * delta_A
        self.B += self.eta * delta_B

        return D_t

def get_sharpe(values: list[float]) -> float:
    if len(values) > 1:
        values = np.array(values)
        returns = np.diff(values) / (values[:-1] + 1e-8)
        if len(returns) > 1:
            mean_r = np.mean(returns)
            std_r = np.std(returns) + 1e-8
            return mean_r / std_r
    return 0.0


def get_assets_sharpes(asset_value_history):
    asset_sharpes = [get_sharpe(values) for values in asset_value_history]
    asset_sharpes = np.array(asset_sharpes)

    return asset_sharpes
