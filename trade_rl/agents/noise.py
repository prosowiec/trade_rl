import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.015, sigma=0.15,  sigma_decay=0.99995, min_sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.min_sigma = min_sigma
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        if self.sigma > self.min_sigma:
            self.sigma *= self.sigma_decay
            self.sigma = max(self.min_sigma, self.sigma_decay)
        return self.state
    
    def __call__(self, action):
        """Call to sample noise."""
        res = action.flatten() + self.sample()
        #res = res / res.sum()
        return np.clip(res,0,1)