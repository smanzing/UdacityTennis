import random
import numpy as np
import copy


# Taken from https://github.com/udacity/deep-reinforcement-learning/blob/dc65050c8f47b365560a30a112fb84f762005c6b/ddpg-pendulum/ddpg_agent.py
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        # originally, random.random() was used instead of random.gauss(0., 1.)
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.gauss(0., 1.) for i in range(len(x))])
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
