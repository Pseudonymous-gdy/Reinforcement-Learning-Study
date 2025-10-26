import numpy as np
from typing import Optional, List, Tuple, Dict, Any

class Environment:
    def __init__(self, number_of_bandits: int, distribution: str = 'bernoulli', seed: Optional[int] = None):
        self.number_of_bandits = number_of_bandits
        self.distribution = distribution
        self.seed = seed

        # Use a dedicated RNG per environment for reproducible but non-constant sampling
        # If seed is provided, the sequence is deterministic; otherwise it's random.
        self.rng = np.random.default_rng(seed)

        # Generate parameters using the instance RNG (so they are controlled by seed)
        self.bandit_means = self.rng.random(number_of_bandits)
        self.bandit_means = np.sort(self.bandit_means)[::-1]
        self.standard_deviations = self.rng.random(number_of_bandits) * 0.1  # not used for now
        self.optimal_mean = self.bandit_means[0]

        self.history: List[Tuple[int, float]] = []

    def get_reward(self, action: int) -> float:
        # Sample using the instance RNG without re-seeding on each call
        if self.distribution == 'bernoulli':
            reward = self.rng.binomial(1, float(self.bandit_means[action]))
            self.history.append((action, float(reward)))
            return float(reward)
        elif self.distribution == 'normal':
            reward = self.rng.normal(float(self.bandit_means[action]), float(self.standard_deviations[action]))
            self.history.append((action, float(reward)))
            return float(reward)
        else:
            raise ValueError("Unsupported distribution type")
        
    def get_optimal_action(self) -> int:
        # Keep semantics: sample from the optimal arm's distribution
        return int(self.rng.binomial(1, float(self.optimal_mean)))

    def get_dueling(self, a: int, b: int) -> int:
        """Compare two arms a and b by their mean rewards.

        Returns 1 if arm a is better than arm b (mean[a] > mean[b]), else 0.
        In case of a tie (equal means), returns 0.

        Raises:
            IndexError: if a or b are out of range.
        """
        if not (0 <= a < self.number_of_bandits) or not (0 <= b < self.number_of_bandits):
            raise IndexError("Arm index out of range")
        return 1 if float(self.bandit_means[a]) > float(self.bandit_means[b]) else 0
    
    def get_bandit_means(self) -> List[float]:
        return self.bandit_means.tolist()
    
    def get_history(self) -> List[Tuple[int, float]]:
        return self.history
    
    def reset(self, seed: Optional[int] = None):
        """Reset environment state.

        - Always clears history.
        - If a seed is provided, reinitialize the RNG AND regenerate bandit parameters
          deterministically from that seed to reproduce the same initial state as construction.
        - If no seed is provided, parameters remain as-is and RNG state is unchanged.
        """
        self.history = []
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            # Regenerate parameters to match initial construction state
            self.bandit_means = self.rng.random(self.number_of_bandits)
            self.bandit_means = np.sort(self.bandit_means)[::-1]
            self.standard_deviations = self.rng.random(self.number_of_bandits) * 0.1
            self.optimal_mean = self.bandit_means[0]

    def set_mean(self, arm: int, new_mean: float):
        """Set a new mean for a specific arm (for testing purposes)."""
        if not (0 <= arm < self.number_of_bandits):
            raise IndexError("Arm index out of range")
        self.bandit_means[arm] = new_mean
        self.bandit_means = np.sort(self.bandit_means)[::-1]
        self.optimal_mean = self.bandit_means[0]

    def set_means(self, new_means: List[float]):
        """Set new means for all arms (for testing purposes)."""
        if len(new_means) != self.number_of_bandits:
            raise ValueError("Length of new_means must match number_of_bandits")
        self.bandit_means = np.array(new_means)
        self.bandit_means = np.sort(self.bandit_means)[::-1]
        self.optimal_mean = self.bandit_means[0]