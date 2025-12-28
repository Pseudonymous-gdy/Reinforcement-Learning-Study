import numpy as np
from typing import Optional, List, Tuple, Union

class Environment:
    def __init__(
        self,
        number_of_bandits: int,
        distribution: str = 'bernoulli',
        seed: Optional[int] = None,
        dueling_means: Union[np.ndarray, List, None] = None
    ):
        self.number_of_bandits = int(number_of_bandits)
        self.distribution = distribution
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # DO NOT SORT: arm identity must remain fixed
        self.bandit_means = self.rng.random(self.number_of_bandits)
        self.standard_deviations = self.rng.random(self.number_of_bandits) * 0.1

        self.dueling_means = None
        if dueling_means is not None:
            self.set_dueling_means(dueling_means)

        self.history: List[Tuple[int, float]] = []

    def get_reward(self, action: int) -> float:
        if not (0 <= action < self.number_of_bandits):
            raise IndexError("Arm index out of range")

        if self.distribution == 'bernoulli':
            r = float(self.rng.binomial(1, float(self.bandit_means[action])))
        elif self.distribution == 'normal':
            r = float(self.rng.normal(float(self.bandit_means[action]),
                                      float(self.standard_deviations[action])))
        else:
            raise ValueError("Unsupported distribution type")

        self.history.append((int(action), r))
        return r

    def get_optimal_action(self) -> int:
        # return index, not reward
        return int(np.argmax(self.bandit_means))

    def get_optimal_reward(self) -> float:
        return self.get_reward(self.get_optimal_action())

    def get_dueling(self, a: int, b: int) -> float:
        """Return 1 if a wins b, 0 if b wins a, 0.5 if tie (only possible for normal sampling)."""
        if not (0 <= a < self.number_of_bandits) or not (0 <= b < self.number_of_bandits):
            raise IndexError("Arm index out of range")
        if a == b:
            return 0.5

        if self.distribution == 'normal':
            ra = self.rng.normal(float(self.bandit_means[a]), float(self.standard_deviations[a]))
            rb = self.rng.normal(float(self.bandit_means[b]), float(self.standard_deviations[b]))
            if ra > rb: return 1.0
            if ra < rb: return 0.0
            return 0.5

        if self.distribution == 'bernoulli':
            if self.dueling_means is None:
                prob = (float(self.bandit_means[a]) - float(self.bandit_means[b])) / 2.0 + 0.5
            else:
                prob = float(self.dueling_means[a, b])
            # optional safety
            prob = min(1.0, max(0.0, prob))
            return float(self.rng.binomial(1, prob))

        raise ValueError("Unsupported distribution type")

    def duel(self, a: int, b: int) -> int:
        """Return winner index (a or b). This is the clean interface for algorithms."""
        out = self.get_dueling(a, b)
        if out == 1.0:
            return int(a)
        if out == 0.0:
            return int(b)
        # tie: break randomly
        return int(self.rng.choice([a, b]))

    def reset(self, seed: Optional[int] = None):
        self.history = []
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.bandit_means = self.rng.random(self.number_of_bandits)
            self.standard_deviations = self.rng.random(self.number_of_bandits) * 0.1

    def set_mean(self, arm: int, new_mean: float):
        if not (0 <= arm < self.number_of_bandits):
            raise IndexError("Arm index out of range")
        self.bandit_means[arm] = float(new_mean)

    def set_means(self, new_means: List[float]):
        if len(new_means) != self.number_of_bandits:
            raise ValueError("Length of new_means must match number_of_bandits")
        self.bandit_means = np.array(new_means, dtype=float)

    def set_dueling_means(self, new_means: Union[List[float], np.ndarray]):
        mat = np.array(new_means, dtype=float)
        if mat.shape != (self.number_of_bandits, self.number_of_bandits):
            raise ValueError("dueling_means must be a KxK matrix")
        self.dueling_means = mat
