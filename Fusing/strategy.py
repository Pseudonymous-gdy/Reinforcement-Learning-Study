import os
import sys
import math
# Allow running this file directly: ensure repo root is on sys.path
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
from Fusing.env import Environment


class ProbabilisticStrategy:
    """A simple fusion of reward-UCB and dueling-UCB decisions.

    At each step:
    - With probability ``alpha``, select an arm using UCB1 on reward feedback.
    - With probability ``1 - alpha``, select a pair using a dueling-UCB heuristic,
      sample a duel outcome, then pull the winner arm to obtain a reward.

    Both sources update their own statistics. The step always pulls a single arm and
    returns the pulled arm index.

    Notes
    - Reward setting uses classic UCB1: value[i] + sqrt(2 ln t / n_i). Arms with
      n_i == 0 are pulled first (initial exploration) using the strategy RNG.
    - Dueling setting maintains pairwise win rates v[i, j] ~ P(i beats j) and a
      symmetric comparison count n[i, j] == n[j, i]. Pair selection favors uncertain
      pairs; elimination removes j if some i has LCB(i, j) > 0.5.
    """

    def __init__(self, env: Environment, alpha: float = 0.5, seed: Optional[int] = None):
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.env = env
        self.alpha = alpha  # Probability to choose reward-UCB branch

        # Internal RNG for strategy decisions (separate from env RNG)
        self.rng = np.random.default_rng(seed)

        # strategy 1: reward setting (UCB Adopted)
        self.counts: List[int] = [0] * env.number_of_bandits
        self.values: List[float] = [0.0] * env.number_of_bandits
        self.explore_reward: float = 1.0

        # strategy 2: dueling setting (UCB-based heuristic)
        n = env.number_of_bandits
        self.dueling_counts: np.ndarray = np.zeros((n, n), dtype=int)
        self.dueling_values: np.ndarray = np.zeros((n, n), dtype=float)  # v[i,j] ~ P(i beats j)
        self.explore_dueling: float = 1.0

        # Book-keeping
        self.candidates: List[int] = list(range(env.number_of_bandits))
        # History: (is_reward_step, pulled_arm, compared_pair)
        self.history: List[Tuple[bool, Optional[int], Optional[Tuple[int, int]]]] = []
        self.reward_pulls: int = 0
        self.dueling_pulls: int = 0

    def step(self) -> Optional[Union[int, Tuple[int, int]]]:
        # Decide which strategy to use this step
        prob = self.rng.random()
        if prob < self.alpha:
            # Reward setting
            action = self._reward_ucb()
            if action is None:
                return None
            self.reward_pulls += 1
            self.history.append((True, action, None))
            return action
        else:
            # Dueling setting
            compared_pair = self._dueling_ucb()
            if compared_pair is None:
                return None
            self.dueling_pulls += 1
            self.history.append((False, None, compared_pair))
            return compared_pair

    def _reward_ucb(self) -> Optional[int]:
        # Classic UCB1 for reward setting (candidates only)
        if not self.candidates:
            return None
        # Global step count on candidates
        total_counts = sum(self.counts[i] for i in self.candidates)
        t = max(total_counts, 1)

        # Ensure sufficient sampling before elimination: drive all candidates
        # to reach at least min_pulls_elim pulls so that bounds shrink and
        # elimination can actually trigger.
        # Dynamic minimum pulls to enable elimination; grows slowly with time
        min_pulls_elim = max(5, int(2 * math.log(t + 1)))
        under_sampled = [i for i in self.candidates if self.counts[i] < min_pulls_elim]
        if under_sampled:
            chosen_arm = int(self.rng.choice(under_sampled))
        else:
            # Reward-side elimination: remove arm j if UCB(j) < LCB(i) for some i
            bounds: Dict[int, Tuple[float, float]] = {}
            for i in self.candidates:
                n_i = self.counts[i]
                if n_i >= min_pulls_elim:
                    # Use self-normalized bound depending on own pulls to avoid widening with global t
                    bonus = math.sqrt(2.0 * max(math.log(n_i), 1e-9) / n_i)
                    ucb = self.values[i] + self.explore_reward * bonus
                    lcb = self.values[i] - self.explore_reward * bonus
                    bounds[i] = (lcb, ucb)

            to_remove = set()
            # Only consider elimination when both i and j have enough samples
            for j in self.candidates:
                if j not in bounds:
                    continue
                ucb_j = bounds[j][1]
                for i in self.candidates:
                    if i == j or i not in bounds:
                        continue
                    lcb_i = bounds[i][0]
                    if ucb_j < lcb_i:
                        to_remove.add(j)
                        break
            if to_remove:
                print("*", end="")
                self.candidates = [k for k in self.candidates if k not in to_remove]
                if not self.candidates:
                    return None
                if len(self.candidates) == 1:
                    only = self.candidates[0]
                    reward = float(self.env.get_reward(only))
                    self.counts[only] += 1
                    self.values[only] += (reward - self.values[only]) / self.counts[only]
                    return int(only)

            # If no elimination, proceed with standard UCB choice among candidates
            if total_counts <= 0:
                chosen_arm = int(self.rng.choice(self.candidates))
            else:
                best_score: Optional[float] = None
                chosen_arm = None
                for i in self.candidates:
                    # Avoid division by zero: if an arm slipped through with 0 count, prioritize it
                    if self.counts[i] == 0:
                        chosen_arm = i
                        break
                    bonus = math.sqrt(2.0 * math.log(total_counts) / self.counts[i])
                    score = self.values[i] + self.explore_reward * bonus
                    if best_score is None or score > best_score:
                        best_score = score
                        chosen_arm = i
                assert chosen_arm is not None

        reward = float(self.env.get_reward(chosen_arm))
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]
        return int(chosen_arm)

    def _dueling_ucb(self) -> Optional[Tuple[int, int]]:
        # Dueling UCB for dueling setting (candidates only)
        if not self.candidates:
            return None

        # Compute LCB for each ordered pair in current candidates
        def lcb(i: int, j: int) -> float:
            n_ij = self.dueling_counts[i, j]
            est = self.dueling_values[i, j]
            t = max(self.dueling_pulls, 1)
            denom = max(n_ij, 1)
            bonus = math.sqrt(2.0 * math.log(t) / denom)
            return est - self.explore_dueling * bonus

        lcb_values: Dict[Tuple[int, int], float] = {}
        for i in self.candidates:
            for j in self.candidates:
                if i != j:
                    lcb_values[(i, j)] = lcb(i, j)

        # Eliminate any j that is worse than some i with high confidence
        to_remove = set()
        for (i, j), l in lcb_values.items():
            if l > 0.5:
                to_remove.add(j)
        if to_remove:
            self.candidates = [k for k in self.candidates if k not in to_remove]

        # Fallbacks after elimination
        if not self.candidates:
            return None
        if len(self.candidates) == 1:
            only = self.candidates[0]
            # No pair to compare; skip dueling update
            return None

        # Select a pair to compare — prefer uncertain pairs (min LCB)
        feasible = [((i, j), lcb_values[(i, j)]) for i in self.candidates for j in self.candidates if i != j and (i, j) in lcb_values]
        if not feasible:
            return None
        (i, j), _ = min(feasible, key=lambda x: x[1])

        # Sample duel outcome and update dueling stats only
        outcome = float(self.env.get_dueling(i, j))  # 1.0 if i beats j, 0.0 if j beats i, 0.5 tie

        self.dueling_counts[i, j] += 1
        self.dueling_counts[j, i] += 1
        self.dueling_values[i, j] += (outcome - self.dueling_values[i, j]) / self.dueling_counts[i, j]
        self.dueling_values[j, i] += ((1.0 - outcome) - self.dueling_values[j, i]) / self.dueling_counts[j, i]

        return (i, j)
    
if __name__ == "__main__":
    # Quick sanity run for both branches
    env = Environment(10, seed=42)
    strategy = ProbabilisticStrategy(env, alpha=0.5, seed=123)
    for _ in range(10000):
        result = strategy.step()
        if _ % 1000 == 999:
            if isinstance(result, tuple):
                print(f"Duel compared: {result}")
            else:
                print(f"Pulled arm: {result}")
    print("Candidates left:", strategy.candidates)
    print("True means:", env.get_bandit_means())