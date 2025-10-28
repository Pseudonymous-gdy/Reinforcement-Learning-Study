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
    """A simple fusion of reward-UCB and dueling-UCB decisions (ELIMFUSION-style bounds).

    At each step:
    - With probability ``alpha``, select an arm using a reward-UCB rule.
    - With probability ``1 - alpha``, select a pair using a dueling-UCB heuristic,
      sample a duel outcome, then (in this simplified variant) DO NOT pull any arm
      for reward in this branch (kept consistent with your original interface/logic).

    Both sources update their own statistics. The step returns either a single arm
    index (reward branch) or a compared pair (dueling branch).

    Design notes (matched to "Fusing Reward and Dueling Bandits"):
    - Confidence radii use the unified form
        CR_R(k,t) = sqrt( 2 * log( K * t / delta ) / N_k )
        CR_D(i,j,t) = sqrt( 2 * log( K * t / delta ) / M_{i,j} )
      where K is the number of arms and t is a global time index advanced per step.
    - Reward-side elimination (paper-aligned):
        eliminate k if UCB_R(k) <= LCB_R(k_best),
        where k_best = argmax empirical mean on reward side.
    - Dueling-side elimination (paper-aligned):
        eliminate k if there exists ℓ with UCB_D(k,ℓ) < 0.5.

    IMPORTANT:
    - Interface preserved: constructor signature unchanged; return types of step() unchanged.
    - We keep your probabilistic branch selection and history bookkeeping.
    """

    def __init__(self, env: Environment, alpha: float = 0.5, seed: Optional[int] = None):
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self.env = env
        self.alpha = alpha  # Probability to choose reward-UCB branch

        # Internal RNG for strategy decisions (separate from env RNG)
        self.rng = np.random.default_rng(seed)

        # ---------- Reward (UCB) state ----------
        self.counts: List[int] = [0] * env.number_of_bandits
        self.values: List[float] = [0.0] * env.number_of_bandits
        self.explore_reward: float = 0.5  # multiplier for reward confidence radius

        # ---------- Dueling (pairwise) state ----------
        n = env.number_of_bandits
        self.dueling_counts: np.ndarray = np.zeros((n, n), dtype=int)
        self.dueling_values: np.ndarray = np.zeros((n, n), dtype=float)  # v[i,j] ~ P(i beats j)
        self.explore_dueling: float = 0.5  # multiplier for duel confidence radius

        # ---------- Candidate set & history ----------
        self.candidates: List[int] = list(range(env.number_of_bandits))
        # History: (is_reward_step, pulled_arm, compared_pair)
        self.history: List[Tuple[bool, Optional[int], Optional[Tuple[int, int]]]] = []
        self.reward_pulls: int = 0
        self.dueling_pulls: int = 0

        # ---------- Global "time" and confidence parameter ----------
        # Global time index t used in log(K * t / delta). We advance it once per step().
        self.t: int = 0
        self._delta: float = 0.05  # confidence level (kept internal to preserve the constructor interface)
        self.K: int = env.number_of_bandits

    # =========================
    # Confidence radius helpers
    # =========================
    def _log_arg(self) -> float:
        """Argument inside the logarithm: K * t / delta, made numerically safe."""
        # Ensure strictly positive argument; t is at least 1 once the first step completes.
        K = max(1, self.K)
        t = max(1, self.t)
        delta = max(self._delta, 1e-12)
        return max((K * t) / delta, 1.0)

    def _CR_R(self, k: int) -> float:
        """Reward-side confidence radius for arm k."""
        n_k = max(1, self.counts[k])
        return math.sqrt(2.0 * math.log(self._log_arg()) / n_k)

    def _CR_D(self, i: int, j: int) -> float:
        """Dueling-side confidence radius for ordered pair (i, j)."""
        m_ij = max(1, int(self.dueling_counts[i, j]))
        return math.sqrt(2.0 * math.log(self._log_arg()) / m_ij)

    # ========
    # One step
    # ========
    def step(self) -> Optional[Union[int, Tuple[int, int]]]:
        """Execute one decision step. Returns arm index (reward branch) or pair (dueling branch)."""
        if not self.candidates:
            return None

        prob = self.rng.random()
        if prob < self.alpha:
            # Reward setting
            action = self._reward_ucb()
            if action is None:
                return None
            self.reward_pulls += 1
            self.history.append((True, action, None))
            self.t += 1  # advance global time once per step
            return action
        else:
            # Dueling setting
            compared_pair = self._dueling_ucb()
            if compared_pair is None:
                return None
            self.dueling_pulls += 1
            self.history.append((False, None, compared_pair))
            self.t += 1  # advance global time once per step
            return compared_pair

    # ==================
    # Reward-side policy
    # ==================
    def _reward_ucb(self) -> Optional[int]:
        """Reward branch: ELIMFUSION-style bounds for elimination and UCB scoring."""
        if not self.candidates:
            return None

        # --- (A) Gentle under-sampling guard (kept from your original design) ---
        # Drive all candidates to reach at least min_pulls_elim pulls so that bounds shrink and
        # elimination can actually trigger with some confidence.
        total_counts = sum(self.counts[i] for i in self.candidates)
        t_local = max(total_counts, 1)
        min_pulls_elim = max(5, int(2 * math.log(t_local + 1)))
        under_sampled = [i for i in self.candidates if self.counts[i] < min_pulls_elim]
        if under_sampled:
            chosen_arm = int(self.rng.choice(under_sampled))
            reward = float(self.env.get_reward(chosen_arm))
            self.counts[chosen_arm] += 1
            self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]
            return int(chosen_arm)

        # --- (B) Paper-aligned elimination: eliminate k if UCB_R(k) <= LCB_R(k_best) ---
        # Build bounds for all candidates (using global-time CR).
        bounds: Dict[int, Tuple[float, float]] = {}
        for i in self.candidates:
            cr = self.explore_reward * self._CR_R(i)
            lcb = self.values[i] - cr
            ucb = self.values[i] + cr
            bounds[i] = (lcb, ucb)

        # Best arm by empirical mean (ties broken arbitrarily by max)
        k_best = max(self.candidates, key=lambda idx: self.values[idx])
        lcb_best = bounds[k_best][0]

        to_remove = set()
        for k in self.candidates:
            ucb_k = bounds[k][1]
            if ucb_k <= lcb_best:
                to_remove.add(k)

        if to_remove:
            # Keep a visual hint (consistent with your print)
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

        # --- (C) UCB arm selection among remaining candidates (paper-style CR) ---
        best_score: Optional[float] = None
        chosen_arm: Optional[int] = None
        for i in self.candidates:
            if self.counts[i] == 0:
                chosen_arm = i
                break
            score = self.values[i] + self.explore_reward * self._CR_R(i)
            if best_score is None or score > best_score:
                best_score = score
                chosen_arm = i
        assert chosen_arm is not None

        # Pull the chosen arm and update reward stats
        reward = float(self.env.get_reward(chosen_arm))
        self.counts[chosen_arm] += 1
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / self.counts[chosen_arm]
        return int(chosen_arm)

    # =================
    # Dueling-side step
    # =================
    def _dueling_ucb(self) -> Optional[Tuple[int, int]]:
        """Dueling branch: ELIMFUSION-style elimination and pair selection."""
        if not self.candidates:
            return None
        if len(self.candidates) == 1:
            # No opponent to duel; keep behavior consistent with your original code
            return None

        # --- (A) Paper-aligned elimination on dueling side ---
        # Eliminate k if ∃ℓ s.t. UCB_D(k,ℓ) < 0.5
        to_remove = set()
        for k in self.candidates:
            # check against any other candidate ℓ
            for ell in self.candidates:
                if ell == k:
                    continue
                # If we have zero observations for (k,ell), UCB will be wide; skip elimination in that case.
                cr = self.explore_dueling * self._CR_D(k, ell)
                ucb_k_ell = self.dueling_values[k, ell] + cr
                if ucb_k_ell < 0.5:
                    to_remove.add(k)
                    break

        if to_remove:
            self.candidates = [x for x in self.candidates if x not in to_remove]
            if not self.candidates:
                return None
            if len(self.candidates) == 1:
                return None  # remain consistent: do not create a duel with a single candidate

        # --- (B) Pair selection: prefer uncertain pairs by minimum LCB_D(i,j) ---
        def lcb(i: int, j: int) -> float:
            cr = self.explore_dueling * self._CR_D(i, j)
            return self.dueling_values[i, j] - cr

        lcb_values: Dict[Tuple[int, int], float] = {}
        for i in self.candidates:
            for j in self.candidates:
                if i != j:
                    lcb_values[(i, j)] = lcb(i, j)

        feasible = [((i, j), lcb_values[(i, j)])
                    for i in self.candidates for j in self.candidates
                    if i != j and (i, j) in lcb_values]
        if not feasible:
            return None

        (i, j), _ = min(feasible, key=lambda x: x[1])

        # --- (C) Sample duel outcome and update pairwise stats only ---
        # outcome = 1.0 if i beats j, 0.0 if j beats i, 0.5 tie
        outcome = float(self.env.get_dueling(i, j))

        self.dueling_counts[i, j] += 1
        self.dueling_counts[j, i] += 1
        self.dueling_values[i, j] += (outcome - self.dueling_values[i, j]) / self.dueling_counts[i, j]
        self.dueling_values[j, i] += ((1.0 - outcome) - self.dueling_values[j, i]) / self.dueling_counts[j, i]

        return (i, j)


if __name__ == "__main__":
    # Quick sanity run for both branches
    env = Environment(5, seed=42)
    strategy = ProbabilisticStrategy(env, alpha=0.5, seed=123)
    for t in range(10000):
        result = strategy.step()
        # progress reporting every 1000 steps
        if t % 1000 == 999:
            if isinstance(result, tuple):
                print(f"\nDuel compared: {result}")
            else:
                print(f"\nPulled arm: {result}")
    print("\nCandidates left:", strategy.candidates)
    print("True means:", env.get_bandit_means())
