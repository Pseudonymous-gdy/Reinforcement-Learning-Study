r"""
Fusing2.combined_sequential_halving

CombinedSequentialHalving — sums Borda score and Reward Mean, then eliminates in one step.

Key logic:
- In each round, budget T_r is split into Duel budget (nD) and Reward budget (nR) based on alpha.
    nD = floor(T_r * (1 - alpha))
    nR = T_r - nD
- Perform duels among current candidates C to update Borda scores.
- Perform reward sampling among current candidates C to update Empirical Means (mu_hat).
- Compute Combined Score: Score(i) = Borda(i) + mu_hat(i).
    - Borda(i) is in [0, 1].
    - mu_hat(i) is in [0, 1] (assuming Bernoulli/bounded rewards).
- Rank candidates by Combined Score and keep top fraction (episode_keep, default 0.5).

"""

import os
import sys
import math
import json
import platform
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional, Any, Union, Dict

import numpy as np

# Allow running this file directly
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Fusing2.env import Environment


def _to_py(x: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-serializable python types."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _clamp_keep_count_floor(ratio: float, n: int) -> int:
    """Keep count based on floor(ratio*n) but clamped to [1,n]."""
    if n <= 1:
        return 1
    if ratio <= 0.0:
        return 1
    if ratio >= 1.0:
        return n
    return max(1, min(n, int(math.floor(ratio * n))))


def _topk_with_tiebreak(items: List[int], score: Dict[int, float], k: int) -> List[int]:
    """
    Deterministic: score desc, index asc.
    """
    k = max(1, min(int(k), len(items)))
    return sorted([int(x) for x in items], key=lambda a: (-float(score[int(a)]), int(a)))[:k]


class CombinedSequentialHalving:
    def __init__(
        self,
        env: Environment,
        total_budget: int,
        alpha: float = 0.5,
        episode_keep: Union[float, np.float64] = 0.5,
        seed: Optional[int] = None,
        num_rounds: Optional[int] = None,
        trace_enabled: bool = False,
        c1: Optional[float] = None,
        c2: Optional[float] = None
    ):
        self.env = env
        self.K = int(env.number_of_bandits)

        self.total_budget = int(total_budget)
        self.alpha = float(alpha)
        self.episode_keep = float(episode_keep)

        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0,1]"
        assert 0.0 <= self.episode_keep <= 1.0, "episode_keep must be in [0,1]"
        assert self.total_budget >= 0, "total_budget must be nonnegative"

        self.rng = np.random.default_rng(seed)

        self.candidate_set: List[int] = list(range(self.K))
        self.history: List[Tuple[Any, ...]] = []

        self.trace_enabled = bool(trace_enabled)
        self.round_trace: List[Dict[str, Any]] = []

        self.spent_budget: int = 0
        self.round: int = 0

        if num_rounds is None:
            self.num_rounds = 1 if self.K <= 1 else int(math.ceil(math.log2(self.K)))
        else:
            self.num_rounds = max(1, int(num_rounds))

        self.budget_per_round = (
            int(math.ceil(self.total_budget / self.num_rounds)) if self.num_rounds > 0 else self.total_budget
        )

        # assignment on c_1 and c_2:
        if c1 is None and c2 is None:
            self.c1 = (1-alpha) / (1+alpha+1e-8)
            self.c2 = 1 - self.c1
        elif c1 is not None and c2 is not None:
            self.c1 = float(c1)
            self.c2 = float(c2)
        else:
            raise ValueError("Both c1 and c2 must be provided together, or both left as None for default 0.5 each.")

        # Reward stats
        self.N = np.zeros(self.K, dtype=np.int64)
        self.reward_sum = np.zeros(self.K, dtype=np.float64)
        self.mu_hat = np.zeros(self.K, dtype=np.float64)

        # Dueling stats
        self.M = np.zeros((self.K, self.K), dtype=np.int64)
        self.W = np.zeros((self.K, self.K), dtype=np.int64)
        # Initialize nu_hat to 0.5 (unknown/tie)
        self.nu_hat = np.full((self.K, self.K), 0.5, dtype=np.float64)
        np.fill_diagonal(self.nu_hat, 0.5)

    # -----------------------
    # Dueling interface bridge
    # -----------------------
    def _duel_winner(self, i: int, j: int) -> int:
        if hasattr(self.env, "duel"):
            w = int(self.env.duel(i, j))
            if w not in (i, j):
                raise ValueError(f"env.duel(i,j) must return i or j; got {w}")
            return w

        if hasattr(self.env, "get_dueling"):
            out = float(self.env.get_dueling(i, j))
            if out == 1.0:
                return int(i)
            if out == 0.0:
                return int(j)
            return int(self.rng.choice([i, j]))

        ri = float(self.env.get_reward(i))
        rj = float(self.env.get_reward(j))
        return int(i if ri >= rj else j)

    def _borda_scores(self, C: List[int]) -> Dict[int, float]:
        n = len(C)
        if n <= 1:
            return {int(C[0]): 0.5} if n == 1 else {}
        denom = float(n - 1)
        out: Dict[int, float] = {}
        for i in C:
            s = 0.0
            for j in C:
                if j == i:
                    continue
                s += float(self.nu_hat[i, j])
            out[int(i)] = float(s / denom)
        return out

    def _uniform_reward_allocate(self, C: List[int], nR: int) -> List[int]:
        if nR <= 0 or len(C) == 0:
            return []
        m = len(C)
        base = nR // m
        rem = nR % m
        schedule: List[int] = []
        for a in C:
            schedule.extend([int(a)] * base)
        if rem > 0:
            # Distribute remainder randomly
            extra = self.rng.choice(C, size=rem, replace=False)
            schedule.extend([int(x) for x in extra.tolist()])
        return schedule

    # -----------------------
    # Core: one round
    # -----------------------
    def run_round(self, T_r: Optional[int] = None) -> List[int]:
        C = list(self.candidate_set)
        n_r = len(C)

        if n_r <= 1:
            return self.candidate_set

        if T_r is None:
            remaining = self.total_budget - self.spent_budget
            T_r = min(self.budget_per_round, max(0, remaining))
        T_r = int(T_r)

        if T_r <= 0:
            return self.candidate_set

        self.round += 1

        # Budget Split
        # alpha controls proportion of budget for Rewards (or Duels?)
        # Convention in FusionSequentialHalving was: nD = floor(T_r * (1 - alpha))
        # If alpha=1 -> nD=0 (Reward only)
        # If alpha=0 -> nD=T_r (Duel only)
        # NOTE: User did not specify direction of alpha, assuming consistency with Fusion params makes sense.
        nD = int(math.floor(T_r * (1.0 - self.alpha)))
        nR = int(T_r - nD)

        round_log: Dict[str, Any] = {}
        if self.trace_enabled:
            round_log = {
                "round": int(self.round),
                "T_r": int(T_r),
                "n_r": int(n_r),
                "candidate_pre": list(map(int, C)),
                "nD": int(nD),
                "nR": int(nR),
            }

        # -------------------
        # 1) Dueling Phase
        # -------------------
        if nD > 0 and n_r >= 2:
            for _ in range(nD):
                ii = int(self.rng.integers(0, n_r))
                jj = int(self.rng.integers(0, n_r - 1))
                if jj >= ii:
                    jj += 1
                i = int(C[ii])
                j = int(C[jj])

                winner = self._duel_winner(i, j)
                self.history.append(("D", i, j, int(winner)))

                self.M[i, j] += 1
                self.M[j, i] += 1
                if winner == i:
                    self.W[i, j] += 1
                else:
                    self.W[j, i] += 1
                
                self.nu_hat[i, j] = self.W[i, j] / self.M[i, j]
                self.nu_hat[j, i] = self.W[j, i] / self.M[j, i]

        # -------------------
        # 2) Reward Phase
        # -------------------
        if nR > 0:
            schedule = self._uniform_reward_allocate(C, nR)
            for a in schedule:
                r = float(self.env.get_reward(int(a)))
                self.history.append(("R", int(a), float(r)))
                self.N[a] += 1
                self.reward_sum[a] += r
                self.mu_hat[a] = self.reward_sum[a] / self.N[a]

        # -------------------
        # 3) Combine Scores & Elimination
        # -------------------
        borda = self._borda_scores(C)
        
        # Combined Score = Borda + mu_hat
        combined_scores: Dict[int, float] = {}
        for a in C:
            # If no info, default to 0.5 for borda, 0.0 for mu_hat (or maybe neutral 0.5?)
            # Usually mu starts at 0. Let's use current mu_hat value.
            # Borda defaults to 0.5 if no duels/comparisons in _borda_scores for isolated nodes? 
            # Actually _borda_scores uses nu_hat which defaults to 0.5.
            s_borda = float(borda.get(a, 0.5))
            s_reward = float(self.mu_hat[a])
            combined_scores[a] = self.c1 * s_borda + self.c2 * s_reward

        # Decide keep count
        m_next = _clamp_keep_count_floor(self.episode_keep, n_r)

        # Rank by combined score
        C_next = _topk_with_tiebreak(C, combined_scores, m_next)
        
        self.candidate_set = C_next
        self.spent_budget += T_r

        if self.trace_enabled:
            round_log["m_next"] = int(m_next)
            round_log["scores"] = {str(k): round(combined_scores[k], 4) for k in C}
            round_log["candidate_post"] = list(map(int, self.candidate_set))
            self.round_trace.append(round_log)

        return self.candidate_set

    def _finalize_if_needed(self) -> None:
        """
        If budget ends with multiple candidates, pick a single best arm.
        """
        C = list(self.candidate_set)
        if len(C) <= 1:
            return

        borda = self._borda_scores(C)
        
        combined_scores: Dict[int, float] = {}
        for a in C:
            combined_scores[a] = self.c1 * float(borda.get(a, 0.5)) + self.c2 * float(self.mu_hat[a])
        
        best = _topk_with_tiebreak(C, combined_scores, 1)[0]
        self.candidate_set = [int(best)]
        
        if self.trace_enabled:
            self.round_trace.append({
                "round": int(self.round),
                "mode": "finalize",
                "candidate_pre": C,
                "candidate_post": self.candidate_set
            })

    def run(self) -> List[Tuple[Any, ...]]:
        self.candidate_set = list(range(self.K))
        self.history = []
        self.round_trace = []
        self.round = 0
        self.spent_budget = 0
        self.N[:] = 0
        self.reward_sum[:] = 0.0
        self.mu_hat[:] = 0.0
        self.M[:, :] = 0
        self.W[:, :] = 0
        self.nu_hat[:, :] = 0.5
        np.fill_diagonal(self.nu_hat, 0.5)

        while self.spent_budget < self.total_budget and len(self.candidate_set) > 1:
            self.run_round()

        self._finalize_if_needed()
        return self.history

    # Getters
    def get_candidate_set(self) -> List[int]:
        return list(self.candidate_set)

    def get_history(self) -> List[Tuple[Any, ...]]:
        return list(self.history)

    def get_round(self) -> int:
        return int(self.round)
    
    def get_round_tot(self) -> int:
        return int(self.spent_budget)

# ============================================================
# Unit Test
# ============================================================
if __name__ == "__main__":
    TEST_ID = "CombinedSH_Test"
    TOTAL_BUDGET = 2048
    ENV_SEED = 12345
    ALGO_SEED = 67890

    # Example fixed means
    fixed_means = [
        0.68, 0.80, 0.66, 0.60, 0.72, 0.46, 0.40, 0.30,
        0.20, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01, 0.00,
        0.89, 0.78, 0.58, 0.48, 0.38, 0.28, 0.18, 0.14,
        0.37, 0.26, 0.19, 0.54, 0.90, 0.25, 0.64, 0.915
    ]
    K = len(fixed_means)

    dueling_means = np.zeros((K, K), dtype=float)
    for a in range(K):
        for b in range(K):
            if a == b:
                dueling_means[a, b] = 0.5
            else:
                p = 0.5 + (fixed_means[a] - fixed_means[b]) / 2.0
                dueling_means[a, b] = float(min(1.0, max(0.0, p)))

    env = Environment(
        number_of_bandits=K,
        distribution="bernoulli",
        seed=ENV_SEED,
        dueling_means=dueling_means,
    )
    if hasattr(env, "bandit_means"):
        env.bandit_means = np.array(fixed_means, dtype=float)
        env.optimal_mean = float(np.max(env.bandit_means))

    algo = CombinedSequentialHalving(
        env=env,
        total_budget=TOTAL_BUDGET,
        alpha=0.8,
        episode_keep=0.5,
        seed=ALGO_SEED,
        trace_enabled=True,
        # c1=1,
        # c2=1
    )
    history = algo.run()
    final_candidates = algo.get_candidate_set()

    print(f"Spent Budget: {algo.spent_budget}")
    print(f"Final Candidates: {final_candidates}")
    print(f"Best Arm: {np.argmax(fixed_means)}")
    if len(final_candidates) == 1 and final_candidates[0] == np.argmax(fixed_means):
        print("✅ Correct arm indentified.")
    else:
        print("⚠️ Failed to identify best arm.")
