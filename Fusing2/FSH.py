# fsh_duel_then_reward_no_cr_pair_balanced_strict_v3.py
# ─────────────────────────────────────────────────────────────────────────────
# FusionSequentialHalving-DuelThenReward (no-CR)
# (pair-balanced duels; proof-aligned) — STRICT to the UPDATED pseudocode.
#
# Key points (exactly as your updated pseudocode):
#   - R = ceil(log2 n0)
#   - T_D = (1-zeta)*T , T_R = zeta*T  (NO rounding at budget split)
#   - Per-round budgets: T_{D,r} = T_D/R , T_{R,r} = T_R/R (REAL)
#   - Ceil is used ONLY when converting to integer sampling counts:
#       m_r   = ceil( 2*T_{D,r} / (K_r(K_r-1)) )
#       t_R,r = ceil( T_{R,r} / K_r^(D) )
#       keep_D = ceil((1-alpha)K_r), keep_R = ceil(K_r/2)
#   - Pair-balanced duels: only loop i<j; no (i,j) vs (j,i) duplication.
#   - Reward elimination sorts ONLY by μ̂_{i,r} (per-round mean), tie-break by index.
#   - No use of previous rounds' s_hat as secondary key anywhere.
#
# Extremes:
#   - allow zeta in [0,1], alpha in [0,1/2] for testing endpoints
#   - if m_r = 0 (e.g., zeta=1 => T_D=0), then W/m_r is undefined in pseudocode.
#     Minimal convention used ONLY for this undefined case:
#       set ν̂_{i,j}=1/2 for all i≠j  => all s_hat equal => dueling filter reduces to tie-break (index).
#
# Budget policy:
#   - No hard cap at T (as requested). spent_budget may exceed T slightly due to ceil() on m_r and t_R,r.
#
# Environment interface expected:
#   - env.get_reward(i) -> float
#   - env.duel(i, j) -> winner index i or j
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import sys
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Allow running this file directly
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Fusing2.env import Environment


def _topk_by_score(items: List[int], score: Dict[int, float], k: int) -> List[int]:
    """Deterministic: score desc, index asc."""
    if not items:
        return []
    k = max(1, min(int(k), len(items)))
    return sorted([int(x) for x in items], key=lambda a: (-float(score[int(a)]), int(a)))[:k]


@dataclass
class RoundTrace:
    r: int
    K_r: int
    T_D_r: float
    T_R_r: float
    m_r: int
    duels_executed: int
    keep_D: int
    K_D: int
    t_R_r: int
    pulls_executed: int
    keep_R: int
    C_pre: List[int]
    C_tilde: List[int]
    C_post: List[int]
    note: str = ""


class FusionSequentialHalving_Final:
    def __init__(
        self,
        env: Environment,
        total_budget: int,
        zeta: float,
        alpha: float,
        seed: Optional[int] = None,
        trace_enabled: bool = False,
    ):
        self.env = env
        self.n0 = int(env.number_of_bandits)

        self.T = int(total_budget)
        self.zeta = float(zeta)
        self.alpha = float(alpha)

        assert self.T >= 0, "total_budget must be nonnegative"
        # endpoints allowed to test extremes
        assert 0.0 <= self.zeta <= 1.0, "zeta must be in [0,1]"
        assert 0.0 <= self.alpha <= 0.5, "alpha must be in [0,1/2]"

        self.rng = np.random.default_rng(seed)

        self.trace_enabled = bool(trace_enabled)
        self.round_trace: List[RoundTrace] = []
        self.history: List[Tuple[Any, ...]] = []

        self.candidate_set: List[int] = list(range(self.n0))
        self.spent_budget: int = 0

        # Pseudocode: R <- ceil(log2 n0)
        self.R: int = 1 if self.n0 <= 1 else int(math.ceil(math.log2(self.n0)))

        # Updated pseudocode: T_D=(1-zeta)T, T_R=zeta T
        self.T_D: float = (1.0 - self.zeta) * float(self.T)
        self.T_R: float = self.zeta * float(self.T)

        # Pseudocode beta (not used operationally; included for completeness)
        self.beta: float = 1.0 - 1.0 / (2.0 * (1.0 - self.alpha)) if self.alpha < 1.0 else float("nan")

    def run(self) -> int:
        # reset run-state
        self.history = []
        self.round_trace = []
        self.candidate_set = list(range(self.n0))
        self.spent_budget = 0

        if self.n0 <= 1:
            return 0 if self.n0 == 1 else -1

        C_r: List[int] = list(self.candidate_set)

        for r in range(1, self.R + 1):
            if len(C_r) <= 1:
                break

            C_pre = list(map(int, C_r))
            K_r = len(C_r)

            # Updated pseudocode: per-round budgets are REAL
            T_D_r: float = self.T_D / float(self.R)
            T_R_r: float = self.T_R / float(self.R)

            # ============================================================
            # 1) Dueling stage (pair-balanced; only i<j pairs are dueled)
            # ============================================================
            # m_r = ceil(2*T_{D,r} / (K_r(K_r-1)))
            if K_r <= 1 or T_D_r <= 0.0:
                m_r = 0
            else:
                m_r = int(math.ceil((2.0 * T_D_r) / (float(K_r) * float(K_r - 1))))

            # Directed win counts W^{(r)}_{i,j} for i≠j (even though we only DUEL i<j)
            W_r: Dict[int, Dict[int, int]] = {}
            for i in C_r:
                ii = int(i)
                W_r[ii] = {int(j): 0 for j in C_r if int(j) != ii}

            duels_executed = 0
            arms_sorted = sorted([int(x) for x in C_r])

            if m_r > 0 and K_r >= 2:
                for p in range(len(arms_sorted)):
                    i = arms_sorted[p]
                    for q in range(p + 1, len(arms_sorted)):
                        j = arms_sorted[q]  # i<j
                        for _ in range(m_r):
                            winner = int(self.env.duel(i, j))
                            if winner not in (i, j):
                                raise ValueError("env.duel(i,j) must return i or j")
                            self.history.append(("D", i, j, winner))
                            if winner == i:
                                W_r[i][j] += 1
                            else:
                                W_r[j][i] += 1
                            duels_executed += 1
                            self.spent_budget += 1

            # 1.1) Empirical Borda scores
            # nu_hat_{i,j} = W_{i,j}/m_r, s_hat_{i,r} = (1/K_r)(1/2 + sum_{j!=i} nu_hat_{i,j})
            s_hat: Dict[int, float] = {}
            note = ""
            if m_r == 0:
                # undefined division in pseudocode; minimal convention
                # nu_hat_{i,j}=1/2 for all i≠j => all s_hat=1/2
                for i in C_r:
                    s_hat[int(i)] = 0.5
                note = "m_r=0 => set nu_hat(i,j)=1/2 for all i≠j (undefined case only)"
            else:
                inv_m = 1.0 / float(m_r)
                for i in C_r:
                    ii = int(i)
                    S_i = 0.0
                    for j in C_r:
                        jj = int(j)
                        if jj == ii:
                            continue
                        S_i += float(W_r[ii][jj]) * inv_m
                    s_hat[ii] = (0.5 + S_i) / float(K_r)

            # 1.2) Keep top ceil((1-alpha)K_r) by s_hat
            keep_D = int(math.ceil((1.0 - self.alpha) * float(K_r)))
            keep_D = max(1, min(keep_D, K_r))
            C_tilde = _topk_by_score(C_r, s_hat, keep_D)

            if len(C_tilde) <= 1:
                C_r = list(map(int, C_tilde))
                self.candidate_set = list(C_r)
                if self.trace_enabled:
                    self.round_trace.append(
                        RoundTrace(
                            r=r,
                            K_r=K_r,
                            T_D_r=float(T_D_r),
                            T_R_r=float(T_R_r),
                            m_r=int(m_r),
                            duels_executed=int(duels_executed),
                            keep_D=int(keep_D),
                            K_D=len(C_tilde),
                            t_R_r=0,
                            pulls_executed=0,
                            keep_R=0,
                            C_pre=C_pre,
                            C_tilde=list(map(int, C_tilde)),
                            C_post=list(map(int, C_r)),
                            note="break: |C_tilde|<=1" + (f" ({note})" if note else ""),
                        )
                    )
                break

            # ============================================================
            # 2) Reward stage on C_tilde (balanced pulls)
            # ============================================================
            K_D = len(C_tilde)
            # t_{R,r} = ceil(T_{R,r}/K_D)
            t_R_r = int(math.ceil(T_R_r / float(K_D))) if (T_R_r > 0.0 and K_D > 0) else 0

            mu_hat_round: Dict[int, float] = {int(i): 0.0 for i in C_tilde}
            n_round: Dict[int, int] = {int(i): 0 for i in C_tilde}

            pulls_executed = 0
            for i in C_tilde:
                ii = int(i)
                for _ in range(t_R_r):
                    x = float(self.env.get_reward(ii))
                    self.history.append(("R", ii, x))
                    n_round[ii] += 1
                    mu_hat_round[ii] += (x - mu_hat_round[ii]) / float(n_round[ii])
                    pulls_executed += 1
                    self.spent_budget += 1

            # 2.1) Keep top ceil(K_r/2) arms by mu_hat_round (ONLY mu_hat; tie-break index)
            keep_R = int(math.ceil(float(K_r) / 2.0))
            keep_R = max(1, min(keep_R, len(C_tilde)))

            C_sorted = sorted([int(x) for x in C_tilde], key=lambda a: (-float(mu_hat_round[a]), int(a)))
            C_next = C_sorted[:keep_R]

            C_r = list(map(int, C_next))
            self.candidate_set = list(C_r)

            if self.trace_enabled:
                self.round_trace.append(
                    RoundTrace(
                        r=r,
                        K_r=K_r,
                        T_D_r=float(T_D_r),
                        T_R_r=float(T_R_r),
                        m_r=int(m_r),
                        duels_executed=int(duels_executed),
                        keep_D=int(keep_D),
                        K_D=int(K_D),
                        t_R_r=int(t_R_r),
                        pulls_executed=int(pulls_executed),
                        keep_R=int(keep_R),
                        C_pre=C_pre,
                        C_tilde=list(map(int, C_tilde)),
                        C_post=list(map(int, C_r)),
                        note=note,
                    )
                )

        # Return remaining arm (ties ignored; deterministic by index)
        if len(self.candidate_set) == 0:
            return -1
        if len(self.candidate_set) == 1:
            return int(self.candidate_set[0])
        return int(min(self.candidate_set))

    def get_candidate_set(self) -> List[int]:
        return list(map(int, self.candidate_set))

    def get_history(self) -> List[Tuple[Any, ...]]:
        return list(self.history)

    def get_spent_budget(self) -> int:
        return int(self.spent_budget)


# ─────────────────────────────────────────────────────────────────────────────
# Optional smoke test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    K = 17
    T = 5000
    env = Environment(number_of_bandits=K, distribution="bernoulli", seed=12345)

    tests = [
        dict(zeta=0.0, alpha=0.0),
        dict(zeta=0.0, alpha=0.5),
        dict(zeta=1.0, alpha=0.0),
        dict(zeta=1.0, alpha=0.5),
        dict(zeta=0.1, alpha=0.45),
        dict(zeta=0.8, alpha=0.2),
    ]

    for cfg in tests:
        algo = FusionSequentialHalving_Final(
            env=env,
            total_budget=T,
            zeta=cfg["zeta"],
            alpha=cfg["alpha"],
            seed=6789,
            trace_enabled=True,
        )
        out = algo.run()
        print("\ncfg =", cfg)
        print("estimated best arm =", out)
        print("true best arm      =", env.get_optimal_action())
        print("spent_budget       =", algo.get_spent_budget())
        print("final C            =", algo.get_candidate_set())
        print("rounds traced      =", len(algo.round_trace))
