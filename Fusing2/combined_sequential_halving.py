r"""
Fusing2.combined_sequential_halving

CombinedSequentialHalving — sums Borda score and Reward Mean, then eliminates in
one step. EXACTLY ALIGNED WITH FSH LOGIC (per-round means, ceil keeping, ceil budget).
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


def _to_py(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def _topk_with_tiebreak(items: List[int], score: Dict[int, float], k: int) -> List[int]:
    if not items: return []
    k = max(1, min(int(k), len(items)))
    return sorted([int(x) for x in items], key=lambda a: (-float(score[int(a)]), int(a)))[:k]

class CombinedSequentialHalving:
    def __init__(
        self,
        env,
        total_budget: int,
        alpha: float = 0.5,
        episode_keep: float = 0.5,
        seed: Optional[int] = None,
        num_rounds: Optional[int] = None,
        trace_enabled: bool = False,
        c1: Optional[float] = None,
        c2: Optional[float] = None
    ):
        self.env = env
        self.total_budget = int(total_budget)
        self.alpha = float(alpha)
        self.episode_keep = float(episode_keep)
        self.K = int(env.number_of_bandits)
        self.rng = np.random.default_rng(seed)
        self.trace_enabled = bool(trace_enabled)
        
        if c1 is None and c2 is None:
            self.c1 = (1.0 - self.alpha) / (1.0 + self.alpha + 1e-8)
            self.c2 = 1.0 - self.c1
        elif c1 is not None and c2 is not None:
            self.c1 = float(c1)
            self.c2 = float(c2)
        else:
            raise ValueError("Both c1 and c2 must be provided together.")

        self.candidate_set: List[int] = list(range(self.K))
        self.history: List[Tuple[Any, ...]] = []
        self.round_trace: List[Dict[str, Any]] = []

        self.spent_budget: int = 0
        self.round: int = 0

        self.R = 1 if self.K <= 1 else int(math.ceil(math.log2(self.K)))
        self.num_rounds = self.R if num_rounds is None else max(1, int(num_rounds))

        # Budgets logically exactly like FSH
        self.T_D: float = (1.0 - self.alpha) * float(self.total_budget)
        self.T_R: float = self.alpha * float(self.total_budget)

    def run(self) -> List[Tuple[Any, ...]]:
        self.candidate_set = list(range(self.K))
        self.history = []
        self.round_trace = []
        self.round = 0
        self.spent_budget = 0

        C_r: List[int] = list(self.candidate_set)

        for r in range(1, self.num_rounds + 1):
            if len(C_r) <= 1:
                break

            C_pre = list(map(int, C_r))
            n_r = len(C_r)

            T_D_r: float = self.T_D / float(self.num_rounds)
            T_R_r: float = self.T_R / float(self.num_rounds)

            # 1) Duels
            if n_r <= 1 or T_D_r <= 0.0:
                m_r = 0
            else:
                m_r = int(math.ceil((2.0 * T_D_r) / (float(n_r) * float(n_r - 1))))
                
            W_r: Dict[int, Dict[int, int]] = {int(i): {int(j): 0 for j in C_r if int(j) != int(i)} for i in C_r}
            M_r: Dict[int, Dict[int, int]] = {int(i): {int(j): 0 for j in C_r if int(j) != int(i)} for i in C_r}

            duels_executed = 0
            arms_sorted = sorted([int(x) for x in C_r])
            for i_idx in range(len(arms_sorted)):
                for j_idx in range(i_idx + 1, len(arms_sorted)):
                    a_i = arms_sorted[i_idx]
                    a_j = arms_sorted[j_idx]
                    for _ in range(m_r):
                        winner = int(self.env.duel(a_i, a_j))
                        self.history.append(("D", a_i, a_j, winner))
                        duels_executed += 1
                        self.spent_budget += 1
                        M_r[a_i][a_j] += 1
                        M_r[a_j][a_i] += 1
                        if winner == a_i:
                            W_r[a_i][a_j] += 1
                        else:
                            W_r[a_j][a_i] += 1

            s_borda: Dict[int, float] = {}
            for i in C_r:
                ii = int(i)
                if n_r <= 1:
                    s_borda[ii] = 0.5
                else:
                    su = 0.0
                    for j in C_r:
                        jj = int(j)
                        if jj == ii: continue
                        if M_r[ii][jj] > 0:
                            su += float(W_r[ii][jj]) / float(M_r[ii][jj])
                        else:
                            su += 0.5
                    s_borda[ii] = su / float(n_r - 1)

            # 2) Rewards
            t_R_r = int(math.ceil(T_R_r / float(n_r))) if (T_R_r > 0.0 and n_r > 0) else 0
                
            mu_hat_round: Dict[int, float] = {int(i): 0.0 for i in C_r}
            pulls_executed = 0
            for i in C_r:
                ii = int(i)
                sum_r = 0.0
                for _ in range(t_R_r):
                    x = float(self.env.get_reward(ii))
                    self.history.append(("R", ii, x))
                    sum_r += x
                    pulls_executed += 1
                    self.spent_budget += 1
                if t_R_r > 0:
                    mu_hat_round[ii] = sum_r / float(t_R_r)
                else:
                    mu_hat_round[ii] = 0.0

            # 3) Combine Scores
            combined_scores: Dict[int, float] = {}
            for i in C_r:
                ii = int(i)
                combined_scores[ii] = self.c1 * s_borda[ii] + self.c2 * mu_hat_round[ii]

            m_next = int(math.ceil(self.episode_keep * float(n_r)))
            if m_next < 1: m_next = 1
            if m_next > n_r: m_next = n_r

            C_r = _topk_with_tiebreak(C_r, combined_scores, m_next)
            self.candidate_set = list(C_r)
            self.round += 1
            
            if self.trace_enabled:
                self.round_trace.append({
                    "round": r,
                    "m_r": m_r,
                    "t_R_r": t_R_r,
                    "duels_executed": duels_executed,
                    "pulls_executed": pulls_executed,
                    "candidates_pre": C_pre,
                    "candidates_post": list(self.candidate_set),
                    "combined_scores": {str(k): round(v,4) for k,v in combined_scores.items()}
                })

        return self.history

    def get_candidate_set(self) -> List[int]:
        return list(self.candidate_set)

    def get_history(self) -> List[Tuple[Any, ...]]:
        return list(self.history)

    def get_round(self) -> int:
        return int(self.round)
    
    def get_round_tot(self) -> int:
        return self.num_rounds

    def dump_trace(self, filepath: Union[str, Path]) -> None:
        if not self.trace_enabled: return
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([_to_py(rt) for rt in self.round_trace], f, indent=2)

