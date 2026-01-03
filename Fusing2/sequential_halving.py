r"""
Fusing2.sequential_halving

FusionSequentialHalving-DuelThenReward (no-CR) — robust to extreme hyperparameters.

Key robustness rules (extremes / degeneracy):
- alpha=1  => reward-only sequential halving (skip duel filtering regardless of gamma)
- alpha=0  => duel-only sequential halving (skip reward ranking; eliminate by Borda)
- gamma in {0,1} handled by clamped keep count; if nD==0, gamma is ignored (no duel filtering)
- episode_keep in {0,1} handled by clamped keep count; if keep==1 no elimination this round (but budget still caps total rounds)
- If budget ends with |C|>1, finalize to a single arm using available estimates:
    - if any reward samples on C: pick by mu_hat (tie-break by Borda)
    - else: pick by Borda (tie-break by index)

Sampling:
- Duel phase: uniform random pairs from current candidates (with replacement, i != j)
- Reward phase: uniform allocation (as-even-as-possible) across C_tilde to reduce variance and match SH behavior
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


def _clamp_keep_count_ceil(ratio: float, n: int) -> int:
    """Keep count based on ceil(ratio*n) but clamped to [1,n]."""
    if n <= 1:
        return 1
    if ratio <= 0.0:
        return 1
    if ratio >= 1.0:
        return n
    return max(1, min(n, int(math.ceil(ratio * n))))


def _topk_with_tiebreak(items: List[int], score: Dict[int, float], k: int) -> List[int]:
    """
    Deterministic: score desc, index asc.
    """
    k = max(1, min(int(k), len(items)))
    return sorted([int(x) for x in items], key=lambda a: (-float(score[int(a)]), int(a)))[:k]


class FusionSequentialHalving:
    def __init__(
        self,
        env: Environment,
        total_budget: int,
        alpha: float = 0.1,
        gamma: Union[float, np.float64] = 1 / np.sqrt(2),
        episode_keep: Union[float, np.float64] = 0.5,  # reward keep-ratio (default 1/2)
        seed: Optional[int] = None,
        num_rounds: Optional[int] = None,
        trace_enabled: bool = False,
    ):
        self.env = env
        self.K = int(env.number_of_bandits)

        self.total_budget = int(total_budget)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.episode_keep = float(episode_keep)

        assert 0.0 <= self.alpha <= 1.0, "alpha must be in [0,1]"
        assert 0.0 <= self.gamma <= 1.0, "gamma must be in [0,1]"
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

        # Reward stats
        self.N = np.zeros(self.K, dtype=np.int64)
        self.reward_sum = np.zeros(self.K, dtype=np.float64)
        self.mu_hat = np.zeros(self.K, dtype=np.float64)

        # Dueling stats
        self.M = np.zeros((self.K, self.K), dtype=np.int64)
        self.W = np.zeros((self.K, self.K), dtype=np.int64)
        self.nu_hat = np.full((self.K, self.K), 0.5, dtype=np.float64)
        np.fill_diagonal(self.nu_hat, 0.5)

    # -----------------------
    # Dueling interface bridge
    # -----------------------
    def _duel_winner(self, i: int, j: int) -> int:
        """
        Preferred: env.duel(i,j) -> winner index i or j
        Fallback: env.get_dueling(i,j) -> 1/0/0.5 (1 means i wins)
        """
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
            return int(self.rng.choice([i, j]))  # tie break

        # last resort: compare sampled rewards
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
        """
        Return a list of arms to pull of length nR, allocated as evenly as possible across C.
        This reduces variance and matches the spirit of sequential halving.
        """
        if nR <= 0 or len(C) == 0:
            return []
        m = len(C)
        base = nR // m
        rem = nR % m
        schedule: List[int] = []
        # base pulls
        for a in C:
            schedule.extend([int(a)] * base)
        # remainder: random subset (or with replacement if rem>m, though rem<=m here)
        if rem > 0:
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

        # Split budget (same convention as your code)
        nD = int(math.floor(T_r * (1.0 - self.alpha)))
        nR = int(T_r - nD)

        # Degeneracy modes (for trace)
        mode = "fusion"
        if nD == 0 and nR > 0:
            mode = "reward_only_SH"
        elif nR == 0 and nD > 0:
            mode = "duel_only_SH"
        elif nD == 0 and nR == 0:
            mode = "no_budget"

        round_log: Dict[str, Any] = {}
        if self.trace_enabled:
            round_log = {
                "round": int(self.round),
                "mode": mode,
                "T_r": int(T_r),
                "n_r": int(n_r),
                "candidate_pre": list(map(int, C)),
                "nD": int(nD),
                "nR": int(nR),
            }

        # -------------------
        # 1) Dueling phase
        # -------------------
        if nD > 0 and n_r >= 2:
            # uniform random pairs from C, with replacement, i != j
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

        # Empirical Borda on C (always computed because used for tie-break/fallback/finalize)
        borda_C = self._borda_scores(C)

        # -------------------
        # 2) Duel-based filtering (C_tilde)
        # -------------------
        # If no duels, DO NOT filter by Borda (all 0.5 -> arbitrary): keep all.
        # If gamma==1, keep all.
        if nD == 0 or self.gamma >= 1.0:
            C_tilde = list(C)
            mD_eff = int(n_r)
        else:
            mD_eff = _clamp_keep_count_ceil(self.gamma, n_r)
            C_tilde = _topk_with_tiebreak(C, borda_C, mD_eff)

        if self.trace_enabled:
            round_log["mD_eff"] = int(mD_eff)
            # store only top-5 borda for readability
            top5_borda = sorted([(i, borda_C[i]) for i in borda_C], key=lambda x: x[1], reverse=True)[:5]
            round_log["top5_borda_on_C"] = top5_borda
            round_log["C_tilde"] = list(map(int, C_tilde))

        if len(C_tilde) <= 1:
            # done this round
            self.candidate_set = list(C_tilde)
            self.spent_budget += T_r
            if self.trace_enabled:
                round_log["candidate_post"] = list(map(int, self.candidate_set))
                round_log["spent_budget_after"] = int(self.spent_budget)
                self.round_trace.append(round_log)
            return self.candidate_set

        # -------------------
        # 3) Reward phase
        # -------------------
        if nR > 0:
            # more SH-like: allocate pulls evenly on C_tilde
            schedule = self._uniform_reward_allocate(C_tilde, nR)
            for a in schedule:
                r = float(self.env.get_reward(int(a)))
                self.history.append(("R", int(a), float(r)))
                self.N[a] += 1
                self.reward_sum[a] += r
                self.mu_hat[a] = self.reward_sum[a] / self.N[a]

        # -------------------
        # 4) Decide next candidate set size (elimination)
        # -------------------
        # Target keep based on episode_keep (default 0.5 => SH halving).
        m_next = _clamp_keep_count_floor(self.episode_keep, n_r)
        # cannot exceed available candidates after duel-filtering
        m_next = min(m_next, len(C_tilde))

        # Ranking rule:
        # - If we have reward samples on C_tilde this round (or historically), rank by mu_hat
        # - Else rank by Borda (duel-only mode)
        has_reward_info = any(self.N[int(a)] > 0 for a in C_tilde)

        if has_reward_info:
            # primary: mu_hat; secondary: borda on C_tilde
            borda_tilde = self._borda_scores(C_tilde)
            def key(a: int):
                return (-float(self.mu_hat[a]), -float(borda_tilde.get(a, 0.5)), int(a))
            C_sorted = sorted([int(x) for x in C_tilde], key=key)
        else:
            # duel-only: use borda on C_tilde
            borda_tilde = self._borda_scores(C_tilde)
            C_sorted = _topk_with_tiebreak(C_tilde, borda_tilde, k=len(C_tilde))

        self.candidate_set = C_sorted[:m_next]

        self.spent_budget += T_r

        if self.trace_enabled:
            round_log["m_next"] = int(m_next)
            # snapshot for audit
            round_log["mu_hat_snapshot_on_C_tilde"] = {str(int(k)): float(self.mu_hat[int(k)]) for k in C_tilde}
            round_log["candidate_post"] = list(map(int, self.candidate_set))
            round_log["spent_budget_after"] = int(self.spent_budget)
            self.round_trace.append(round_log)

        return self.candidate_set

    def _finalize_if_needed(self) -> None:
        """
        If budget ends with multiple candidates, pick a single best arm deterministically.
        """
        C = list(self.candidate_set)
        if len(C) <= 1:
            return

        borda = self._borda_scores(C)
        has_reward = any(self.N[int(a)] > 0 for a in C)

        if has_reward:
            def key(a: int):
                return (-float(self.mu_hat[a]), -float(borda.get(a, 0.5)), int(a))
            best = sorted([int(x) for x in C], key=key)[0]
        else:
            # duel-only
            best = _topk_with_tiebreak(C, borda, k=1)[0]

        self.candidate_set = [int(best)]
        if self.trace_enabled:
            self.round_trace.append({
                "round": int(self.round),
                "mode": "finalize",
                "candidate_pre": list(map(int, C)),
                "candidate_post": list(map(int, self.candidate_set)),
                "note": "budget ended with |C|>1, finalized to single arm",
            })

    # -----------------------
    # Run full algorithm
    # -----------------------
    def run(self) -> List[Tuple[Any, ...]]:
        # reset dynamic state
        self.candidate_set = list(range(self.K))
        self.history = []
        self.round_trace = []
        self.round = 0
        self.spent_budget = 0

        # reset estimates
        self.N[:] = 0
        self.reward_sum[:] = 0.0
        self.mu_hat[:] = 0.0
        self.M[:, :] = 0
        self.W[:, :] = 0
        self.nu_hat[:, :] = 0.5
        np.fill_diagonal(self.nu_hat, 0.5)

        # Run rounds until budget exhausted or 1 candidate left
        while self.spent_budget < self.total_budget and len(self.candidate_set) > 1:
            remaining = self.total_budget - self.spent_budget
            T_r = min(self.budget_per_round, remaining)
            self.run_round(T_r=T_r)

        # If budget ends but still multiple candidates, finalize
        self._finalize_if_needed()
        return self.history

    # -----------------------
    # Getters
    # -----------------------
    def get_candidate_set(self) -> List[int]:
        return list(self.candidate_set)

    def get_history(self) -> List[Tuple[Any, ...]]:
        return list(self.history)

    def get_round(self) -> int:
        return int(self.round)

    def get_round_tot(self) -> int:
        return int(self.spent_budget)

    def get_total_budget(self) -> int:
        return int(self.total_budget)

    def get_budget(self) -> int:
        return int(self.budget_per_round)

    def get_alpha(self) -> float:
        return float(self.alpha)

    def get_gamma(self) -> float:
        return float(self.gamma)


# ============================================================
# Optional: unit test (kept compatible with your original style)
# ============================================================
if __name__ == "__main__":
    TEST_ID = "FSH_DuelThenReward_noCR_extreme_robust"
    TOTAL_BUDGET = 4096
    ENV_SEED = 12345
    ALGO_SEED = 67890

    # Example fixed means (length defines K)
    fixed_means = [
        0.68, 0.80, 0.66, 0.60, 0.72, 0.46, 0.40, 0.30,
        0.20, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01, 0.00,
        0.89, 0.78, 0.58, 0.48, 0.38, 0.28, 0.18, 0.14,
        0.37, 0.26, 0.19, 0.54, 0.90, 0.25, 0.64, 0.97
    ]
    K = len(fixed_means)

    # Dueling matrix consistent with means
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

    # IMPORTANT: your Environment.set_means() sorts; here we want fixed identity order,
    # so we directly overwrite bandit_means if present.
    if hasattr(env, "bandit_means"):
        env.bandit_means = np.array(fixed_means, dtype=float)
        env.optimal_mean = float(np.max(env.bandit_means))

    # Try a few extreme settings quickly
    test_settings = [
        dict(alpha=1.0, gamma=0.0, episode_keep=0.5),  # reward-only SH
        dict(alpha=0.0, gamma=1.0, episode_keep=0.5),  # duel-only SH
        dict(alpha=0.2, gamma=0.0, episode_keep=0.5),  # aggressive duel filter
        dict(alpha=0.2, gamma=1.0, episode_keep=1.0),  # no elimination, finalize at end
    ]

    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    for idx, hp in enumerate(test_settings):
        algo = FusionSequentialHalving(
            env=env,
            total_budget=TOTAL_BUDGET,
            seed=ALGO_SEED,
            trace_enabled=True,
            **hp,
        )
        history = algo.run()
        final_candidates = algo.get_candidate_set()

        # basic invariants
        num_duels = sum(1 for h in history if h[0] == "D")
        num_rewards = sum(1 for h in history if h[0] == "R")
        assert num_duels + num_rewards == algo.spent_budget
        assert int(np.sum(algo.N)) == num_rewards
        assert int(np.sum(algo.M)) == 2 * num_duels
        assert len(final_candidates) == 1, "algorithm must output a single arm after finalize"

        means_arr = np.array(env.bandit_means, dtype=float)
        best_arm = int(np.argmax(means_arr))
        ok = int(final_candidates[0] == best_arm)

        base_name = f"{TEST_ID}_{idx}_K{K}_B{TOTAL_BUDGET}_env{ENV_SEED}_algo{ALGO_SEED}"
        md_path = report_dir / f"{base_name}.md"
        json_path = report_dir / f"{base_name}.json"

        report = {
            "config": {
                "K": K,
                "total_budget": TOTAL_BUDGET,
                **hp,
                "env_seed": ENV_SEED,
                "algo_seed": ALGO_SEED,
            },
            "summary": {
                "spent_budget": int(algo.spent_budget),
                "duels": int(num_duels),
                "rewards": int(num_rewards),
                "rounds_executed": int(algo.round),
                "final_candidates": list(map(int, final_candidates)),
                "best_arm_by_means": int(best_arm),
                "correct": int(ok),
            },
            "round_trace": algo.round_trace,
        }
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        lines = []
        lines.append(f"# Unit Test Report: {TEST_ID} (case {idx})")
        lines.append("")
        lines.append("## Hyperparameters")
        for k, v in hp.items():
            lines.append(f"- {k} = {v}")
        lines.append("")
        lines.append("## Summary")
        lines.append(f"- spent_budget = {algo.spent_budget}")
        lines.append(f"- duels = {num_duels}")
        lines.append(f"- rewards = {num_rewards}")
        lines.append(f"- rounds_executed = {algo.round}")
        lines.append(f"- final_candidates = {final_candidates}")
        lines.append(f"- best_arm_by_means = {best_arm}")
        lines.append(f"- correct = {ok}")
        lines.append("")
        lines.append("## Round trace (compact)")
        for lg in algo.round_trace:
            lines.append(f"- r={lg.get('round')} mode={lg.get('mode')} | "
                         f"n_r={lg.get('n_r')} nD={lg.get('nD')} nR={lg.get('nR')} "
                         f"mD_eff={lg.get('mD_eff')} m_next={lg.get('m_next')} "
                         f"| post_size={len(lg.get('candidate_post', [])) if isinstance(lg.get('candidate_post'), list) else 'NA'}")
        md_path.write_text("\n".join(lines), encoding="utf-8")

    print("✅ extreme-robust unit tests PASSED.")
    print(f"Reports saved under: {report_dir}")
