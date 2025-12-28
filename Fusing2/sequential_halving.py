r"""
Fusing2.sequential_halving

FusionSequentialHalving-DuelThenReward (no-CR)

Semantics (matches pasted.txt):
1) Dueling phase: sample pairs (i,j), observe winner, update M and nu_hat
2) Compute empirical Borda from nu_hat on current candidate set
3) Keep top ceil(gamma * n_r) -> C_tilde
4) Reward phase: pull arms from C_tilde, update N and mu_hat
5) Keep top floor(n_r/2) -> C_{r+1}

Sampling is WITH replacement in both phases.
"""

import os
import sys
import math
import json
import platform
from dataclasses import dataclass
from pathlib import Path
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


class FusionSequentialHalving:
    def __init__(
        self,
        env: Environment,
        total_budget: int,
        alpha: float = 0.1,
        gamma: Union[float, np.float64] = 1 / np.sqrt(2),
        episode_keep: Union[float, np.float64] = 0.5,   # legacy; algorithm uses 1/2 in reward elimination
        seed: Optional[int] = None,
        num_rounds: Optional[int] = None,
        trace_enabled: bool = False,  # NEW: enable round-by-round trace for auditability
    ):
        self.env = env
        self.K = int(env.number_of_bandits)

        self.total_budget = int(total_budget)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.episode_keep = float(episode_keep)

        assert 0.0 < self.alpha < 1.0, "alpha must be in (0,1)"
        assert 0.5 < self.gamma <= 1.0, "gamma must be in (1/2,1]"
        assert self.total_budget >= 0, "total_budget must be nonnegative"

        self.rng = np.random.default_rng(seed)

        # Candidates
        self.candidate_set: List[int] = list(range(self.K))

        # History: ("D", i, j, winner) or ("R", a, reward)
        self.history: List[Tuple[Any, ...]] = []

        # Trace (round-level structured logs)
        self.trace_enabled = bool(trace_enabled)
        self.round_trace: List[Dict[str, Any]] = []

        # Budget accounting
        self.spent_budget: int = 0
        self.round: int = 0

        # Number of rounds (sequential halving)
        if num_rounds is None:
            self.num_rounds = 1 if self.K <= 1 else int(math.ceil(math.log2(self.K)))
        else:
            self.num_rounds = max(1, int(num_rounds))

        self.budget_per_round = int(math.ceil(self.total_budget / self.num_rounds)) if self.num_rounds > 0 else self.total_budget

        # Reward stats (global K)
        self.N = np.zeros(self.K, dtype=np.int64)
        self.reward_sum = np.zeros(self.K, dtype=np.float64)
        self.mu_hat = np.zeros(self.K, dtype=np.float64)

        # Dueling stats (global KxK)
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
            w = self.env.duel(i, j)
            w = int(w)
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

        # last resort
        ri = float(self.env.get_reward(i))
        rj = float(self.env.get_reward(j))
        return int(i if ri >= rj else j)

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

        # Split budget (match LaTeX)
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
        # 1) Dueling phase
        # -------------------
        if nD > 0 and n_r >= 2:
            for _ in range(nD):
                i = int(self.rng.choice(C))
                C_wo_i = [x for x in C if x != i]
                j = int(self.rng.choice(C_wo_i))

                winner = self._duel_winner(i, j)
                self.history.append(("D", i, j, int(winner)))

                # update counts
                self.M[i, j] += 1
                self.M[j, i] += 1
                if winner == i:
                    self.W[i, j] += 1
                else:
                    self.W[j, i] += 1

                self.nu_hat[i, j] = self.W[i, j] / self.M[i, j]
                self.nu_hat[j, i] = self.W[j, i] / self.M[j, i]

        # Empirical Borda on C
        denom = max(1, n_r - 1)
        borda: Dict[int, float] = {}
        for k in C:
            s = 0.0
            for j in C:
                if j == k:
                    continue
                s += float(self.nu_hat[k, j])
            borda[int(k)] = float(s / denom)

        # Keep top ceil(gamma * n_r)
        mD = max(1, int(math.ceil(self.gamma * n_r)))
        C_sorted_duel = sorted(C, key=lambda k: borda[int(k)], reverse=True)
        C_tilde = C_sorted_duel[:mD]

        if self.trace_enabled:
            round_log["borda"] = {str(k): float(v) for k, v in borda.items()}
            round_log["C_tilde"] = list(map(int, C_tilde))

        if len(C_tilde) <= 1:
            self.candidate_set = C_tilde
            self.spent_budget += T_r
            if self.trace_enabled:
                round_log["candidate_post"] = list(map(int, self.candidate_set))
                round_log["spent_budget_after"] = int(self.spent_budget)
                self.round_trace.append(round_log)
            return self.candidate_set

        # -------------------
        # 2) Reward phase
        # -------------------
        if nR > 0:
            for _ in range(nR):
                a = int(self.rng.choice(C_tilde))
                r = float(self.env.get_reward(a))
                self.history.append(("R", a, r))

                self.N[a] += 1
                self.reward_sum[a] += r
                self.mu_hat[a] = self.reward_sum[a] / self.N[a]

        # Keep top floor(n_r/2) by mu_hat
        m_r = max(1, int(math.floor(n_r / 2)))
        m_r = min(m_r, len(C_tilde))
        C_sorted_reward = sorted(C_tilde, key=lambda k: float(self.mu_hat[k]), reverse=True)
        self.candidate_set = C_sorted_reward[:m_r]

        self.spent_budget += T_r

        if self.trace_enabled:
            round_log["mu_hat_snapshot_on_C_tilde"] = {str(int(k)): float(self.mu_hat[k]) for k in C_tilde}
            round_log["candidate_post"] = list(map(int, self.candidate_set))
            round_log["spent_budget_after"] = int(self.spent_budget)
            self.round_trace.append(round_log)

        return self.candidate_set

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

        while self.spent_budget < self.total_budget and len(self.candidate_set) > 1:
            remaining = self.total_budget - self.spent_budget
            T_r = min(self.budget_per_round, remaining)
            self.run_round(T_r=T_r)

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
# Unit test + reproducible/auditable report
# ============================================================
if __name__ == "__main__":
    # ----------------------------
    # 0) Deterministic test config
    # ----------------------------
    TEST_ID = "FSH_DuelThenReward_noCR"
    K = 31
    TOTAL_BUDGET = 4096
    ALPHA = 0.2
    GAMMA = float(1 / np.sqrt(2))
    ENV_SEED = 12345
    ALGO_SEED = 67890

    # Use descending means to avoid arm-identity confusion if your env sorts internally
    fixed_means = [0.68, 0.80, 0.66, 0.60, 0.72, 0.46, 0.40, 0.30,
                   0.20, 0.10, 0.08, 0.06, 0.04, 0.02, 0.01, 0.00,
                   0.89, 0.78, 0.58, 0.48, 0.38, 0.28, 0.18, 0.14,
                   0.37, 0.26, 0.19, 0.54, 0.90, 0.25, 0.64]

    # Construct a consistent dueling matrix P[a,b] = 0.5 + (mu[a]-mu[b])/2
    dueling_means = np.zeros((K, K), dtype=float)
    for a in range(K):
        for b in range(K):
            if a == b:
                dueling_means[a, b] = 0.5
            else:
                p = 0.5 + (fixed_means[a] - fixed_means[b]) / 2.0
                dueling_means[a, b] = float(min(1.0, max(0.0, p)))

    # ----------------------------
    # 1) Build env + algorithm
    # ----------------------------
    env = Environment(
        number_of_bandits=K,
        distribution="bernoulli",
        seed=ENV_SEED,
        dueling_means=dueling_means,
    )

    # Force env means (note: if your env sorts internally, we gave means already sorted desc)
    if hasattr(env, "set_means"):
        env.set_means(fixed_means)

    algo = FusionSequentialHalving(
        env=env,
        total_budget=TOTAL_BUDGET,
        alpha=ALPHA,
        gamma=GAMMA,
        seed=ALGO_SEED,
        trace_enabled=True,
    )

    # ----------------------------
    # 2) Run
    # ----------------------------
    history = algo.run()
    final_candidates = algo.get_candidate_set()

    # ----------------------------
    # 3) Invariant checks (unit test assertions)
    # ----------------------------
    # A) Budget accounting
    num_duels = sum(1 for h in history if h[0] == "D")
    num_rewards = sum(1 for h in history if h[0] == "R")
    assert num_duels + num_rewards == algo.spent_budget, "spent_budget must equal total recorded samples"
    assert algo.spent_budget <= TOTAL_BUDGET, "must not exceed total budget"

    # B) Reward counters consistent
    assert int(np.sum(algo.N)) == num_rewards, "sum(N) must equal number of reward samples"

    # C) Duel counters consistent: each duel increments M twice (i,j) and (j,i)
    assert int(np.sum(algo.M)) == 2 * num_duels, "sum(M) must equal 2 * number_of_duels"

    # D) nu_hat consistent with W/M where M>0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            if algo.M[i, j] > 0:
                est = algo.W[i, j] / algo.M[i, j]
                assert abs(float(algo.nu_hat[i, j]) - float(est)) < 1e-12, "nu_hat must equal W/M when M>0"

    # E) Candidate size rule each round: |C_{r+1}| = min(floor(n_r/2), ceil(gamma*n_r)) but gamma>1/2 => floor(n_r/2) usually binds
    prev_n = K
    for log in algo.round_trace:
        n_r = int(log["n_r"])
        post = log["candidate_post"]
        expected = max(1, min(int(math.floor(n_r / 2)), int(math.ceil(GAMMA * n_r))))
        assert len(post) == expected, f"candidate size mismatch: got {len(post)} expected {expected}"
        prev_n = len(post)

    # ----------------------------
    # 4) Generate auditable report
    # ----------------------------
    report_dir = Path(__file__).resolve().parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # deterministic-ish filename (depends only on config)
    base_name = f"{TEST_ID}_K{K}_B{TOTAL_BUDGET}_env{ENV_SEED}_algo{ALGO_SEED}"
    json_path = report_dir / f"{base_name}.json"
    md_path = report_dir / f"{base_name}.md"

    # Collect report content
    report = {
        "meta": {
            "test_id": TEST_ID,
            "generated_at_local": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "config": {
            "K": K,
            "total_budget": TOTAL_BUDGET,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "env_seed": ENV_SEED,
            "algo_seed": ALGO_SEED,
            "distribution": getattr(env, "distribution", None),
        },
        "environment": {
            "bandit_means": _to_py(np.array(getattr(env, "bandit_means", fixed_means), dtype=float)),
            "optimal_arm_by_means": int(np.argmax(np.array(getattr(env, "bandit_means", fixed_means), dtype=float))),
            "dueling_means_shape": list(np.array(dueling_means).shape),
        },
        "run_summary": {
            "spent_budget": int(algo.spent_budget),
            "num_duels": int(num_duels),
            "num_rewards": int(num_rewards),
            "num_rounds_executed": int(algo.round),
            "final_candidates": list(map(int, final_candidates)),
        },
        "estimates": {
            "N": _to_py(algo.N),
            "mu_hat": _to_py(algo.mu_hat),
            # nu_hat can be large; keep it but it's useful for audit
            "M_sum": int(np.sum(algo.M)),
            "W_sum": int(np.sum(algo.W)),
        },
        "round_trace": algo.round_trace,      # already JSON-friendly
        "history_head_50": [tuple(map(_to_py, h)) for h in history[:50]],
        "history_tail_50": [tuple(map(_to_py, h)) for h in history[-50:]],
    }

    # Write JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Write Markdown (human-readable)
    lines = []
    lines.append(f"# Unit Test Report: {TEST_ID}")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- K = {K}")
    lines.append(f"- total_budget = {TOTAL_BUDGET}")
    lines.append(f"- alpha = {ALPHA}")
    lines.append(f"- gamma = {GAMMA}")
    lines.append(f"- env_seed = {ENV_SEED}")
    lines.append(f"- algo_seed = {ALGO_SEED}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- spent_budget = {algo.spent_budget}")
    lines.append(f"- duels = {num_duels}")
    lines.append(f"- rewards = {num_rewards}")
    lines.append(f"- rounds_executed = {algo.round}")
    lines.append(f"- final_candidates = {final_candidates}")
    lines.append("")
    lines.append("## Bandit means")
    means_arr = np.array(getattr(env, "bandit_means", fixed_means), dtype=float)
    best_arm = int(np.argmax(means_arr))
    lines.append(f"- means = {means_arr.tolist()}")
    lines.append(f"- best_arm_by_means = {best_arm}")
    lines.append("")
    lines.append("## Round-by-round trace")
    lines.append("")
    for log in algo.round_trace:
        r = log["round"]
        lines.append(f"### Round {r}")
        lines.append(f"- T_r={log['T_r']}  n_r={log['n_r']}  nD={log['nD']}  nR={log['nR']}")
        lines.append(f"- C_pre={log['candidate_pre']}")
        lines.append(f"- C_tilde={log.get('C_tilde', [])}")
        lines.append(f"- C_post={log['candidate_post']}")
        # show top-3 borda
        borda = log.get("borda", {})
        if isinstance(borda, dict) and len(borda) > 0:
            top3 = sorted([(int(k), float(v)) for k, v in borda.items()], key=lambda x: x[1], reverse=True)[:3]
            lines.append(f"- top3_borda={top3}")
        # show mu_hat snapshot on C_tilde
        mu_snap = log.get("mu_hat_snapshot_on_C_tilde", {})
        if isinstance(mu_snap, dict) and len(mu_snap) > 0:
            snap_sorted = sorted([(int(k), float(v)) for k, v in mu_snap.items()], key=lambda x: x[1], reverse=True)
            lines.append(f"- mu_hat_on_C_tilde(sorted)={snap_sorted}")
        lines.append(f"- spent_budget_after={log['spent_budget_after']}")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    best_arm_by_means = int(np.argmax(np.array(getattr(env, "bandit_means", fixed_means), dtype=float)))
    assert final_candidates == [best_arm_by_means], "did not identify the best arm under fixed seeds"
    
    # ----------------------------
    # 5) Print pointers
    # ----------------------------
    print("✅ Unit test PASSED.")
    print(f"Report (JSON): {json_path}")
    print(f"Report (MD)  : {md_path}")
