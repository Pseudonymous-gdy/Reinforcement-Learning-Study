# Epsilon_Greedy.py
# ------------------------------------------------------------
# Compatible with System.py and the UCB2 driver.
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
import System
import matplotlib.pyplot as plt


class EpsilonGreedy:
    """
    Decaying‑ε greedy algorithm  (Auer et al., 2002, Thm 4).

    ε_t  =  min{1,  c·n / (Δ²·t)},   Δ = p_best − p_second_best
    """

    # ---------------- initialisation -----------------
    def __init__(self, environment: System.System, c: float = 0.05):
        if c <= 0:
            raise ValueError("c must be positive")

        self.env = environment
        self.n   = environment.n
        self.p   = np.asarray(environment.p, dtype=float)

        # gap Δ
        best, second_best = np.partition(self.p, -2)[-2:]
        self.delta = max(best - second_best, 1e-12)

        self.c        = c
        self.optimal  = int(np.argmax(self.p))

        # internal buffers (reset each run)
        self.rewards: np.ndarray
        self.counts:  np.ndarray
        self.t: int
        self.regret: float
        self._reset()

    # ---------------- helpers -----------------
    def _reset(self):
        self.rewards = np.zeros(self.n)
        self.counts  = np.zeros(self.n, dtype=int)
        self.t       = 0
        self.regret  = 0.0

    def _epsilon(self) -> float:
        return 1.0 if self.t == 0 else min(1.0, self.c * self.n / (self.delta**2 * self.t))

    def _select_arm(self) -> int:
        if np.random.random() < self._epsilon():                   # explore
            return np.random.randint(self.n)
        means = self.rewards / np.clip(self.counts, 1, None)       # exploit
        return int(np.argmax(means))

    def _step(self, table: list[np.ndarray]):
        arm    = self._select_arm()
        reward = table[arm][self.t]                                # reward at time t
        self.rewards[arm] += reward
        self.counts[arm]  += 1
        self.regret += table[self.optimal][self.t] - reward
        self.t += 1

    # ---------------- public API -----------------
    def simulate(self, horizon: int, runs: int = 100) -> tuple[float, float]:
        """
        Monte‑Carlo average over `runs` trajectories of length `horizon`.
        Returns (average cumulative regret, average optimal‑arm pull ratio).
        """
        if horizon < self.n:
            raise ValueError(f"horizon ({horizon}) must be ≥ number of arms ({self.n})")

        total_reg, opt_frac = 0.0, 0.0
        for seed in range(runs):
            table = self.env.simulate(times=horizon, seed=seed)
            self._reset()
            for _ in range(horizon):
                self._step(table)
            total_reg += self.regret
            opt_frac  += self.counts[self.optimal] / horizon

        return total_reg / runs, opt_frac / runs

    # identical to UCB2 helper — collect whole learning curve in one go
    def simulate_path(self, horizon: int, runs: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (avg_cum_regret[0:horizon], avg_opt_pull_frac[0:horizon]).
        """
        if horizon < self.n:
            raise ValueError(f"horizon ({horizon}) must be ≥ number of arms ({self.n})")

        reg_curve   = np.zeros(horizon)
        opt_curve   = np.zeros(horizon)

        for seed in range(runs):
            table = self.env.simulate(times=horizon, seed=seed)
            self._reset()

            opt_pulls = 0
            for t in range(horizon):
                self._step(table)
                if self.counts[self.optimal] > opt_pulls:
                    opt_pulls += 1
                reg_curve[t] += self.regret
                opt_curve[t] += opt_pulls / (t + 1)

        reg_curve /= runs
        opt_curve /= runs
        print(self.rewards/self.counts)
        return reg_curve, opt_curve


# ------------------------------------------------------------------ #
# Demo – identical style to UCB2 driver
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    n_arms = 10
    p      = [0.2, 0.4, 0.6, 0.8, 0.9,
              0.1, 0.3, 0.5, 0.7, 0.87]

    env = System.System(n_arms, p)

    c_list   = [0.3, 0.2, 0.15, 0.1, 0.05]   # exploration constants
    horizon  = 100_000                  # up to 10⁵ plays
    runs     = 20                       # Monte‑Carlo repetitions

    # X‑axis: log₁₀(time)
    X = np.linspace(1, 5, 100)          # 10¹ … 10⁵
    plays = (10 ** X).astype(int)

    reg_paths, opt_paths = {}, {}

    for c in c_list:
        algo = EpsilonGreedy(env, c)
        reg_curve, opt_curve = algo.simulate_path(horizon=max(plays), runs=runs)
        reg_paths[c] = reg_curve[plays - 1]    # align indices
        opt_paths[c] = opt_curve[plays - 1]
        print(f"c={c:<5}  done")

    # ------------- plotting -----------------
    # 1) cumulative regret
    plt.figure(figsize=(7, 4))
    for c, y in reg_paths.items():
        plt.semilogx(plays, y, label=f"c={c}")
    plt.title("Epsilon‑Greedy · cumulative regret")
    plt.xlabel("plays (log scale)")
    plt.ylabel("average cumulative regret")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2) optimal‑arm pull ratio
    plt.figure(figsize=(7, 4))
    for c, y in opt_paths.items():
        plt.semilogx(plays, y, label=f"c={c}")
    plt.title("Epsilon‑Greedy · best‑arm pull fraction")
    plt.xlabel("plays (log scale)")
    plt.ylabel("fraction of pulls on optimal arm")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
