# UCB_Normal.py
# ------------------------------------------------------------
# Compatible with System.py and previous UCB2 / EpsilonGreedy drivers.
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np
import System
import matplotlib.pyplot as plt


class UCBNormal:
    r"""
    UCB‑Normal / UCB‑Empirical‑Variance  (Auer, Cesa‑Bianchi & Fischer, 2002).

    Each arm i maintains
        μ̂_i(t)  – empirical mean
        σ̂²_i(t) – unbiased empirical variance
    and is selected according to

        i_t = argmax_i  μ̂_i(t) + √[  2 σ̂²_i(t) ln t / n_i(t) ]
                                    + 3 ln t / n_i(t)            (1)

    where the second term (3 ln t / n_i) is the high‑probability bound
    when σ̂² might underestimate the true variance.

    We keep the same API as UCB2:

        algo = UCBNormal(env)
        avg_reg, opt_ratio = algo.simulate(horizon=10_000, runs=100)
    """

    # ---------------- initialisation -----------------
    def __init__(self, environment: System.System):
        self.env = environment
        self.n   = environment.n
        self.p   = np.asarray(environment.p, dtype=float)

        self.optimal = int(np.argmax(self.p))
        self._reset()

    # ---------------- helpers -----------------
    def _reset(self):
        self.counts  = np.zeros(self.n, dtype=int)
        self.sum_r   = np.zeros(self.n)           # ∑ rewards
        self.sum_r2  = np.zeros(self.n)           # ∑ rewards²
        self.t       = 0
        self.regret  = 0.0

    def _empirical_mean(self) -> np.ndarray:
        return self.sum_r / np.clip(self.counts, 1, None)

    def _empirical_var(self) -> np.ndarray:
        # unbiased variance; when n_i < 2 return maximal variance 0.25
        var = (self.sum_r2 - (self.sum_r**2) / np.clip(self.counts, 1, None))
        denom = np.clip(self.counts - 1, 1, None)
        est = var / denom
        est[self.counts < 2] = 0.25               # safe upper bound for Bernoulli
        return est

    def _bonus(self) -> np.ndarray:
        """UCB‑Normal confidence width (cf. eq. (1) above)."""
        with np.errstate(divide="ignore"):
            ln_t = np.log(max(self.t, 1))
            n_i  = np.clip(self.counts, 1, None)
            var  = self._empirical_var()
            return np.sqrt(2 * var * ln_t / n_i) + 3 * ln_t / n_i

    def _select_arm(self) -> int:
        if self.t < self.n:                # initialise: pull each arm once
            return self.t
        return int(np.argmax(self._empirical_mean() + self._bonus()))

    def _step(self, table: list[np.ndarray]):
        arm    = self._select_arm()
        reward = table[arm][self.t]
        self.t += 1

        self.counts[arm] += 1
        self.sum_r[arm]  += reward
        self.sum_r2[arm] += reward**2
        self.regret      += table[self.optimal][self.t - 1] - reward

    # ---------------- public API -----------------
    def simulate(self, horizon: int, runs: int = 100) -> tuple[float, float]:
        """
        Average cumulative regret & optimal‑arm ratio over `runs` episodes.
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

    def simulate_path(self, horizon: int, runs: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Return full curves: (avg_cum_regret, avg_opt_pull_frac)
        Shape = (horizon,)
        """
        if horizon < self.n:
            raise ValueError(f"horizon ({horizon}) must be ≥ number of arms ({self.n})")

        reg_curve = np.zeros(horizon)
        opt_curve = np.zeros(horizon)

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
        return reg_curve, opt_curve


# ------------------------------------------------------------------ #
# Quick demo identical to previous drivers
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    n_arms = 10
    p      = [0.2, 0.4, 0.6, 0.8, 0.9,
              0.1, 0.3, 0.5, 0.7, 0.87]
    env = System.System(n_arms, p)

    horizon = 100_000
    runs    = 100

    algo = UCBNormal(env)
    reg, opt = algo.simulate(horizon, runs)
    print(f"Avg regret  : {reg:.2f}")
    print(f"Opt‑arm freq: {opt:.3f}")

    # full learning curve (optional)
    plays = np.logspace(1, 5, 100, base=10, dtype=int)  # 10¹ … 10⁵
    reg_curve, opt_curve = algo.simulate_path(horizon=max(plays), runs=runs)

    plt.figure(figsize=(7, 4))
    plt.semilogx(plays, reg_curve[plays - 1])
    plt.title("UCB‑Normal · cumulative regret")
    plt.xlabel("plays (log scale)")
    plt.ylabel("average cumulative regret")
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.semilogx(plays, opt_curve[plays - 1])
    plt.title("UCB‑Normal · best‑arm pull fraction")
    plt.xlabel("plays (log scale)")
    plt.ylabel("fraction of pulls on optimal arm")
    plt.grid(True, ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()
