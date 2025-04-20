# UCB_2.py
import numpy as np
import System
import matplotlib.pyplot as plt


class UCB2(System.UCB):
    """
    An implementation of the UCB‑2 algorithm from
        Auer, Cesa‑Bianchi & Fischer (2002).
    The parameter `alpha` ∈ (0, 1] controls the width of the confidence
    interval: smaller alpha ⇒ wider bonus ⇒ more exploration.
    """

    def __init__(self, environment: System.System, alpha: float = 0.1):
        super().__init__(environment)              # initialise parent class
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.epochs = np.zeros(self.n, dtype=int)  # r_i  in the paper

    # ------------------------------------------------------------------ #
    # Helper functions
    # ------------------------------------------------------------------ #
    def _tau(self, r: int) -> int:
        """Epoch length τ_i(r) = ⌈(1+α)^r⌉ (eq. 6 in the paper)."""
        return int((1.0 + self.alpha) ** r)+1

    def _bonus(self, arm: int, t: int) -> float:
        """
        Confidence‑bonus term in UCB‑2 (eq. 7).
        Uses current time step `t` and the epoch length for this arm.
        """
        tau_r = self._tau(self.epochs[arm])
        return np.sqrt(((1.0 + self.alpha) * np.log(np.e * t / tau_r))
                       / (2.0 * tau_r))
    # ------------------------------------------------------------------ #

    def simulate(self, horizon: int, runs: int = 100):
        """
        Run `runs` independent Monte‑Carlo simulations, each of length
        `horizon`, and return:
            – average cumulative regret
            – average fraction of pulls on the optimal arm
        """

        if horizon < self.n:  # ← add this line
            raise ValueError("horizon must be at least the number of arms")
        total_regret = 0.0
        opt_fraction = 0.0

        for seed in range(runs):
            rewards_table = self.environment.simulate(times=horizon, seed=seed)

            # reset state for this run
            self.rewards.fill(0.0)
            self.counts.fill(0)
            self.epochs.fill(0)
            self.regret = 0.0

            # pull each arm once for initial estimates
            for arm in range(self.n):
                reward = rewards_table[arm][arm]
                self.rewards[arm] += reward
                self.counts[arm] += 1
            t = self.n  # current time step

            # main loop
            while t < horizon:
                means = self.rewards / self.counts
                bonuses = np.array([self._bonus(i, t) for i in range(self.n)])
                arm = int(np.argmax(means + bonuses))

                # how many pulls remain in the current epoch for this arm?
                pulls_left = self._tau(self.epochs[arm] + 1) \
                             - self._tau(self.epochs[arm])
                pulls_left = min(pulls_left, horizon - t)

                for _ in range(pulls_left):
                    reward = rewards_table[arm][t]
                    self.rewards[arm] += reward
                    self.counts[arm] += 1
                    self.regret += rewards_table[self.optimal][t] - reward
                    t += 1

                self.epochs[arm] += 1  # finished one epoch on this arm

            total_regret += self.regret
            opt_fraction += self.counts[self.optimal] / horizon

        return total_regret / runs, opt_fraction / runs


# --------------------------------------------------------------------- #
# Quick self‑test
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    # ----- experiment set‑up -----
    n_arms   = 10
    p        = [0.2, 0.4, 0.6, 0.8, 0.9,
                0.1, 0.3, 0.5, 0.7, 0.99]

    alphas   = [1e-1, 1e-2, 1e-3, 1e-4]     # 0.1 → 0.0001
    horizon  = 100_000                      # 10⁵ plays
    runs     = 20                           # Monte‑Carlo repetitions
                                             # (increase if you need smoother curves)

    env = System.System(n_arms, p)

    # containers for the curves
    regret_paths   = {}
    opt_pull_paths = {}
    X = np.linspace(1, 4, 100)

    for alpha in alphas:
        model = UCB2(env, alpha)
        Y1 = []
        Y2 = []
        for i in range(len(X)):
            a, b = model.simulate(int(10 ** X[i]))
            Y1.append(a)
            Y2.append(b)
            print('\r', '*' * int(20 * (10 ** X[i]) / 10000) + '-' * (20 - int(20 * (10 ** X[i]) / 10000)), '|',
                  f'Times:{int(10 ** X[i])}/100000', f'Regret: {a}', f'Most arm played: {b}', end='')
        regret_paths[alpha] = Y1
        opt_pull_paths[alpha] = Y2

    for key in regret_paths:
        plt.plot(X, regret_paths[key], label=f'α={key}')
    plt.title(f'UCB2 Cumulative Actual Regret')
    plt.legend()
    plt.show()
    for key in opt_pull_paths:
        plt.plot(X, opt_pull_paths[key], label=f'α={key}')
    plt.title(f'Most Arm Played')
    plt.legend()
    plt.show()