import numpy as np
import random
import System


class UCB_Tuned(System.UCB):
    def __init__(self, environment: System.System):
        self.environment = environment
        self.n = environment.n
        self.p = environment.p
        self.rewards = np.zeros(self.n)
        self.ucb = np.inf * np.ones(self.n)
        self.counts = np.zeros(self.n)
        self.reward_square = np.zeros(self.n)
        self.optimal = np.argmax(self.p)
        self.regret = 0

    def simulate(self, times):
        regret_lst = 0
        most_arm_played = 0
        for seed in range(100):
            table = self.environment.simulate(times=times, seed=seed)
            for time in range(times):
                arm = self.select_arm()
                reward = table[arm][time]
                self.reward_square[arm] += reward ** 2
                self.rewards[arm] += reward
                self.counts[arm] += 1
                for arm1 in range(self.n):
                    self.ucb[arm1] = self.rewards[arm1] / self.counts[arm1] + self.UCB(arm1)
                self.regret += table[self.optimal][time] - reward
            regret_lst += self.regret
            most_arm_played += self.counts[self.optimal]/times
            # if seed == 99:
            #     a = self.rewards/self.counts
            #     print(a)
            #     print(np.argmax(a))
            self.rewards = np.zeros(self.n)
            self.ucb = np.inf * np.ones(self.n)
            self.counts = np.zeros(self.n)
            self.reward_square = np.zeros(self.n)
            self.regret = 0
        return regret_lst/100, most_arm_played/100

    def UCB(self, arm):
        # override the previous UCB function
        variance = self.reward_square[arm] / self.counts[arm] - (self.rewards[arm] / self.counts[arm]) ** 2 + np.sqrt(
            2 * np.log(np.sum(self.counts)) / self.counts[arm])
        return np.sqrt(2 * np.log(np.sum(self.counts)) / self.counts[arm] * np.min([1 / 4, variance]))

if __name__ == '__main__':
    n = 10
    p = [0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.99]
    system = System.System(n, p)
    print(np.shape(system.simulate(1000)))  # 1
    ucb = UCB_Tuned(system)
    print(ucb.simulate(10000))