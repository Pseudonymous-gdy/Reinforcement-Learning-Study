import System
import numpy as np
import random

class UCB_NORMAL(System.UCB):
    def __init__(self, environment: System.System):
        self.environment = environment
        self.n = environment.n
        self.p = environment.p
        self.rewards = np.zeros(self.n)
        self.ucb = np.zeros(self.n)
        self.counts = np.zeros(self.n)
        self.reward_square = np.zeros(self.n)

    def select_arm(self):
        for time in range(len(self.counts)):
            if self.counts[time] <= 1+int(np.log(np.sum(self.counts)+1)):
                return time
        return np.argmax(self.ucb)

    def simulate(self, arm, process=True):
        reward = self.environment.simulate(arm)
        self.reward_square[arm] += reward**2
        self.rewards[arm] += reward
        self.counts[arm] += 1
        self.ucb[arm] = self.rewards[arm]/self.counts[arm] + self.UCB(arm)
        if process is True:
            print("Arm:", arm+1, "Reward:", reward)

    def UCB(self, arm):
        return np.sqrt(16*(self.reward_square[arm]-self.rewards[arm]**2/self.counts[arm])/(self.counts[arm]-1)*np.log(self.counts.sum())/np.log(np.e)/self.counts[arm])

if __name__=='__main__':
    n = 10
    p = [0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.87]
    system = System.System(n, p)
    ucb = UCB_NORMAL(system)
    ucb.run(10000, process=False) # 5