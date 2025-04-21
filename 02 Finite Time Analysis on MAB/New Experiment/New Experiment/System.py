import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bernoulli


class System:
    # This clas is used for creating a system with n arms, each with a bernoulli reward distributions
    def __init__(self, n: int, p: list[float]):
        self.n = n  # number of samples
        self.p = p  # probability of reward

    def simulate(self, times, seed=42):
        rng = np.random.default_rng(seed)
        samples = []
        for i in range(self.n):
            sample = rng.binomial(n=1, p=self.p[i], size=times)
            samples.append(sample)
        return samples

class UCB:
    '''
        A BASIC CLASS FOR GENERAL UCB ALGORITHM
        This class is used for developing UCB algorithms.
        Input:
            environment (System): the environment to conduct MAB
            rewards (list): the rewards of each arm
            ucb (list): the upper confidence bound of each arm
            counts (list): the number of times each arm has been selected
    '''
    def __init__(self, environment: System):
        self.environment = environment
        self.n = environment.n
        self.p = environment.p
        self.rewards = np.zeros(self.n)
        self.ucb = np.inf * np.ones(self.n)
        self.counts = np.zeros(self.n)
        self.optimal = np.argmax(self.p)  # the optimal arm
        self.regret = 0

    def select_arm(self):
        '''
        Policy Selection: select the arm with the highest UCB value
        '''
        return np.argmax(self.ucb)

    def simulate(self, times):
        regret_lst = 0
        most_arm_played = 0
        for seed in range(100):
            table = self.environment.simulate(times=times, seed=seed)
            for time in range(times):
                arm = self.select_arm()
                reward = table[arm][time]
                self.rewards[arm] += reward
                self.counts[arm] += 1
                for arm1 in range(self.n):
                    self.ucb[arm] = self.rewards[arm] / self.counts[arm] + self.UCB(arm)
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
            self.regret = 0
        return regret_lst/100, most_arm_played/100

    def UCB(self, arm):
        # a virtual function to be implemented in the child class
        # the following is a typical UCB reward function
        return np.sqrt(2 * np.log(np.sum(self.counts)) / self.counts[arm])

if __name__=='__main__':
    n = 10
    p = [0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.99]
    system = System(n, p)
    print(np.shape(system.simulate(1000))) # 1
    ucb = UCB(system)
    print(ucb.simulate(10000))