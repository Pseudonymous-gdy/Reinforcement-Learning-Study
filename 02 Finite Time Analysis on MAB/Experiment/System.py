import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

class System:
    # This clas is used for creating a system with n arms, each with a bernoulli reward distributions
    def __init__(self, n:int, p:list[float]):
        self.n = n # number of samples
        self.p = p # probability of reward
    
    def simulate(self, arm):
        # get the reward of the arm
        i = random.random()
        if i < self.p[arm]:
            return 1
        else:
            return 0

class UCB:
    def __init__(self, environment: System):
        self.environment = environment
        self.n = environment.n
        self.p = environment.p
        self.rewards = np.zeros(self.n)
        self.ucb = np.inf*np.ones(self.n)
        self.counts = np.zeros(self.n)
    
    def select_arm(self):
        return np.argmax(self.ucb)
    
    def simulate(self, arm, process=True):
        reward = self.environment.simulate(arm)
        self.rewards[arm] += reward
        self.counts[arm] += 1
        self.ucb[arm] = self.rewards[arm]/self.counts[arm] + self.UCB(arm)
        if process is True:
            print("Arm:", arm+1, "Reward:", reward)

    def UCB(self, arm):
        # a virtual function to be implemented in the child class
        # the following is a typical UCB reward function
        return np.sqrt(2*np.log(np.sum(self.counts))/self.counts[arm])
    
    def run(self, T, process=True, result=True, optimal=0):
        for i in range(T):
            arm = self.select_arm()
            self.simulate(arm, process=process)
        if result is True:
            print("Reward of Arms:", self.rewards/self.counts)
            print("Optimal Arm:", np.argmax(self.rewards/self.counts)+1)
        m = np.argmax(self.rewards/self.counts)
        if optimal==0:
            return m+1, self.counts[m]/self.counts.sum()
        else:
            return m+1, self.counts[optimal-1]/self.counts.sum()


if __name__=='__main__':
    n = 10
    p = [0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.99]
    system = System(n, p)
    print(system.simulate(int(input())-1)) # 1
    ucb = UCB(system)
    ucb.run(1000, process=False, result=True, optimal=10)