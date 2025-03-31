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
        self.ucb = np.inf*np.ones(self.n)
        self.counts = np.zeros(self.n)
        self.optimal = np.argmax(self.p) # the optimal arm
        self.regret = 0
    
    def select_arm(self):
        '''
        Policy Selection: select the arm with the highest UCB value
        '''
        return np.argmax(self.ucb)
    
    def simulate(self, arm, process=True):
        '''
        Simulate the reward of the arm
        Input:
            arm (int): the arm to be simulated
            process (bool): whether to print the process
        Return: None
        '''
        reward = self.environment.simulate(arm)
        self.rewards[arm] += reward
        self.counts[arm] += 1
        self.ucb[arm] = self.rewards[arm]/self.counts[arm] + self.UCB(arm)
        self.regret += self.environment.p[self.optimal] - self.rewards[arm]/self.counts[arm]
        if process is True:
            print("Arm:", arm+1, "Reward:", reward)

    def UCB(self, arm):
        # a virtual function to be implemented in the child class
        # the following is a typical UCB reward function
        return np.sqrt(2*np.log(np.sum(self.counts))/self.counts[arm])
    
    def run(self, T, process=False, result=True):
        '''
        Run the UCB algorithm for T rounds.
        Input:
            T (int): the number of rounds
            process (bool): whether to print the process
            result (bool): whether to print the result
        Return: None
        '''
        for i in range(T):
            arm = self.select_arm()
            self.simulate(arm, process=process)
        if result is True:
            print("Reward of Arms:", self.rewards/self.counts)
            print("Optimal Arm:", np.argmax(self.rewards/self.counts)+1)
        m = np.argmax(self.rewards/self.counts)
        a = self.regret
        self.regret = 0
        return m+1, self.counts[self.optimal]/self.counts.sum(), a


if __name__=='__main__':
    n = 10
    p = [0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.99]
    system = System(n, p)
    print(system.simulate(int(input())-1)) # 1
    ucb = UCB(system)
    print(ucb.run(1000, process=False, result=False))