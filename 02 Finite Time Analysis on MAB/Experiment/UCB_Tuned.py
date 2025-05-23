import numpy as np
import random
import System

class UCB_Tuned(System.UCB):
    def __init__(self, environment: System.System):
        self.environment = environment
        self.n = environment.n
        self.p = environment.p
        self.rewards = np.zeros(self.n)
        self.ucb = np.inf*np.ones(self.n)
        self.counts = np.zeros(self.n)
        self.reward_square = np.zeros(self.n)
        self.optimal = np.argmax(self.p)
        self.regret = 0
    
    def simulate(self, arm, process=False):
        reward = self.environment.simulate(arm)
        self.reward_square[arm] += reward**2
        self.rewards[arm] += reward
        self.counts[arm] += 1
        self.ucb[arm] = self.rewards[arm]/self.counts[arm] + self.UCB(arm)
        self.regret += self.p[self.optimal] - self.p[arm]
        if process is True:
            print("Arm:", arm+1, "Reward:", reward)

    
    def UCB(self, arm):
        # override the previous UCB function
        variance = self.reward_square[arm]/self.counts[arm] - (self.rewards[arm]/self.counts[arm])**2 + np.sqrt(2*np.log(np.sum(self.counts))/self.counts[arm])
        return np.sqrt(2*np.log(np.sum(self.counts))/self.counts[arm]*np.min([1/4, variance]))
    
if __name__=='__main__':
    n = 10
    p = [0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.87]
    system = System.System(n, p)
    ucb = UCB_Tuned(system)
    print(ucb.run(1000, process=False)) # 5