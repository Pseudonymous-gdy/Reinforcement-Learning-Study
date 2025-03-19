import System
import random
import math
import numpy as np

class Epsilon_Greedy():
    def __init__(self, environment: System.System, c, d):
        self.environment = environment
        self.c = c
        self.d = d
        self.n = environment.n
        self.p = environment.p
        self.rewards = np.zeros(self.n)
        self.counts = np.zeros(self.n)
        self.time = 0
        self.epsilon = 1

    def select_arm(self):
        if self.time == 0:
            return random.randint(0, self.n-1)
        self.epsilon = min(1, self.c*self.n/(self.d**2*self.time))
        if random.random() < self.epsilon:
            return random.randint(0, self.n-1)
        else:
            return np.argmax(self.rewards/self.counts)
    
    def simulate(self, arm, process=True):
        arm = self.select_arm()
        reward = self.environment.simulate(arm)
        self.rewards[arm] += reward
        self.counts[arm] += 1
        self.time += 1
        if process:
            print("Arm:", arm+1, "Reward:", reward)
        return reward
    
    def run(self, steps, process=True, result=True, optimal=0):
        for i in range(steps):
            self.simulate(i, process=process)
        m = np.argmax(self.rewards/self.counts)
        if optimal==0:
            return m+1, self.counts[m]/self.counts.sum()
        else:
            return m+1, self.counts[optimal-1]/self.counts.sum()

if __name__=='__main__':
    # Test
    n = 10
    p = [0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.87]
    system = System.System(n, p)
    eg = Epsilon_Greedy(system, 0.1, 0.1)
    eg.run(1000, process=False, optimal=5)