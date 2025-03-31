import System
import random
import math
import numpy as np

class Epsilon_Greedy():
    def __init__(self, environment: System.System, c):
        self.environment = environment
        self.c = c
        self.n = environment.n
        self.p = environment.p
        # we let d be the difference between the maximum
        #  and second maximum probability of the arms in the system
        self.d = np.max(self.p) - np.sort(self.p)[-2]
        self.rewards = np.zeros(self.n)
        self.counts = np.zeros(self.n)
        self.time = 0
        self.epsilon = 1
        self.optimal = np.argmax(self.p)
        self.regret = 0

    def select_arm(self):
        if self.time == 0:
            return random.randint(0, self.n-1)
        self.epsilon = min(1, self.c*self.n/(self.d**2*self.time))
        if random.random() < self.epsilon:
            return random.randint(0, self.n-1)
        else:
            return np.argmax(self.rewards/self.counts)
    
    def simulate(self, arm, process=False):
        arm = self.select_arm()
        reward = self.environment.simulate(arm)
        self.rewards[arm] += reward
        self.counts[arm] += 1
        self.time += 1
        self.regret += self.p[self.optimal] - self.p[arm]
        if process:
            print("Arm:", arm+1, "Reward:", reward)
        return reward
    
    def run(self, steps, process=False, result=False):
        for i in range(steps):
            self.simulate(i, process=process)
        m = np.argmax(self.rewards/self.counts)
        a = self.regret
        self.regret = 0
        if result:
            print(self.rewards/self.counts)
        return m+1, self.counts[self.optimal]/self.counts.sum(), a

if __name__=='__main__':
    # Test
    n = 10
    p = [0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.87]
    system = System.System(n, p)
    eg = Epsilon_Greedy(system, 0.05)
    print(eg.run(1000, process=False))