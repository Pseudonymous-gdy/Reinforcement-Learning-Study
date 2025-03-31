import System
import numpy as np
import math

class UCB_2(System.UCB):
    def __init__(self, environment: System.System, ALPHA=1):
        self.environment = environment
        self.n = environment.n
        self.p = environment.p
        self.ALPHA = ALPHA
        self.rewards = np.zeros(self.n)
        self.ucb = np.inf*np.ones(self.n)
        self.counts = np.zeros(self.n)
        self.optimal = np.argmax(self.p)
        self.regret = 0
    
    def Tau(self, arm):
        try:
            m = int(np.power(1+self.ALPHA, self.counts[arm]))+1
        except:
            m = 10000
        return m
    
    def UCB(self, arm):
        x = float(self.Tau(arm))
        a = (1+self.ALPHA)*math.log(np.e*float(self.counts.sum())/x, np.e) / (2*x)
        return np.sqrt(a) 
    
    def run(self, T, process=False, result=True):
        i = self.n
        for j in range(self.n):
            self.simulate(j, process=process)
        while i < T:
            arm = self.select_arm()
            for j in range(max(int(self.ALPHA*self.Tau(arm)+1), 1)):
                self.simulate(arm, process=process)
                i += 1
                if i >= T:
                    break
        if result is True:
            print("Reward of Arms:", self.rewards/self.counts)
            print("Optimal Arm:", np.argmax(self.rewards/self.counts)+1)
        m = np.argmax(self.rewards/self.counts)
        a = self.regret
        self.regret = 0
        return m+1, self.counts[self.optimal-1]/self.counts.sum(),a

if __name__=='__main__':
    n=10
    p = [0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.99]
    system = System.System(n, p)
    ucb = UCB_2(system,0.01)
    print(ucb.run(1000, process=False, result=True))