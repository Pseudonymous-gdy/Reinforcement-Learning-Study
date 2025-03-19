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
    
    def run(self, T, process=True, result=True, optimal=0):
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
        if optimal==0:
            return m+1, self.counts[m]/self.counts.sum()
        else:
            return m+1, self.counts[optimal-1]/self.counts.sum()

if __name__=='__main__':
    n=10
    p = [0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.5, 0.7, 0.99]
    system = System.System(n, p)
    ucb = UCB_2(system,0.01)
    ucb.run(10000, process=False, result=True, optimal=1)