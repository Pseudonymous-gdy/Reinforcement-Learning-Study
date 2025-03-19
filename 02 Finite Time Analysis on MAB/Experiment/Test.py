import System
import UCB_Tuned
import numpy as np
import matplotlib.pyplot as plt
import UCB_2
import UCB_Normal
import Epsilon_Greedy

# binary armed bandit
environment = System.System(2, [0.9, 0.6])
def plot(environment, optimal=0):
    ucb = UCB_Tuned.UCB_Tuned(environment)
    X = []
    Y = []
    for i in np.linspace(np.log10(2), 3, 1000):
        X.append(i)
        num = int(np.pow(10, i))
        Y.append(ucb.run(num, process=False, result=False, optimal=optimal)[1])
    ucb = UCB_Normal.UCB_NORMAL(environment)
    plt.plot(X, Y, label="UCB_Tuned")
    X = []
    Y = []
    for i in np.linspace(np.log10(2), 3, 1000):
        X.append(i)
        num = int(np.pow(10, i))
        Y.append(ucb.run(num, process=False, result=False, optimal=optimal)[1])
    plt.plot(X, Y, label="UCB_Normal")
    ep = Epsilon_Greedy.Epsilon_Greedy(environment, 0.1, 0.1)
    X = []
    Y = []
    for i in np.linspace(np.log10(2), 3, 1000):
        X.append(i)
        num = int(np.pow(10, i))
        Y.append(ep.run(num, process=False, result=False, optimal=optimal)[1])
    plt.plot(X, Y, label="Epsilon_Greedy")
    ucb = UCB_2.UCB_2(environment, 0.01)
    X = []
    Y = []
    for i in np.linspace(np.log10(2), 3, 1000):
        X.append(i)
        num = int(np.pow(10, i))
        Y.append(ucb.run(num, process=False, result=False, optimal=optimal)[1])
    plt.plot(X, Y, label="UCB_2")
    plt.legend()
    plt.show()

plot(environment, optimal=1)
plot(System.System(2,[0.9, 0.8]), optimal=1)
plot(System.System(2,[0.55, 0.45]), optimal=1)
plot(System.System(10, [0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]), optimal=1)
plot(System.System(10, [0.9, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6]), optimal=1)
plot(System.System(10, [0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),  optimal=1)
plot(System.System(10, [0.55, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]), optimal=1)