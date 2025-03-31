import System
import UCB_Tuned
import numpy as np
import matplotlib.pyplot as plt
import UCB_2
import UCB_Normal
import Epsilon_Greedy

# binary armed bandit
def collect(system:System.System, algorithm, Total_steps=5, *args, **kwargs):
    '''
    Collects the result of the algorithm on the system
    :return: tuple, with the first element being the best arm played percentage and the second element being the regret
    '''
    if algorithm == "UCB":
        algorithm = UCB_Tuned.UCB_Tuned(system)
    elif algorithm == "UCB_2":
        algorithm = UCB_2.UCB_2(system)
    elif algorithm == "UCB_Normal":
        algorithm = UCB_Normal.UCB_NORMAL(system)
    elif algorithm == "Epsilon_Greedy":
        algorithm = Epsilon_Greedy.Epsilon_Greedy(system, 0.15)
    else:
        raise ValueError("Invalid algorithm name")
    X = np.linspace(0, Total_steps, 1000)
    Y = []
    Z = []
    for i in X:
        times = int(np.pow(10,i))
        y,z = algorithm.run(times, process=False, result=False)[1:]
        Y.append(y)
        Z.append(z)
    return Y,Z


    


def plot(system:System.System, Total_steps = 5):
    plt.grid()
    X = np.linspace(0, Total_steps, 1000)
    Y,Z1 = collect(system, "UCB", Total_steps)
    plt.plot(X, Y, label="UCB")
    Y,Z2 = collect(system, "UCB_2", Total_steps)
    plt.plot(X, Y, label="UCB_2")
    Y,Z3 = collect(system, "UCB_Normal", Total_steps)
    plt.plot(X, Y, label="UCB_Normal")
    Y,Z4 = collect(system, "Epsilon_Greedy", Total_steps)
    plt.plot(X, Y, label="Epsilon_Greedy")
    plt.legend()
    plt.show()
    plt.grid()
    plt.plot(X, Z1, label="UCB")
    plt.plot(X, Z2, label="UCB_2")
    plt.plot(X, Z3, label="UCB_Normal")
    plt.plot(X, Z4, label="Epsilon_Greedy")
    plt.legend()
    plt.show()



plot(System.System(2, [0.9, 0.6]))
plot(System.System(2,[0.9, 0.8]))
plot(System.System(2,[0.55, 0.45]))
plot(System.System(10, [0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]))
plot(System.System(10, [0.9, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6]))
plot(System.System(10, [0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]))
plot(System.System(10, [0.55, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]))