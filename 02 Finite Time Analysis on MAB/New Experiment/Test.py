import System
import UCB_Tuned
import numpy as np
import matplotlib.pyplot as plt


envs = [System.System(2, [0.9, 0.6]),
        System.System(2, [0.9, 0.8]),
        System.System(2, [0.55, 0.45]),
        System.System(10, [0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]),
        System.System(10, [0.9, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6]),
        System.System(10, [0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),
        System.System(10, [0.55, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45])]

for env in envs:
    X = np.linspace(0, 4, 100)
    model = UCB_Tuned.UCB_Tuned(env)
    Y1 = []
    Y2 = []
    for i in range(len(X)):
        a, b = model.simulate(int(10**X[i]))
        Y1.append(a)
        Y2.append(b)
        print('\r', '*' * int(20 * (10 ** X[i]) / 10000) + '-' * (20 - int(20 * (10 ** X[i]) / 10000)), '|', f'Times:{int(10**X[i])}/100000', f'Regret: {a}', f'Most arm played: {b}', end='')
    plt.plot(X, Y1, label='Cumulative Actual Regret')
    plt.title(f'UCB Tuned Cumulative Actual Regret for the environment No.{envs.index(env)}')
    plt.legend()
    plt.show()
    plt.plot(X, Y2, label='Most arm played')
    plt.title(f'UCB Tuned Most Arm Played for the environment No.{envs.index(env)}')
    plt.legend()
    plt.show()