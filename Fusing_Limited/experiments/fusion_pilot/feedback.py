import numpy as np

def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def sample_reward(i, X, theta_star, rng):
    mean = float(X[i] @ theta_star)
    y = mean + rng.normal(0, 1)
    return y


def sample_duel(i, j, X, theta_star, rng):
    u = float((X[i] - X[j]) @ theta_star)
    p = sigmoid(u)
    y = rng.binomial(1, p)
    return y
