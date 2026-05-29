import numpy as np

def make_unit_instance():
    d = 10
    K = 10
    X = np.eye(d)
    theta = np.zeros(d)
    theta[0] = 0.0
    theta[1] = -0.35
    theta[2] = -0.45
    theta[3:] = -1.5
    i_star = int(np.argmax(X @ theta))
    return X, theta, i_star, {"K": K, "d": d}


def make_special_instance():
    d = 10
    e1 = np.zeros(d); e1[0] = 1.0
    e2 = np.zeros(d); e2[1] = 1.0
    e3 = np.zeros(d); e3[2] = 1.0
    x_d1 = np.cos(0.55) * e1 + np.sin(0.55) * e2
    x_d2 = np.cos(0.65) * e1 + np.sin(0.65) * e3
    X = np.vstack([e1, e2, e3, np.zeros((d-3, d))])
    # Fill remaining rows as standard basis for simplicity
    for i in range(3, d):
        X[i, i] = 1.0
    # replace second and third arms with rotated ones
    X[1] = x_d1
    X[2] = x_d2
    theta = np.zeros(d); theta[0] = 1.0
    i_star = int(np.argmax(X @ theta))
    return X, theta, i_star, {"K": d+2 if False else d, "d": d}


def make_general_instance(seed, K=20, d=5, rng=None, max_attempts=10000):
    if rng is None:
        rng = np.random.RandomState(seed)
    attempts = 0
    while attempts < max_attempts:
        attempts += 1
        X = rng.normal(size=(K, d))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / norms
        theta = rng.normal(size=(d,))
        theta = theta / np.linalg.norm(theta)
        scores = X @ theta
        i_star = int(np.argmax(scores))
        gaps = (X[i_star] - X) @ theta
        gaps[i_star] = np.inf
        delta_min = np.min(gaps)
        count_near = int(np.sum(gaps <= 0.35))
        if 0.12 <= delta_min <= 0.25 and count_near >= 3:
            meta = {"K": K, "d": d, "attempts": attempts}
            return X, theta, i_star, meta
    raise RuntimeError("Failed to generate general instance within max_attempts")
