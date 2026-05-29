import numpy as np


def deterministic_round(n_m, q, rng=None):
    # q sums to 1
    tilde = n_m * q
    n = np.floor(tilde).astype(int)
    rem = int(n_m - np.sum(n))
    frac = tilde - np.floor(tilde)
    idx = np.argsort(-frac)
    for k in range(rem):
        n[idx[k]] += 1
    seq = []
    for i, ni in enumerate(n):
        seq += [i] * int(ni)
    if rng is not None:
        rng.shuffle(seq)
    return np.array(n), np.array(seq)
