import numpy as np


def deterministic_round(n_m, q, rng=None, return_seq=False):
    # q sums to 1
    tilde = n_m * q
    n = np.floor(tilde).astype(int)
    rem = int(n_m - np.sum(n))
    if rem > 0:
        frac = tilde - np.floor(tilde)
        idx = np.argsort(-frac)
        for k in range(min(rem, len(idx))):
            n[idx[k]] += 1
    if return_seq:
        seq = []
        for i, ni in enumerate(n):
            seq += [i] * int(ni)
        if rng is not None:
            rng.shuffle(seq)
        return np.array(n), np.array(seq)
    return np.array(n)
