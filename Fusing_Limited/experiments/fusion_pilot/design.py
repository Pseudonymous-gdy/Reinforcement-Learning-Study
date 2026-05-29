import numpy as np

def actions_from_X(X, regime):
    K = X.shape[0]
    actions = []
    if regime == 'reward_only' or regime == 'reward-only' or regime == 'reward':
        for i in range(K):
            actions.append(('R', i))
    elif regime == 'duel_only' or regime == 'duel-only' or regime == 'duel':
        for i in range(K):
            for j in range(K):
                if i == j: continue
                actions.append(('D', i, j))
    else:
        # fusion: include both reward and duel
        for i in range(K):
            actions.append(('R', i))
        for i in range(K):
            for j in range(K):
                if i == j: continue
                actions.append(('D', i, j))
    return actions


def compute_Jm(a, X, theta_hat):
    # return placeholder Jm value and direction v
    if a[0] == 'R':
        v = X[a[1]]
    else:
        v = X[a[1]] - X[a[2]]
    # placeholder scalar objective
    Jm_value = float(np.linalg.norm(v))
    return Jm_value, v


def solve_design(actions, X, theta_hat, Y_id, bar_x_m, eps_m, zeta, C_safe):
    # heuristic design: weight actions by their vector norm (prefer informative directions)
    if len(actions) == 0:
        return [], 'no_actions'
    norms = []
    for a in actions:
        if a[0] == 'R':
            v = X[a[1]]
        else:
            v = X[a[1]] - X[a[2]]
        norms.append(float(np.linalg.norm(v)))
    norms = np.array(norms, dtype=float)
    # avoid zeros
    norms += 1e-8
    q = norms / norms.sum()
    return q.tolist(), 'norm_heuristic'


def B_from_q(actions, X, theta_hat, q):
    d = X.shape[1]
    B = np.zeros((d, d))
    for idx, a in enumerate(actions):
        if a[0] == 'R':
            v = X[a[1]]
        else:
            v = X[a[1]] - X[a[2]]
        weight = float(q[idx]) if q is not None and len(q) > idx else 0.0
        B += weight * np.outer(v, v)
    # regularize slightly to avoid singular and limit pseudoinverse magnitude
    B += 1e-3 * np.eye(d)
    return B
