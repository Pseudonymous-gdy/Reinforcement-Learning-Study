import numpy as np
from scipy.optimize import minimize


def compute_Jm(action, X, theta_hat):
    # action: ('R', i) or ('D', i, j)
    if action[0] == 'R':
        v = X[action[1]]
        return np.outer(v, v), v
    else:
        i, j = action[1], action[2]
        v = X[i] - X[j]
        # w(u) = sigma(u)(1-sigma(u))
        u = float(v @ theta_hat)
        sig = 1.0 / (1.0 + np.exp(-u))
        w = sig * (1 - sig)
        return w * np.outer(v, v), v


def actions_from_X(X, regime):
    K = X.shape[0]
    actions = []
    for i in range(K):
        if regime in ('reward_only', 'fusion'):
            actions.append(('R', i))
    for i in range(K):
        for j in range(i + 1, K):
            if regime in ('duel_only', 'fusion'):
                actions.append(('D', i, j))
    return actions


def B_from_q(actions, X, theta_hat, q):
    d = X.shape[1]
    B = np.zeros((d, d))
    for a, qa in zip(actions, q):
        J, _ = compute_Jm(a, X, theta_hat)
        B += qa * J
    return B


def leverage_and_target(actions, X, theta_hat, q, Y_id):
    B = B_from_q(actions, X, theta_hat, q)
    Bpinv = np.linalg.pinv(B, rcond=1e-8)
    L = 0.0
    for a in actions:
        _, v = compute_Jm(a, X, theta_hat)
        val = float(v @ (Bpinv @ v))
        if val > L:
            L = val
    T = 0.0
    for y in Y_id:
        val = float(y @ (Bpinv @ y))
        if val > T:
            T = val
    return L, T


def solve_design(actions, X, theta_hat, Y_id, bar_x_m, eps_m, zeta, C_safe):
    # minimize max{C_safe*bar_x_m*L(q), 128 e^3 bar_x_m T(q)/eps^2}
    nA = len(actions)
    if nA == 0:
        return np.array([]), 'empty'
    x0 = np.ones(nA) / nA

    def objective(q):
        q = np.maximum(q, 1e-12)
        q = q / np.sum(q)
        L, T = leverage_and_target(actions, X, theta_hat, q, Y_id)
        v1 = C_safe * bar_x_m * L
        v2 = 128 * np.e ** 3 * bar_x_m * T / (eps_m ** 2)
        return max(v1, v2)

    cons = ({'type': 'eq', 'fun': lambda q: np.sum(q) - 1.0},)
    bnds = [(0.0, 1.0) for _ in range(nA)]
    res = minimize(objective, x0, bounds=bnds, constraints=cons, method='SLSQP', options={'maxiter': 200})
    if res.success:
        q = np.maximum(res.x, 0.0)
        q = q / np.sum(q)
        status = 'slsqp'
    else:
        # fallback greedy: start uniform and iteratively add mass
        q = x0.copy()
        status = 'fallback'
        for _ in range(500):
            best_improve = 0.0
            best_idx = None
            cur_obj = objective(q)
            for i in range(nA):
                q2 = q.copy()
                q2[i] += 1e-3
                q2 = np.maximum(q2, 0)
                q2 = q2 / np.sum(q2)
                val = objective(q2)
                improve = cur_obj - val
                if improve > best_improve:
                    best_improve = improve
                    best_idx = i
            if best_idx is None:
                break
            q[best_idx] += 1e-3
            q = q / np.sum(q)
    return q, status
