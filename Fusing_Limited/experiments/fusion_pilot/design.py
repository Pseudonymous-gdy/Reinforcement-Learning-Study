import numpy as np
from scipy.optimize import minimize
from .feedback import sigmoid


def actions_from_X(X, regime):
    """
    Build canonical action sets according to regime.
    regime: one of 'reward-only', 'duel-only', 'fusion'.
    Duel actions are canonical with i < j.
    """
    K = X.shape[0]
    actions = []
    if regime == 'reward-only':
        for i in range(K):
            actions.append(('R', i))
    elif regime == 'duel-only':
        for i in range(K):
            for j in range(i + 1, K):
                actions.append(('D', i, j))
    elif regime == 'fusion':
        for i in range(K):
            actions.append(('R', i))
        for i in range(K):
            for j in range(i + 1, K):
                actions.append(('D', i, j))
    else:
        raise ValueError(f"Unknown regime: {regime}")
    return actions


def fisher_increment(action, X, theta_hat):
    """
    Return Fisher information increment J, measurement vector v, and weight.
    For reward: J = x x^T (weight=1).
    For duel: J = w * v v^T where w = sigma(u)*(1-sigma(u)).
    """
    if action[0] == 'R':
        i = action[1]
        v = X[i]
        weight = 1.0
    else:
        _, i, j = action
        v = X[i] - X[j]
        u = float(v @ theta_hat)
        p = sigmoid(u)
        weight = float(p * (1.0 - p))
    J = weight * np.outer(v, v)
    return J, v, weight


def compute_Jm(action, X, theta_hat):
    return fisher_increment(action, X, theta_hat)


def B_from_q(actions, X, theta_hat, q):
    """Compute B(q)=sum_a q(a) J_m(a) without adding large ridge."""
    d = X.shape[1]
    B = np.zeros((d, d))
    for idx, a in enumerate(actions):
        J, _, _ = fisher_increment(a, X, theta_hat)
        weight = float(q[idx]) if q is not None and len(q) > idx else 0.0
        B += weight * J
    # tiny numerical regularizer only
    B += 1e-12 * np.eye(d)
    return B


def design_objective_and_stats(q, actions, X, theta_hat, Y_id, bar_x_m, eps_m, C_safe, rcond=1e-10):
    """
    Compute design objective and diagnostics for given q.
    Returns (obj, stats) where stats include L, T, rank_B, min_eig_B, cond_B.
    """
    d = X.shape[1]
    B = B_from_q(actions, X, theta_hat, q)
    # diagnostics on B
    try:
        eigs = np.linalg.eigvalsh(B)
        min_eig = float(np.min(eigs))
        max_eig = float(np.max(eigs))
        cond_B = float(max_eig / max(min_eig, 1e-30))
        rank_B = int(np.sum(eigs > 1e-12))
    except Exception:
        min_eig = float('nan')
        cond_B = float('nan')
        rank_B = 0

    Bpinv = np.linalg.pinv(B, rcond=rcond)

    # build X_mix: reward vectors and raw duel vectors
    X_mix = []
    for a in actions:
        if a[0] == 'R':
            _, i = a
            X_mix.append(X[i])
        else:
            _, i, j = a
            v = X[i] - X[j]
            X_mix.append(v)

    # compute L_m(q)
    L = 0.0
    if len(X_mix) > 0:
        for v in X_mix:
            L = max(L, float(v @ (Bpinv @ v)))

    # compute T_m(q) over identifiable targets Y_id
    T = 0.0
    if len(Y_id) > 0:
        for y in Y_id:
            T = max(T, float(y @ (Bpinv @ y)))

    if not np.isfinite(L) or not np.isfinite(T):
        objective = 1e30
    else:
        objective = max(C_safe * bar_x_m * L, 128 * np.e ** 3 * bar_x_m * T / (eps_m ** 2))
    stats = dict(L=L, T=T, objective=objective, rank_B=rank_B, min_eig_B=min_eig, cond_B=cond_B)
    return objective, stats


def solve_design(actions, X, theta_hat, Y_id, bar_x_m, eps_m, zeta, C_safe, solver='greedy-fw', previous_q=None, max_iters=50):
    """
    Solve design optimization over simplex. Default: greedy Frank-Wolfe style.
    Returns q (list) and status string.
    """
    n = len(actions)
    if n == 0:
        return [], 'no_actions'

    # precompute J_list
    J_list = []
    for a in actions:
        J, _, _ = fisher_increment(a, X, theta_hat)
        J_list.append(J)

    if previous_q is None:
        q = np.ones(n, dtype=float) / n
    else:
        q = np.asarray(previous_q, dtype=float)
        if q.size != n:
            q = np.ones(n, dtype=float) / n

    if solver == 'slsqp':
        # constrained minimization over simplex
        def obj(q_var):
            # keep objective defined for SLSQP; return large value if outside simplex
            if np.any(q_var < -1e-8):
                return 1e30
            s = np.sum(q_var)
            if not np.isfinite(s) or abs(s - 1.0) > 1e-4:
                return 1e30
            val, _ = design_objective_and_stats(q_var, actions, X, theta_hat, Y_id, bar_x_m, eps_m, C_safe)
            if not np.isfinite(val):
                return 1e30
            return float(val)

        cons = ({'type': 'eq', 'fun': lambda qvar: np.sum(qvar) - 1.0},)
        bounds = [(0.0, 1.0) for _ in range(n)]
        res = minimize(obj, q, bounds=bounds, constraints=cons, method='SLSQP', options={'maxiter': 200, 'ftol': 1e-8})
        if res.success:
            q_opt = np.maximum(res.x, 0.0)
            if np.sum(q_opt) <= 0:
                q_opt = np.ones(n, dtype=float) / n
            else:
                q_opt = q_opt / np.sum(q_opt)
            qo = q_opt.tolist()
            status = 'slsqp:ok'
            return qo, status
        # fallback to greedy-fw if SLSQP fails
        q_fallback, status_fb = solve_design(actions, X, theta_hat, Y_id, bar_x_m, eps_m, zeta, C_safe, solver='greedy-fw', previous_q=q, max_iters=max_iters)
        return q_fallback, f'slsqp:fail->{status_fb}'

    # default greedy-FW style solver
    best_q = q.copy()
    best_val, _ = design_objective_and_stats(best_q, actions, X, theta_hat, Y_id, bar_x_m, eps_m, C_safe)
    for t in range(max_iters):
        B = B_from_q(actions, X, theta_hat, q)
        Bpinv = np.linalg.pinv(B, rcond=1e-10)

        # find worst direction across X_mix and Y_id
        worst_val = -np.inf
        worst_vec = None

        # X_mix uses reward vectors and raw duel vectors
        for a in actions:
            if a[0] == 'R':
                _, i = a
                v = X[i]
                val = C_safe * bar_x_m * float(v @ (Bpinv @ v))
                if val > worst_val:
                    worst_val = val
                    worst_vec = v
            else:
                _, i, j = a
                v = X[i] - X[j]
                val = C_safe * bar_x_m * float(v @ (Bpinv @ v))
                if val > worst_val:
                    worst_val = val
                    worst_vec = v

        # Y_id for target term
        for y in Y_id:
            val = 128 * np.e ** 3 * bar_x_m * float(y @ (Bpinv @ y)) / (eps_m ** 2)
            if val > worst_val:
                worst_val = val
                worst_vec = y

        r = worst_vec
        if r is None:
            break

        # compute gains for each action
        gains = np.zeros(n, dtype=float)
        for idx in range(n):
            J = J_list[idx]
            # gain ~ r^T Bpinv J Bpinv r
            tmp = Bpinv @ r
            gains[idx] = float(tmp @ (J @ tmp))

        a_star = int(np.argmax(gains))
        eta = 2.0 / (t + 2.0)
        q = (1.0 - eta) * q
        q[a_star] += eta

        val, _ = design_objective_and_stats(q, actions, X, theta_hat, Y_id, bar_x_m, eps_m, C_safe)
        if val < best_val:
            best_val = val
            best_q = q.copy()

    # normalize
    if np.sum(best_q) <= 0:
        best_q = np.ones(n, dtype=float) / n
    else:
        best_q = best_q / np.sum(best_q)

    return best_q.tolist(), 'greedy-fw'
