import numpy as np
from .instances import make_unit_instance, make_special_instance, make_general_instance
from .feedback import sample_reward, sample_duel, sigmoid
from .mle import fit_joint_mle
from .design import actions_from_X, compute_Jm, solve_design, B_from_q
from .rounding import deterministic_round
from .config import burnin_repeats, ridge_lambda, projection_tol, zeta, C_safe

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def build_action_directions(X, regime):
    actions = actions_from_X(X, regime)
    V = []
    for a in actions:
        if a[0] == 'R':
            V.append(X[a[1]])
        else:
            V.append(X[a[1]] - X[a[2]])
    V = np.vstack(V) if len(V) > 0 else np.zeros((0, X.shape[1]))
    return actions, V


def run_phased_mixed_xy_bai(X, theta_star, delta, regime, seed, config=None):
    rng = np.random.RandomState(seed)
    K, d = X.shape[0], X.shape[1]
    actions, V = build_action_directions(X, regime)
    # determine Y_id: all differences x_i - x_j projected to span(V)
    Ys = []
    for i in range(K):
        for j in range(K):
            if i == j: continue
            y = X[i] - X[j]
            Ys.append(y)
    Ys = np.vstack(Ys)
    # identify those in span(V)
    if V.shape[0] == 0:
        Y_id = []
    else:
        U, s, _ = np.linalg.svd(V.T, full_matrices=False)
        rank = np.sum(s > projection_tol)
        basis = U[:, :rank]
        proj = basis @ (basis.T @ Ys.T)
        resid = Ys.T - proj
        norms = np.linalg.norm(resid, axis=0)
        Y_id = [Ys[i] for i in range(Ys.shape[0]) if norms[i] <= projection_tol]
    identifiable_ratio = len(Y_id) / max(1, Ys.shape[0])

    # burn-in: pick actions that increase rank
    observations = []
    selected_dirs = []
    dir_mat = np.zeros((0, d))
    for a_idx, a in enumerate(actions):
        if len(selected_dirs) >= d:
            break
        if a[0] == 'R':
            v = X[a[1]]
        else:
            v = X[a[1]] - X[a[2]]
        # check if adds to rank
        if dir_mat.shape[0] == 0:
            add = True
        else:
            M = np.vstack([dir_mat, v])
            if np.linalg.matrix_rank(M) > np.linalg.matrix_rank(dir_mat):
                add = True
            else:
                add = False
        if add:
            dir_mat = np.vstack([dir_mat, v]) if dir_mat.shape[0] > 0 else v.reshape(1, -1)
            selected_dirs.append(a)
    T_r_burn = 0
    T_d_burn = 0
    for a in selected_dirs:
        for _ in range(burnin_repeats):
            if a[0] == 'R':
                y = sample_reward(a[1], X, theta_star, rng)
                observations.append({'rho': 'R', 'v': X[a[1]], 'y': y})
                T_r_burn += 1
            else:
                y = sample_duel(a[1], a[2], X, theta_star, rng)
                observations.append({'rho': 'D', 'v': X[a[1]] - X[a[2]], 'y': y})
                T_d_burn += 1

    total_samples = T_r_burn + T_d_burn
    T_r_main = 0
    T_d_main = 0

    theta_hat = np.zeros(d)
    mle_status = None
    final_phase = 0
    m = 1
    # Fixed-confidence loop: run until the stopping rule is satisfied
    while True:
        # phase params
        delta_m = 6 * delta / (np.pi ** 2 * m ** 2)
        bar_x_m = d * np.log(5) + np.log((np.pi ** 2 * m ** 2) / (3 * delta))
        eps_m = 2 ** (-m)

        # fit MLE
        theta_hat, success, status, obj = fit_joint_mle(observations, d, theta_init=theta_hat, ridge=ridge_lambda)
        mle_status = status

        # design
        q_m, design_status = solve_design(actions, X, theta_hat, Y_id, bar_x_m, eps_m, zeta, C_safe)
        if len(q_m) == 0:
            break
        # phase size
        # approximate L and T
        L, T = 0.0, 0.0
        B = B_from_q(actions, X, theta_hat, q_m)
        Bpinv = np.linalg.pinv(B, rcond=1e-8)
        for a in actions:
            _, v = compute_Jm(a, X, theta_hat)
            L = max(L, float(v @ (Bpinv @ v)))
        for y in Y_id:
            T = max(T, float(y @ (Bpinv @ y)))
        if d is None:
            raise RuntimeError
        r_zeta = d ** 2
        n_m = int(np.ceil(max(r_zeta, (1 + zeta) * C_safe * bar_x_m * L, (1 + zeta) * 128 * np.e ** 3 * bar_x_m * T / (eps_m ** 2))))
        # do not cap n_m by an artificial budget; allow large runs for fixed-confidence setting

        # rounding
        n_a, seq = deterministic_round(n_m, q_m, rng)
        # execute sequence with progress
        if tqdm is not None:
            seq_iter = tqdm(seq, desc=f'Phase {m} exec', leave=False)
        else:
            seq_iter = seq
        for idx in seq_iter:
            a = actions[idx]
            if a[0] == 'R':
                y = sample_reward(a[1], X, theta_star, rng)
                observations.append({'rho': 'R', 'v': X[a[1]], 'y': y})
                T_r_main += 1
            else:
                y = sample_duel(a[1], a[2], X, theta_star, rng)
                observations.append({'rho': 'D', 'v': X[a[1]] - X[a[2]], 'y': y})
                T_d_main += 1
            total_samples += 1

        # stopping check
        theta_hat, success, status, obj = fit_joint_mle(observations, d, theta_init=theta_hat, ridge=ridge_lambda)
        # empirical fisher
        H = np.zeros((d, d))
        for obs in observations:
            v = obs['v']
            if obs['rho'] == 'R':
                H += np.outer(v, v)
            else:
                u = float(v @ theta_hat)
                w = sigmoid(u) * (1 - sigmoid(u))
                H += w * np.outer(v, v)
        Hpinv = np.linalg.pinv(H, rcond=1e-8)
        scores = X @ theta_hat
        i_hat = int(np.argmax(scores))
        beta_hat = 4 * np.e ** (3 / 2) * np.sqrt(2 * bar_x_m)
        ok = True
        for j in range(K):
            if j == i_hat: continue
            y = X[i_hat] - X[j]
            lhs = float(y @ theta_hat)
            rhs = beta_hat * np.sqrt(float(y @ (Hpinv @ y)))
            if lhs <= rhs:
                ok = False
                break
        final_phase = m
        print(f"Phase {m} finished: total_samples={total_samples}, n_m={n_m}, design_status={design_status}, mle_status={mle_status}")
        if ok:
            break
        m += 1

    T_r = T_r_burn + T_r_main
    T_d = T_d_burn + T_d_main
    result = dict(
        i_hat=i_hat,
        i_star=int(np.argmax(X @ theta_star)),
        success=(int(np.argmax(X @ theta_star)) == i_hat),
        error=False,
        error_type="",
        error_message="",
        T_r=T_r,
        T_d=T_d,
        T_total=T_r + T_d,
        T_r_burn=T_r_burn,
        T_d_burn=T_d_burn,
        T_r_main=T_r_main,
        T_d_main=T_d_main,
        final_phase=final_phase,
        final_loglik=obj,
        mle_status=mle_status,
        design_solver_status=design_status,
        identifiable_ratio=identifiable_ratio,
        p_D=(T_d / max(1, T_r + T_d)) if (T_r + T_d) > 0 else 0.0,
    )
    return result
