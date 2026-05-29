import numpy as np
import time
from .instances import make_unit_instance, make_special_instance, make_general_instance
from .feedback import sample_reward, sample_duel, sigmoid
from .mle import fit_joint_mle
from .design import actions_from_X, compute_Jm, solve_design, B_from_q, design_objective_and_stats, fisher_increment
from .rounding import deterministic_round
from .config import burnin_repeats, ridge_lambda, projection_tol, zeta, C_safe, T_max

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


def run_phased_mixed_xy_bai(X, theta_star, delta, regime, seed, config=None, progress_emitter=None):
    """
    Phased joint-MLE mixed-XY BAI algorithm (strict implementation).
    """
    rng = np.random.default_rng(seed)
    K, d = X.shape[0], X.shape[1]

    # config options
    design_solver = None
    debug_cap_n_m = None
    verbose = False
    t_max = T_max
    if isinstance(config, dict):
        design_solver = config.get('design_solver', None)
        debug_cap_n_m = config.get('debug_cap_n_m', None)
        verbose = bool(config.get('verbose', False))
        if 'T_max' in config:
            t_max = config.get('T_max')

    # build actions and measurement vectors
    actions = actions_from_X(X, regime)
    n_actions = len(actions)
    action_vs = np.zeros((n_actions, d))
    action_types = []
    for idx, a in enumerate(actions):
        if a[0] == 'R':
            action_vs[idx] = X[a[1]]
            action_types.append('R')
        else:
            action_vs[idx] = X[a[1]] - X[a[2]]
            action_types.append('D')

    # build all pairwise differences Y (for identifiability and targets)
    Ys = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            Ys.append(X[i] - X[j])
    Ys = np.vstack(Ys)

    # sufficient statistics per action
    N_a = np.zeros(n_actions, dtype=float)
    S_a = np.zeros(n_actions, dtype=float)

    # burn-in: select up to d actions that expand span
    selected_idx = []
    dir_mat = np.zeros((0, d))
    for idx in range(n_actions):
        if len(selected_idx) >= d:
            break
        v = action_vs[idx]
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
            selected_idx.append(idx)

    T_r_burn = 0
    T_d_burn = 0
    for idx in selected_idx:
        for _ in range(burnin_repeats):
            if action_types[idx] == 'R':
                mean = float(action_vs[idx] @ theta_star)
                y = rng.normal(loc=mean)
                S_a[idx] += float(y)
                N_a[idx] += 1.0
                T_r_burn += 1
            else:
                p = sigmoid(float(action_vs[idx] @ theta_star))
                y = rng.binomial(1, p)
                S_a[idx] += float(y)
                N_a[idx] += 1.0
                T_d_burn += 1

    total_samples = T_r_burn + T_d_burn
    T_r_main = 0
    T_d_main = 0

    theta_hat = np.zeros(d)
    mle_status = None
    mle_success = False
    final_phase = 0
    m = 1
    i_hat = int(np.argmax(X @ theta_hat))
    stopped = True
    stop_fail = False

    # precompute Y_id function (depends on span of X_mix which changes with theta_hat)
    def compute_identifiable_targets(action_vs_local, theta_local):
        # build basis from X_mix (reward vectors and raw duel vectors)
        V = []
        for t, v in zip(action_types, action_vs_local):
            if t == 'R':
                V.append(v)
            else:
                V.append(v)
        V = np.vstack(V) if len(V) > 0 else np.zeros((0, d))
        if V.shape[0] == 0:
            return []
        U, s, _ = np.linalg.svd(V.T, full_matrices=False)
        rank = int(np.sum(s > projection_tol))
        if rank == 0:
            return []
        basis = U[:, :rank]
        proj = basis @ (basis.T @ Ys.T)
        resid = Ys.T - proj
        norms = np.linalg.norm(resid, axis=0)
        Y_id = [Ys[i] for i in range(Ys.shape[0]) if norms[i] <= projection_tol]
        return Y_id

    # Fixed-confidence loop
    while True:
        # phase params
        delta_m = 6 * delta / (np.pi ** 2 * m ** 2)
        bar_x_m = d * np.log(5) + np.log((np.pi ** 2 * m ** 2) / (3 * delta))
        eps_m = 2.0 ** (-m)

        # fit MLE from sufficient stats
        t0 = time.time()
        mle_input = {'action_vs': action_vs, 'action_types': action_types, 'N_a': N_a, 'S_a': S_a}
        theta_hat, success, status, obj = fit_joint_mle(mle_input, d, theta_init=theta_hat, ridge=ridge_lambda)
        mle_time_sec = time.time() - t0
        mle_status = status
        mle_success = bool(success)

        # compute identifiable targets under current theta_hat
        Y_id = compute_identifiable_targets(action_vs, theta_hat)
        identifiable_ratio = len(Y_id) / max(1, Ys.shape[0])

        # design solver
        t1 = time.time()
        q_m, design_status = solve_design(actions, X, theta_hat, Y_id, bar_x_m, eps_m, zeta, C_safe, solver=(design_solver or 'greedy-fw'))
        design_time_sec = time.time() - t1
        if len(q_m) == 0:
            break

        # compute B and diagnostics
        B = B_from_q(actions, X, theta_hat, q_m)
        Bpinv = np.linalg.pinv(B, rcond=1e-10)
        # compute L_m (over X_mix) and T_m (over Y_id)
        L = 0.0
        # X_mix uses raw duel vectors
        for t, v in zip(action_types, action_vs):
            L = max(L, float(v @ (Bpinv @ v)))
        T = 0.0
        for y in Y_id:
            T = max(T, float(y @ (Bpinv @ y)))

        r_zeta = d ** 2
        raw_nm = max(r_zeta, (1 + zeta) * C_safe * bar_x_m * L, (1 + zeta) * 128 * np.e ** 3 * bar_x_m * T / (eps_m ** 2))

        # debug cap only if provided in config
        if debug_cap_n_m is not None:
            if not np.isfinite(raw_nm):
                n_m = int(debug_cap_n_m)
            else:
                n_m = int(min(np.ceil(raw_nm), debug_cap_n_m))
        else:
            if not np.isfinite(raw_nm):
                # propagate error instead of capping silently
                raise RuntimeError(f"Non-finite phase size raw_nm={raw_nm}")
            n_m = int(np.ceil(raw_nm))

        # enforce T_max if configured
        n_m_exec = n_m
        forced_stop = False
        if t_max is not None:
            remaining = int(t_max - total_samples)
            if remaining <= 0:
                n_m_exec = 0
                forced_stop = True
            elif total_samples + n_m > t_max:
                n_m_exec = remaining
                forced_stop = True

        # rounding -> get counts only
        q_arr = np.asarray(q_m, dtype=float)
        n_a = deterministic_round(n_m_exec, q_arr, rng=rng, return_seq=False)
        rounding_l1 = float(np.sum(np.abs(n_a / max(1, n_m_exec) - q_arr)))

        # batch sampling per action
        t2 = time.time()
        for a_idx, count in enumerate(n_a):
            if count <= 0:
                continue
            v = action_vs[a_idx]
            if action_types[a_idx] == 'R':
                ys = rng.normal(loc=float(v @ theta_star), scale=1.0, size=int(count))
                S_a[a_idx] += float(np.sum(ys))
                N_a[a_idx] += float(count)
                T_r_main += int(count)
            else:
                p = sigmoid(float(v @ theta_star))
                wins = int(rng.binomial(n=int(count), p=p))
                S_a[a_idx] += float(wins)
                N_a[a_idx] += float(count)
                T_d_main += int(count)
            total_samples += int(count)
        sampling_time_sec = time.time() - t2

        # fit MLE again after sampling and evaluate stopping
        t3 = time.time()
        mle_input = {'action_vs': action_vs, 'action_types': action_types, 'N_a': N_a, 'S_a': S_a}
        theta_hat, success, status, obj = fit_joint_mle(mle_input, d, theta_init=theta_hat, ridge=ridge_lambda)
        mle_success = bool(success)

        # compute empirical Fisher H using current theta_hat
        H = np.zeros((d, d))
        for idx, a in enumerate(actions):
            J, _, _ = compute_Jm(a, X, theta_hat)
            H += N_a[idx] * J
        Hpinv = np.linalg.pinv(H, rcond=1e-10)

        # stopping rule
        scores = X @ theta_hat
        i_hat = int(np.argmax(scores))
        beta_hat = 4 * np.e ** (3 / 2) * np.sqrt(2 * bar_x_m)
        ok = True
        min_gap_hat = float('inf')
        max_radius = 0.0
        for j in range(K):
            if j == i_hat:
                continue
            y = X[i_hat] - X[j]
            lhs = float(y @ theta_hat)
            rhs = beta_hat * np.sqrt(float(y @ (Hpinv @ y)))
            min_gap_hat = min(min_gap_hat, lhs)
            max_radius = max(max_radius, rhs)
            if lhs <= rhs:
                ok = False
        final_phase = m
        stopping_time_sec = time.time() - t3

        # emit diagnostics
        if progress_emitter is not None:
            try:
                # B diagnostics
                eigs_B = np.linalg.eigvalsh(B)
                min_eig_B = float(np.min(eigs_B))
                cond_B = float(np.max(eigs_B) / max(min_eig_B, 1e-30))
                rank_B = int(np.sum(eigs_B > 1e-12))

                eigs_H = np.linalg.eigvalsh(H)
                min_eig_H = float(np.min(eigs_H))
                cond_H = float(np.max(eigs_H) / max(min_eig_H, 1e-30))
                rank_H = int(np.sum(eigs_H > 1e-12))

                obj_design = max((1 + zeta) * C_safe * bar_x_m * L, (1 + zeta) * 128 * np.e ** 3 * bar_x_m * T / (eps_m ** 2))

                evt = {
                    'timestamp': time.time(),
                    'event': 'phase_end',
                    'phase': m,
                    'epsilon_m': eps_m,
                    'delta_m': delta_m,
                    'bar_x_m': bar_x_m,
                    'n_m': n_m,
                    'T_r': int(T_r_burn + T_r_main),
                    'T_d': int(T_d_burn + T_d_main),
                    'T_total': int(T_r_burn + T_r_main + T_d_burn + T_d_main),
                    'q_reward_mass': float(np.sum([q for (q, a) in zip(q_m, actions) if a[0] == 'R'])) if len(q_m) > 0 else 0.0,
                    'q_duel_mass': float(np.sum([q for (q, a) in zip(q_m, actions) if a[0] == 'D'])) if len(q_m) > 0 else 0.0,
                    'L_m': float(L),
                    'T_m': float(T),
                    'design_objective': float(obj_design),
                    'rank_B': rank_B,
                    'min_eig_B': min_eig_B,
                    'cond_B': cond_B,
                    'rank_H': rank_H,
                    'min_eig_H': min_eig_H,
                    'cond_H': cond_H,
                    'rounding_l1': rounding_l1,
                    'i_hat': i_hat,
                    'min_empirical_gap_to_i_hat': float(min_gap_hat),
                    'max_conf_radius_to_i_hat': float(max_radius),
                    'stop_ratio': float(max_radius / max(min_gap_hat, 1e-12)),
                    'mle_status': mle_status,
                    'design_status': design_status,
                    'mle_time_sec': float(mle_time_sec),
                    'design_time_sec': float(design_time_sec),
                    'sampling_time_sec': float(sampling_time_sec),
                    'stopping_time_sec': float(stopping_time_sec),
                }
                progress_emitter(evt)
            except Exception:
                pass

        if verbose:
            print(f"Phase {m} finished: total_samples={total_samples}, n_m={n_m}, design_status={design_status}, mle_status={mle_status}")

        if forced_stop:
            stopped = False
            stop_fail = True
            break
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
        stopped=stopped,
        stop_fail=stop_fail,
        T_r=int(T_r),
        T_d=int(T_d),
        T_total=int(T_r + T_d),
        T_r_burn=int(T_r_burn),
        T_d_burn=int(T_d_burn),
        T_r_main=int(T_r_main),
        T_d_main=int(T_d_main),
        final_phase=final_phase,
        final_loglik=float(obj),
        mle_success=mle_success,
        mle_status=mle_status,
        design_solver_status=design_status,
        identifiable_ratio=identifiable_ratio if Ys.shape[0] > 0 else 0.0,
        p_D=(T_d / max(1, T_r + T_d)) if (T_r + T_d) > 0 else 0.0,
    )
    return result
