import numpy as np
import time
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


def run_phased_mixed_xy_bai(X, theta_star, delta, regime, seed, config=None, progress_emitter=None):
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
        eps_m = 1.0 / np.sqrt(m + 3)
        eps_m = max(eps_m, 1e-6)

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
        # compute candidate phase size components
        cand1 = r_zeta
        cand2 = (1 + zeta) * C_safe * bar_x_m * L
        # avoid dividing by exact zero eps_m although eps_m formula is preserved
        denom = (eps_m ** 2)
        cand3 = (1 + zeta) * 128 * np.e ** 3 * bar_x_m * T / denom if denom != 0 else np.inf
        raw_nm = max(cand1, cand2, cand3)
        # numeric guards: if raw_nm is not finite or extremely large, cap to a safe maximum
        MAX_NM = 50000
        if not np.isfinite(raw_nm):
            n_m = int(MAX_NM)
            capped = True
        else:
            if raw_nm > MAX_NM:
                n_m = int(MAX_NM)
                capped = True
            else:
                n_m = int(np.ceil(raw_nm))
                capped = False

        if capped:
            print(f"Warning: n_m capped to {n_m} (raw={raw_nm}) to avoid excessive allocation")

        # deterministic rounding by counts (avoid building huge seq in memory)
        q_arr = np.asarray(q_m, dtype=float)
        tilde = n_m * q_arr
        n_a = np.floor(tilde).astype(int)
        rem = int(n_m - np.sum(n_a))
        if rem > 0:
            frac = tilde - np.floor(tilde)
            idx = np.argsort(-frac)
            for k in range(min(rem, len(idx))):
                n_a[idx[k]] += 1
        seq = None
        # execute by action counts (batch sampling)
        for a_idx, count in enumerate(n_a):
            if count <= 0:
                continue
            a = actions[a_idx]
            if a[0] == 'R':
                ys = rng.normal(loc=(X[a[1]] @ theta_star), scale=1.0, size=count)
                # extend observations in batch
                observations.extend([{'rho': 'R', 'v': X[a[1]], 'y': float(y)} for y in ys])
                T_r_main += int(count)
            else:
                # duel sampling: binomial wins
                p = 1.0 / (1.0 + np.exp(-float((X[a[1]] - X[a[2]]) @ theta_star)))
                wins = rng.binomial(n=int(count), p=p)
                # store duel observations as wins/individuals
                # approximate by appending wins times success and failures
                if wins > 0:
                    observations.extend([{'rho': 'D', 'v': X[a[1]] - X[a[2]], 'y': 1.0}] * int(wins))
                if count - wins > 0:
                    observations.extend([{'rho': 'D', 'v': X[a[1]] - X[a[2]], 'y': 0.0}] * int(count - wins))
                T_d_main += int(count)
            total_samples += int(count)

        # stopping check
        theta_hat, success, status, obj = fit_joint_mle(observations, d, theta_init=theta_hat, ridge=ridge_lambda)
        # emit phase_end event if emitter provided
        if progress_emitter is not None:
            try:
                evt = {
                    'timestamp': time.time(),
                    'event': 'phase_end',
                    'case': None,
                    'regime': regime,
                    'delta': delta,
                    'seed_id': seed,
                    'phase': m,
                    'T_r': T_r_burn + T_r_main,
                    'T_d': T_d_burn + T_d_main,
                    'T_total': T_r_burn + T_r_main + T_d_burn + T_d_main,
                    'n_m': n_m,
                    'epsilon_m': eps_m,
                    'bar_x_m': bar_x_m,
                    'q_reward_mass': float(np.sum([q for (q, a) in zip(q_m, actions) if a[0] == 'R'])) if len(q_m)>0 else 0.0,
                    'q_duel_mass': float(np.sum([q for (q, a) in zip(q_m, actions) if a[0] == 'D'])) if len(q_m)>0 else 0.0,
                    'p_D_so_far': (T_d_burn + T_d_main) / max(1, (T_r_burn + T_r_main + T_d_burn + T_d_main)),
                    'mle_time_sec': 0.0,
                    'design_time_sec': 0.0,
                    'sampling_time_sec': 0.0,
                    'stopping_time_sec': 0.0,
                    'stop_stat': obj,
                    'best_arm_hat': int(np.argmax(X @ theta_hat)),
                    'elapsed_sec': 0.0,
                }
                progress_emitter(evt)
            except Exception:
                pass
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
