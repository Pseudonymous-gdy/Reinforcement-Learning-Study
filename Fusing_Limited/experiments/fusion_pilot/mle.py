import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize


def logaddexp0(u):
    # log(1+e^u) stable
    return np.log1p(np.exp(u))


def fit_joint_mle_from_stats(action_vs, action_types, N_a, S_a, d, theta_init=None, ridge=1e-6, tol=1e-6, maxiter=200):
    """
    Vectorized joint MLE from aggregated sufficient statistics.

    action_vs: array (n_actions, d)
    action_types: list of 'R' or 'D'
    N_a: array of counts per action
    S_a: array of sum_y per action (for duel: number of wins)
    """
    n_actions = action_vs.shape[0]
    if theta_init is None:
        theta0 = np.zeros(d)
    else:
        theta0 = theta_init.copy()

    action_vs = np.asarray(action_vs)
    N_a = np.asarray(N_a, dtype=float)
    S_a = np.asarray(S_a, dtype=float)

    # precompute indices
    idx_R = np.array([i for i, t in enumerate(action_types) if t == 'R'], dtype=int)
    idx_D = np.array([i for i, t in enumerate(action_types) if t == 'D'], dtype=int)

    def fun_and_grad(theta_vec):
        u = action_vs.dot(theta_vec)  # n_actions

        # rewards
        ll = 0.0
        grad = np.zeros_like(theta_vec)

        if idx_R.size > 0:
            uR = u[idx_R]
            SR = S_a[idx_R]
            NR = N_a[idx_R]
            # ll contribution: SR * uR - 0.5 * NR * uR^2
            ll += np.sum(SR * uR - 0.5 * NR * (uR ** 2))
            # grad: sum (SR - NR * uR) * v
            resR = SR - NR * uR
            grad += action_vs[idx_R].T.dot(resR)

        if idx_D.size > 0:
            uD = u[idx_D]
            SD = S_a[idx_D]
            ND = N_a[idx_D]
            # ll: SD * uD - ND * log(1+e^{uD})
            # use stable log1p(exp)
            ll += np.sum(SD * uD - ND * logaddexp0(uD))
            # grad: sum (SD - ND * sigma(uD)) * v
            sigma_uD = 1.0 / (1.0 + np.exp(-uD))
            resD = SD - ND * sigma_uD
            grad += action_vs[idx_D].T.dot(resD)

        # ridge regularization (numerical)
        ll -= 0.5 * ridge * (theta_vec @ theta_vec)
        grad -= ridge * theta_vec

        neg_ll = -float(ll)
        neg_grad = -grad
        return neg_ll, neg_grad

    def obj(theta_vec):
        f, _ = fun_and_grad(theta_vec)
        return f

    def jac(theta_vec):
        _, g = fun_and_grad(theta_vec)
        return g

    res = minimize(fun=fun_and_grad, x0=theta0, method='L-BFGS-B', jac=True, options={'maxiter': maxiter, 'ftol': tol})
    theta_hat = res.x
    status = res.status
    success = res.success
    final_obj = res.fun
    return theta_hat, success, status, final_obj


def fit_joint_mle(observations, d, theta_init=None, ridge=1e-6, tol=1e-6):
    """
    Backwards-compatible wrapper (keeps previous behavior if given list-of-dict observations).
    """
    if isinstance(observations, dict) and {'action_vs', 'action_types', 'N_a', 'S_a'}.issubset(observations.keys()):
        return fit_joint_mle_from_stats(observations['action_vs'], observations['action_types'], observations['N_a'], observations['S_a'], d, theta_init=theta_init, ridge=ridge, tol=tol)

    # otherwise, original list-of-dict behavior
    if theta_init is None:
        theta = np.zeros(d)
    else:
        theta = theta_init.copy()

    def neglog(theta_vec):
        ll = 0.0
        for obs in observations:
            v = obs['v']
            u = float(v @ theta_vec)
            if obs['rho'] == 'R':
                # Gaussian: y*u - u^2/2
                ll += obs['y'] * u - 0.5 * u * u
            else:
                # Bernoulli logistic: y*u - log(1+e^u)
                ll += obs['y'] * u - logaddexp0(u)
        # regularization (subtract because we maximize)
        ll -= 0.5 * ridge * (theta_vec @ theta_vec)
        return -ll

    res = minimize(neglog, theta, method='L-BFGS-B', tol=tol)
    theta_hat = res.x
    status = res.status
    success = res.success
    final_obj = res.fun
    return theta_hat, success, status, final_obj
