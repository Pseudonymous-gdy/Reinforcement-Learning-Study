import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize


def logaddexp0(u):
    # log(1+e^u)
    return np.log1p(np.exp(u))


def fit_joint_mle(observations, d, theta_init=None, ridge=1e-6, tol=1e-6):
    # observations: list of dicts with keys rho ('R' or 'D'), v (vector), y (scalar)
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
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize


def logaddexp0(u):
    # log(1+e^u)
    return np.log1p(np.exp(u))


def fit_joint_mle(observations, d, theta_init=None, ridge=1e-6, tol=1e-6):
    # observations: list of dicts with keys rho ('R' or 'D'), v (vector), y (scalar)
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
