import numpy as np

# Fixed parameters
sigma_R = 1.0
gamma = 1.0
omega_D = 1
omega_R_values = [1, 2, 4, 8, 16]
delta_values = [0.1, 0.05, 0.01]
seed_count = 500
T_max = 100000

# Algorithmic constants
burnin_repeats = 3
ridge_lambda = 1e-6
projection_tol = 1e-7
zeta = 0.1
r_zeta = None  # set at runtime to d**2 if None
C_safe = 32 * np.e ** 3 * (1 + zeta)
