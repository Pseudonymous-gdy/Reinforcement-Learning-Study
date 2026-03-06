import os
import sys
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Fusing2.env import Environment
from Fusing2.FSH import FusionSequentialHalving_Final
from Fusing2.combined_sequential_halving import CombinedSequentialHalving

env = Environment(number_of_bandits=32, distribution="bernoulli", seed=42)
env.set_means([0.9] + [0.8]*31)

# Pure Dueling via FSH
algo = FusionSequentialHalving_Final(env=env, total_budget=10000, zeta=0.0, alpha=0.5, seed=42)
res = algo.run()
print("FSH Pure Dueling res:", res, algo.spent_budget)

# Pure Reward via FSH
algo2 = FusionSequentialHalving_Final(env=env, total_budget=10000, zeta=1.0, alpha=0.0, seed=42)
res2 = algo2.run()
print("FSH Pure Reward res:", res2, algo2.spent_budget)

# Pure Dueling via CSH
algo3 = CombinedSequentialHalving(env=env, total_budget=10000, alpha=0.0, seed=42)
algo3.run()
print("CSH Pure Dueling res:", algo3.get_candidate_set(), algo3.spent_budget)

# Pure Reward via CSH
algo4 = CombinedSequentialHalving(env=env, total_budget=10000, alpha=1.0, seed=42)
algo4.run()
print("CSH Pure Reward res:", algo4.get_candidate_set(), algo4.spent_budget)
