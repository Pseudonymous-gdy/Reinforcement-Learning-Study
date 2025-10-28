import os
import sys
import math
# Allow running this file directly: ensure repo root is on sys.path
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Optional, Tuple, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
from Fusing.env import Environment
from Fusing.strategy import ProbabilisticStrategy

# directly investigating the behavior of the strategy
class Plot:
    def __init__(self, env: Environment):
        self.env = env
        self.alpha = 0.5  # Probability to choose reward-UCB branch
        self.strategy = ProbabilisticStrategy(env, alpha=self.alpha, seed=env.seed)

    def set_alpha(self, alpha: float):
        self.alpha = alpha
        self.strategy.alpha = alpha
    
    def update_with_new_alpha(self, alpha: float):
        self.set_alpha(alpha)
        self.strategy = ProbabilisticStrategy(self.env, alpha=self.alpha, seed=self.env.seed)
    
    def simulate(self):
        count = 0
        candidate_numbers = []
        while len(self.strategy.candidates) > 1 and count < 100000:
            self.strategy.step()
            count += 1
            candidate_numbers.append(len(self.strategy.candidates))
            if count % 20000 == 19999:
                print(f"Alpha: {self.alpha}, Step: {count}, Candidates left: {len(self.strategy.candidates)}")
        return candidate_numbers
    

plt.figure()
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    all_runs = []
    max_len = 0
    num_seeds = 20
    for seed in range(num_seeds):
        plotter = Plot(Environment(30, seed=seed))
        plotter.update_with_new_alpha(alpha)
        run = plotter.simulate()
        all_runs.append(run)
        if len(run) > max_len:
            max_len = len(run)
    
    # Pad runs to the same length and sum them up
    summed_steps = np.zeros(max_len)
    for run in all_runs:
        padded_run = np.pad(run, (0, max_len - len(run)), 'constant', constant_values=run[-1] if run else 0)
        summed_steps += padded_run
        
    avg_steps = summed_steps / num_seeds
    plt.plot(avg_steps, label=f"alpha={alpha}")
plt.xlabel("Number of steps")
plt.ylabel("Average number of candidates")
plt.title("Number of candidates over time")
plt.legend()
plt.show()
