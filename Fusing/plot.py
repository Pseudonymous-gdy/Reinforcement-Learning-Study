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
        while len(self.strategy.candidates) > 1 and count < 200000:
            self.strategy.step()
            count += 1
            candidate_numbers.append(len(self.strategy.candidates))
            if count % 20000 == 19999:
                print(f"Alpha: {self.alpha}, Step: {count}, Candidates left: {len(self.strategy.candidates)}")
        return candidate_numbers
    
def plot_alpha_effect(mean_data, index):
    plt.figure()
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        all_runs = []
        answers = {}
        max_len = 0
        num_seeds = 10
        candidates = 0
        for seed in range(num_seeds):
            plotter = Plot(Environment(15, seed=seed))
            plotter.env.set_means(mean_data[index])
            plotter.update_with_new_alpha(alpha)
            run = plotter.simulate()
            all_runs.append(run)
            if len(run) > max_len:
                max_len = len(run)
            answers[alpha] = answers.get(alpha, [])
            answers[alpha].append(np.array(plotter.strategy.candidates).argmax() == plotter.env.bandit_means.argmax())
            candidates += len(plotter.strategy.candidates)
        with open(f"./Fusing/record.txt", "a") as f:
            f.write(f"Index: {index}, Alpha: {alpha}, Correct: {sum(answers[alpha]) / num_seeds}, Candidates Left: {candidates/num_seeds}\n")
        print(f"Alpha: {alpha}, Correct: {sum(answers[alpha]) / num_seeds}")
        
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
    plt.savefig(f"./Fusing/alpha_effect_plot_{index}.png")
    plt.close()

data = np.array([
        [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2], # first set of means, 15 arms with equally spaced means
        [0.95, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2], # second set of means, a better top arm
        [0.95, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15], # third set of means, a notably better top arm
        [0.9, 0.88, 0.86, 0.84, 0.82, 0.8, 0.78, 0.76, 0.74, 0.72, 0.7, 0.68, 0.66, 0.64, 0.62],  # fourth set of means, closely spaced arms
        [0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.8, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73],   # fifth set of means, very closely spaced arms
        [0.87, 0.86, 0.84, 0.83, 0.79, 0.63, 0.6, 0.58, 0.55, 0.53, 0.5, 0.28, 0.15, 0.13, 0.04]    # sixth set of means, mixed spacing
    ])

def test1():
    for i in range(len(data)):
        # shuffle the means to avoid any positional bias
        shuffled_means = np.random.permutation(data[i])
        data[i] = shuffled_means
        plot_alpha_effect(data, i)

class Plot_with_Altered_Alpha(Plot):
    def __init__(self, env: Environment, alpha: float, decay):
        super().__init__(env=env)
        self.set_alpha(alpha)
        if decay is None:
            self.decay = lambda self, alpha: alpha
        else:
            self.decay = decay

    def simulate_with_decay(self):
        count = 0
        candidate_numbers = []
        while len(self.strategy.candidates) > 1 and count < 200000:
            self.strategy.alpha = self.decay(self, self.alpha)
            self.strategy.step()
            count += 1
            candidate_numbers.append(len(self.strategy.candidates))
            if count % 20000 == 19999:
                print(f"Alpha: {self.alpha}, Step: {count}, Candidates left: {len(self.strategy.candidates)}")
        return candidate_numbers

def decay1(model: Plot_with_Altered_Alpha, alpha):
    return max(0.0, alpha - 0.1)

def decay2(model: Plot_with_Altered_Alpha, alpha):
    return alpha * 0.9

def decay3(model: Plot_with_Altered_Alpha, alpha, threshold = 20000):
    a = model.strategy.reward_pulls + model.strategy.dueling_counts.sum()
    return alpha if a < threshold else 0.0

def decay4(model: Plot_with_Altered_Alpha, alpha):
    threshold = 10000
    return decay3(model, alpha, threshold)

def decay5(model: Plot_with_Altered_Alpha, alpha):
    threshold = 5000
    return decay3(model, alpha, threshold)

def plot_alpha_altered(mean_data, index, decay):
    plt.figure()
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        all_runs = []
        answers = {}
        max_len = 0
        num_seeds = 10
        candidates = 0
        for seed in range(num_seeds):
            plotter = Plot_with_Altered_Alpha(Environment(15, seed=seed), alpha, decay)
            plotter.env.set_means(mean_data[index])
            run = plotter.simulate_with_decay()
            all_runs.append(run)
            if len(run) > max_len:
                max_len = len(run)
            answers[alpha] = answers.get(alpha, [])
            answers[alpha].append(np.array(plotter.strategy.candidates).argmax() == plotter.env.bandit_means.argmax())
            candidates += len(plotter.strategy.candidates)
        with open(f"./Fusing/record.txt", "a") as f:
            f.write(f"Index: {index}, Decay:{decay.__name__}, Alpha: {alpha}, Correct: {sum(answers[alpha]) / num_seeds}, Candidates Left: {candidates/num_seeds}\n")
        print(f"Alpha: {alpha}, Correct: {sum(answers[alpha]) / num_seeds}")

        
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
    plt.savefig(f"./Fusing/alpha_effect_plot_{index}_with_{decay.__name__}.png")
    plt.close()

def test2(decay):
    for i in range(len(data)):
        shuffled_means = np.random.permutation(data[i])
        data[i] = shuffled_means
        plot_alpha_altered(data, i, decay)

if __name__ == "__main__":
    test2(decay3)
    test2(decay4)
    test2(decay5)