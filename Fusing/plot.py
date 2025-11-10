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
import multiprocessing

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

    def simulate_snapshot(self, record_interval: int = 1000, max_snapshots: Optional[int] = None):
        """Run simulation and return candidate counts and periodic snapshots of estimated values.

        Returns:
            candidate_numbers: list[int]
            snapshot_steps: list[int] -- the step numbers at which snapshots were taken
            snapshots: list[np.ndarray] -- list of arrays shape (K,) containing self.strategy.values copies
        """
        count = 0
        candidate_numbers = []
        snapshot_steps = []
        snapshots: List[np.ndarray] = []
        while len(self.strategy.candidates) > 1 and count < 200000:
            self.strategy.step()
            count += 1
            candidate_numbers.append(len(self.strategy.candidates))
            if record_interval and record_interval > 0 and count % record_interval == 0:
                snapshot_steps.append(count)
                snapshots.append(np.copy(self.strategy.values))
                if max_snapshots and len(snapshots) >= max_snapshots:
                    break
            if count % 20000 == 19999:
                print(f"Alpha: {self.alpha}, Step: {count}, Candidates left: {len(self.strategy.candidates)}")
        # ensure final snapshot included
        if (not snapshots) or (snapshot_steps and snapshot_steps[-1] != count):
            snapshot_steps.append(count)
            snapshots.append(np.copy(self.strategy.values))
        return candidate_numbers, snapshot_steps, snapshots
    
def plot_alpha_effect(mean_data, index, num_seeds: int = 10, compute_variance: bool = False, record_interval: int = 1000, max_snapshots: Optional[int] = None, plot_per_arm: int = 0):
    """Original alpha-effect plot with optional variance plotting.

    If compute_variance is True, the function will run per-seed snapshotting and
    produce an additional plot showing mean variance of estimated arm values over time.
    """
    plt.figure()

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        all_runs = []
        answers = []
        max_len = 0
        num_seeds = 10
        candidates = 0

        # prepare args for workers; we always collect candidate runs (per-seed)
        ctx = multiprocessing.get_context('spawn')
        args_list = [(mean_data[index], alpha, seed, None) for seed in range(num_seeds)]
        with ctx.Pool(processes=min(num_seeds, max(1, os.cpu_count() or 1))) as pool:
            results = pool.map(_worker_seed_run, args_list)

        for run, correct, cand in results:
            all_runs.append(run)
            answers.append(correct)
            candidates += cand
            if len(run) > max_len:
                max_len = len(run)

        with open(f"./Fusing/record.txt", "a", encoding='utf-8') as f:
            f.write(f"Index: {index}, Alpha: {alpha}, Correct: {sum(answers) / num_seeds}, Candidates Left: {candidates/num_seeds}\n")
        print(f"Alpha: {alpha}, Correct: {sum(answers) / num_seeds}")

        # Pad runs to the same length and sum them up
        summed_steps = np.zeros(max_len)
        for run in all_runs:
            padded_run = np.pad(run, (0, max_len - len(run)), 'constant', constant_values=run[-1] if run else 0)
            summed_steps += padded_run

        avg_steps = summed_steps / num_seeds
        plt.plot(avg_steps, label=f"alpha={alpha}")

        # If requested, compute per-step variance across runs and plot CI around mean
        if compute_variance:
            # Build a 2D array runs_matrix shape (num_seeds, max_len)
            runs_matrix = np.zeros((num_seeds, max_len))
            for i, run in enumerate(all_runs):
                padded = np.pad(run, (0, max_len - len(run)), 'constant', constant_values=run[-1] if run else 0)
                runs_matrix[i, :] = padded
            # variance across seeds for each step
            var_steps = np.var(runs_matrix, axis=0, ddof=0)
            # standard error of the mean
            sem = np.sqrt(var_steps / float(num_seeds))
            # 95% CI using z=1.96
            z = 1.96
            lower = avg_steps - z * sem
            upper = avg_steps + z * sem
            x = np.arange(len(avg_steps))
            plt.fill_between(x, lower, upper, alpha=0.3)
    plt.xlabel("Number of steps")
    plt.ylabel("Average number of candidates")
    plt.title("Number of candidates over time")
    plt.legend()
    plt.savefig(f"./Fusing/alpha_effect_plot_{index}.png")
    plt.close()

    # end plot_alpha_effect

data = np.array([
        [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2], # first set of means, 15 arms with equally spaced means
        [0.95, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2], # second set of means, a better top arm
        [0.95, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15], # third set of means, a notably better top arm
        [0.9, 0.88, 0.86, 0.84, 0.82, 0.8, 0.78, 0.76, 0.74, 0.72, 0.7, 0.68, 0.66, 0.64, 0.62],  # fourth set of means, closely spaced arms
        [0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81, 0.8, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73],   # fifth set of means, very closely spaced arms
        [0.87, 0.86, 0.84, 0.83, 0.79, 0.63, 0.6, 0.58, 0.55, 0.53, 0.5, 0.28, 0.15, 0.13, 0.04]    # sixth set of means, mixed spacing
    ])


def _worker_seed_run(args):
    """Top-level worker to run a single seed experiment (picklable)."""
    mean_vector, alpha, seed, decay = args
    env = Environment(len(mean_vector), seed=seed)
    env.set_means(mean_vector)
    if decay is None:
        plotter = Plot(env)
        plotter.update_with_new_alpha(alpha)
        run = plotter.simulate()
        correct = np.array(plotter.strategy.values).argmax() == plotter.env.bandit_means.argmax()
        candidates = len(plotter.strategy.candidates)
    else:
        plotter = Plot_with_Altered_Alpha(env, alpha, decay)
        run = plotter.simulate_with_decay()
        correct = np.array(plotter.strategy.values).argmax() == plotter.env.bandit_means.argmax()
        candidates = len(plotter.strategy.candidates)
    return (run, correct, candidates)


def test1():
    for i in range(len(data)):
        # shuffle the means to avoid any positional bias
        shuffled_means = np.random.permutation(data[i])
        data[i] = shuffled_means
        # produce original plot and variance plot
        plot_alpha_effect(data, i, num_seeds=10, compute_variance=True, record_interval=1000, max_snapshots=50, plot_per_arm=3)

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

def plot_alpha_altered(mean_data, index, decay, num_seeds: int = 10, compute_variance: bool = False):
    """Plot altered-alpha experiments; optionally overlay CI like plot_alpha_effect.

    Args:
        mean_data: array of mean vectors
        index: which mean vector index to run
        decay: decay function passed into Plot_with_Altered_Alpha
        num_seeds: number of seeds to average over
        compute_variance: whether to compute and overlay 95% CI (based on per-seed runs)
    """
    plt.figure()
    # Use pool-based parallelism similar to plot_alpha_effect but pass decay to worker

    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        all_runs = []
        answers = []
        max_len = 0
        candidates = 0

        args_list = [(mean_data[index], alpha, seed, decay) for seed in range(num_seeds)]
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=min(num_seeds, max(1, os.cpu_count() or 1))) as pool:
            results = pool.map(_worker_seed_run, args_list)

        for run, correct, cand in results:
            all_runs.append(run)
            answers.append(correct)
            candidates += cand
            if len(run) > max_len:
                max_len = len(run)

        with open(f"./Fusing/record.txt", "a", encoding='utf-8') as f:
            f.write(f"Index: {index}, Decay:{decay.__name__}, Alpha: {alpha}, Correct: {sum(answers) / num_seeds}, Candidates Left: {candidates/num_seeds}\n")
        print(f"Alpha: {alpha}, Correct: {sum(answers) / num_seeds}")

        summed_steps = np.zeros(max_len)
        for run in all_runs:
            padded_run = np.pad(run, (0, max_len - len(run)), 'constant', constant_values=run[-1] if run else 0)
            summed_steps += padded_run

        avg_steps = summed_steps / num_seeds
        plt.plot(avg_steps, label=f"alpha={alpha}")

        if compute_variance:
            # compute per-step variance across runs and overlay 95% CI
            runs_matrix = np.zeros((num_seeds, max_len))
            for i, run in enumerate(all_runs):
                padded = np.pad(run, (0, max_len - len(run)), 'constant', constant_values=run[-1] if run else 0)
                runs_matrix[i, :] = padded
            var_steps = np.var(runs_matrix, axis=0, ddof=0)
            sem = np.sqrt(var_steps / float(num_seeds))
            z = 1.96
            lower = avg_steps - z * sem
            upper = avg_steps + z * sem
            x = np.arange(len(avg_steps))
            plt.fill_between(x, lower, upper, alpha=0.3)

    plt.xlabel("Number of steps")
    plt.ylabel("Average number of candidates")
    plt.title("Number of candidates over time")
    plt.legend()
    plt.savefig(f"./Fusing/alpha_effect_plot_{index}_with_{decay.__name__}.png")
    plt.close()


# plot_alpha_variance_effect removed: variance-only plotting is deleted to keep CI overlay only

def test2(decay):
    for i in range(len(data)):
        shuffled_means = np.random.permutation(data[i])
        data[i] = shuffled_means
        # run altered-alpha plotting
        plot_alpha_altered(data, i, decay, num_seeds=10, compute_variance=True)
        # variance-only plotting removed; CI is overlaid on the main plot when requested

if __name__ == "__main__":
    # Required on Windows to allow multiprocessing of spawned processes
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass

    # Run the test functions in the main process but internally each test uses a Pool
    # This avoids matplotlib being called in child processes and centralizes file I/O.
    test1()
    test2(decay1)
    test2(decay2)
    test2(decay3)
    test2(decay4)
    test2(decay5)