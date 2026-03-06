import os
import sys
import json
import math
import time
import argparse
import datetime as dt
import multiprocessing as mp
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# progress bar (main process only)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Fusing2.env import Environment
# Modified import for CombinedSequentialHalving
from Fusing2.combined_sequential_halving import CombinedSequentialHalving


# =============================================================================
# Scenarios
# =============================================================================
SCENARIOS = [
    "fusing_bump",             # equally spaced with tiny bump above base top
    "uniform_gap",             # equally spaced, larger range
    "uniform_tiny_gap",        # equally spaced, tiny gaps (hard)
    "unique_best_tiny",        # μ1 > μ2=...=μK (tiny gap)
    "unique_best",             # μ1 > μ2=...=μK (larger gap)
    "gaps_increasing_tiny",    # Δ_{i,i+1} < Δ_{i+1,i+2} (tiny total range)
    "gaps_increasing",
    "gaps_decreasing_tiny",    # Δ_{i,i+1} > Δ_{i+1,i+2} (tiny total range)
    "gaps_decreasing",
    "random",                  # random generation, sorted descending
]


# =============================================================================
# Scenario generation (tiny vs non-tiny within <= 10x total-range)
# =============================================================================
def _scenario_rng_seed(scenario: str, K: int) -> int:
    s = sum((i + 1) * ord(c) for i, c in enumerate(scenario))
    return int((s * 1000003 + 17 * int(K)) % (2**31 - 1))


def _make_sorted_values(scenario: str, K: int) -> np.ndarray:
    """
    Return descending means (length K), as the "sorted template" for a scenario.
    """
    K = int(K)
    if K <= 0:
        return np.array([], dtype=float)

    if scenario == "uniform_gap":
        vals = np.linspace(0.90, 0.10, K, dtype=float)

    elif scenario == "uniform_tiny_gap":
        vals = np.linspace(0.54, 0.46, K, dtype=float)

    elif scenario == "fusing_bump":
        vals = np.linspace(0.80, 0.20, K, dtype=float)
        vals[0] = min(1.0, vals[0] + 0.006)

    elif scenario == "unique_best":
        vals = np.full(K, 0.60, dtype=float)
        vals[0] = 0.80

    elif scenario == "unique_best_tiny":
        vals = np.full(K, 0.60, dtype=float)
        vals[0] = 0.62

    elif scenario in ("gaps_increasing", "gaps_increasing_tiny"):
        top = 0.90
        total_range = 0.50 if scenario == "gaps_increasing" else 0.07
        raw = np.linspace(1.0, float(K - 1), K - 1, dtype=float)
        deltas = raw / raw.sum() * total_range
        vals = np.empty(K, dtype=float)
        vals[0] = top
        vals[1:] = top - np.cumsum(deltas)
        vals = np.clip(vals, 0.0, 1.0)

    elif scenario in ("gaps_decreasing", "gaps_decreasing_tiny"):
        top = 0.90
        total_range = 0.50 if scenario == "gaps_decreasing" else 0.07
        raw = np.linspace(float(K - 1), 1.0, K - 1, dtype=float)
        deltas = raw / raw.sum() * total_range
        vals = np.empty(K, dtype=float)
        vals[0] = top
        vals[1:] = top - np.cumsum(deltas)
        vals = np.clip(vals, 0.0, 1.0)

    elif scenario == "random":
        rng = np.random.default_rng(_scenario_rng_seed(scenario, K))
        vals = rng.uniform(0.05, 0.95, size=K).astype(float)
        vals.sort()
        vals = vals[::-1]

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    vals = np.asarray(vals, dtype=float)
    vals.sort()
    vals = vals[::-1]
    return vals


def permute_mu(mu_sorted: np.ndarray, seed: int, K: int, base_perm_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Deterministic permutation per (seed, K).
    """
    rng = np.random.default_rng(int(base_perm_seed + 100000 * int(K) + int(seed)))
    perm = rng.permutation(int(K)).astype(int)
    mu_p = np.asarray(mu_sorted, dtype=float)[perm]
    return mu_p, perm


# =============================================================================
# Instantiation / Running
# =============================================================================
def instantiate_csh(env: Environment, total_budget: int, alpha: float, c1: float, c2: float, seed: int):
    """
    Instantiate CombinedSequentialHalving with correct parameters.
    """
    try:
        return CombinedSequentialHalving(
            env=env,
            total_budget=total_budget,
            alpha=alpha,
            episode_keep=0.5,
            seed=seed,
            c1=c1,
            c2=c2
        )
    except Exception as e:
        raise RuntimeError(f"Cannot instantiate CombinedSequentialHalving: {e}")


def run_csh_and_get_arm(algo) -> int:
    """
    Run algorithm and return final estimated arm id.
    """
    if hasattr(algo, "run"):
        algo.run()

    # Get candidate set
    if hasattr(algo, "get_candidate_set"):
        C = algo.get_candidate_set()
        if len(C) > 0:
            if len(C) == 1:
                return int(C[0])
            # If multiple remain, pick best score or just first?
            # CombinedSequentialHalving usually reduces to 1.
            return int(C[0])
            
    # Fallback to internal candidate_set
    if hasattr(algo, "candidate_set"):
        C = algo.candidate_set
        if len(C) > 0:
            return int(C[0])

    raise RuntimeError("Cannot extract final arm from algo.")


def get_spent_budget(algo) -> Optional[int]:
    """
    Extract spent budget.
    """
    if hasattr(algo, "spent_budget"):
        return int(algo.spent_budget)
    return None


# =============================================================================
# Multiprocessing
# =============================================================================
def _worker_seed_batch(task: Tuple) -> Dict[str, Any]:
    """
    task = (
      scenario, K, T, c1, alpha,
      base_seed, seed_list,
      permute_flag, base_perm_seed,
      c2
    )
    """
    (scenario, K, T, c1, alpha, base_seed, seed_list, permute_flag, base_perm_seed, c2) = task
    K = int(K)
    T = int(T)
    c1 = float(c1)
    c2 = float(c2)
    alpha = float(alpha)
    permute_flag = bool(permute_flag)
    base_perm_seed = int(base_perm_seed)

    mu_sorted = _make_sorted_values(str(scenario), K)

    n = 0
    acc_sum = 0
    spent_sum = 0.0
    spent_sumsq = 0.0
    error_cnt = 0

    for rep_seed in seed_list:
        rep_seed = int(rep_seed)

        if permute_flag:
            means, _ = permute_mu(mu_sorted, seed=rep_seed, K=K, base_perm_seed=base_perm_seed)
        else:
            means = mu_sorted.copy()

        env_seed = int(base_seed + 10007 * rep_seed + 11)
        algo_seed = int(base_seed + 20011 * rep_seed + 29)

        env = Environment(
            number_of_bandits=K,
            distribution="bernoulli",
            seed=env_seed,
            # Implicit dueling means from bandit means
            dueling_means=None, 
        )
        env.set_means(means.tolist())
        true_best = int(env.get_optimal_action())

        try:
            algo = instantiate_csh(env, T, alpha, c1, c2, algo_seed)
            chosen = run_csh_and_get_arm(algo)
            spent = get_spent_budget(algo)
            if spent is None:
                spent = 0
            
            acc = 1 if int(chosen) == true_best else 0
            
            n += 1
            acc_sum += int(acc)
            spent_sum += float(spent)
            spent_sumsq += float(spent) * float(spent)

        except Exception as e:
            # Handle error cases by logging and skipping
            error_cnt += 1
            # In a real scenario we might want to capture the error msg
            pass

    return dict(
        scenario=str(scenario),
        K=int(K),
        T=int(T),
        c1_req=float(c1),
        c2_req=float(c2),
        alpha_req=float(alpha),
        n=int(n),
        acc_sum=int(acc_sum),
        spent_sum=float(spent_sum),
        spent_sumsq=float(spent_sumsq),
        error_cnt=int(error_cnt),
        permute_mu=int(1 if permute_flag else 0),
        base_perm_seed=int(base_perm_seed),
    )


# =============================================================================
# Aggregation & Plotting
# =============================================================================
def _mean_std_from_sums(n: int, s: float, ss: float) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    mean = s / n
    if n <= 1:
        return mean, 0.0
    var = (ss - n * mean * mean) / (n - 1)
    var = max(0.0, var)
    return mean, math.sqrt(var)

def aggregate_batches(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    g = df.groupby(["scenario", "K", "T", "c1_req", "c2_req", "alpha_req"], as_index=False).agg(
        n=("n", "sum"),
        acc_sum=("acc_sum", "sum"),
        spent_sum=("spent_sum", "sum"),
        spent_sumsq=("spent_sumsq", "sum"),
        error_cnt=("error_cnt", "sum"),
        permute_mu=("permute_mu", "max"),
        base_perm_seed=("base_perm_seed", "max"),
    )

    g["acc_mean"] = g["acc_sum"] / g["n"].clip(lower=1)
    p = g["acc_mean"].astype(float)
    g["acc_std"] = np.sqrt(np.clip(p * (1.0 - p), 0.0, 1.0))
    g["acc_sem"] = g["acc_std"] / np.sqrt(g["n"].clip(lower=1).astype(float))
    g["acc_ci95"] = 1.96 * g["acc_sem"]

    # Spent
    spent_mean = []
    spent_std = []
    for _, r in g.iterrows():
        n = int(r["n"])
        m, sd = _mean_std_from_sums(n, float(r["spent_sum"]), float(r["spent_sumsq"]))
        spent_mean.append(m)
        spent_std.append(sd)
    g["spent_mean"] = spent_mean
    g["spent_std"] = spent_std
    
    return g

def plot_heatmap(df_sum: pd.DataFrame, scenario: str, K: int, T: int, outdir: Path):
    """
    Heatmap of Accuracy: Alpha (Y) vs C1 (X)
    """
    sub = df_sum[(df_sum["scenario"] == scenario) & (df_sum["K"] == K) & (df_sum["T"] == T)].copy()
    if sub.empty:
        return

    # Pivot: Index=Alpha, Columns=C1
    piv = sub.pivot(index="alpha_req", columns="c1_req", values="acc_mean")
    if piv.empty:
        return

    alphas = np.array(piv.index.values, dtype=float)
    c1s = np.array(piv.columns.values, dtype=float)
    Z = piv.values

    plt.figure(figsize=(8, 6))
    # Origin lower means low alpha at bottom
    im = plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[float(c1s.min()), float(c1s.max()), float(alphas.min()), float(alphas.max())]
    )
    plt.colorbar(im, label="acc_mean")
    plt.xlabel("c1 (c1+c2=1)")
    plt.ylabel("alpha")
    plt.title(f"{scenario} | K={K} T={T}: acc_mean")
    
    outpath = outdir / f"{scenario}__K{K}__T{T}__heatmap_acc_mean.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

# =============================================================================
# Main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="./log")
    p.add_argument("--resume_dir", type=str, default=None, 
                   help="If provided, resume from this directory (aggregating new results).")
    p.add_argument("--Ks", type=str, default="32")
    p.add_argument("--Ts", type=str, default="10000,20000,40000")
    p.add_argument("--n_seeds", type=int, default=5000)
    p.add_argument("--base_seed", type=int, default=12345)
    p.add_argument("--n_cpu", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--alphas", type=str, default=None)
    p.add_argument("--alpha_step", type=float, default=0.05)
    p.add_argument("--scenarios", type=str, default=",".join(SCENARIOS))
    p.add_argument("--permute_mu", type=int, default=1)
    p.add_argument("--base_perm_seed", type=int, default=77777)
    return p.parse_args()

def main():
    args = parse_args()

    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    Ts = [int(x) for x in args.Ts.split(",") if x.strip()]

    # Params
    # 5 c1 values
    c1_grid = np.linspace(0.0, 1.0, 5) # 0, 0.25, 0.5, 0.75, 1.0

    if args.alphas:
        alpha_grid = np.array([float(x) for x in args.alphas.split(",") if x.strip()])
    else:
        # 0.0 to 1.0, step 0.05
        step = args.alpha_step
        alpha_grid = np.arange(0.0, 1.0 + step/10.0, step)
    
    scen_list = [x.strip() for x in args.scenarios.split(",") if x.strip()]

    # Setup Dir
    if args.resume_dir and os.path.exists(args.resume_dir):
        run_dir = Path(args.resume_dir)
        print(f"Resuming from {run_dir}")
    else:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.logdir) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Starting new run in {run_dir}")

    # Log file
    log_fp = open(run_dir / "run.log", "a", encoding="utf-8")
    def log(msg: str):
        print(msg)
        log_fp.write(msg + "\n")
        log_fp.flush()
    
    # Restore logic: Check existing batches
    batch_file = run_dir / "all__batches_inc.csv"
    
    if not batch_file.exists():
        # Initialize
        pd.DataFrame(columns=[
            "scenario","K","T","c1_req","c2_req","alpha_req","n",
            "acc_sum","spent_sum","spent_sumsq","error_cnt",
            "permute_mu","base_perm_seed"
        ]).to_csv(batch_file, index=False)

    # Prepare seeds
    seeds = list(range(args.n_seeds))
    n_cpu = max(1, min(args.n_cpu, mp.cpu_count()))
    batch_size = int(math.ceil(len(seeds) / n_cpu))
    
    # Pool
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=n_cpu)

    try:
        all_rows = []
        if batch_file.exists():
            try:
                # Load existing to memory to include in final aggregator
                existing_df = pd.read_csv(batch_file)
                all_rows = existing_df.to_dict('records')
            except Exception:
                pass

        for K in Ks:
            for T in Ts:
                for scenario in scen_list:
                    # Check if we should skip this block (e.g. if summary exists and is complete?)
                    # For now, we will run and append.
                    
                    sub_dir = run_dir / f"{scenario}__K{K}__T{T}"
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    
                    tasks = []
                    # Generate tasks
                    seed_batches = [seeds[i:i + batch_size] for i in range(0, len(seeds), batch_size)]
                    
                    for c1 in c1_grid:
                        c2 = 1.0 - c1
                        for alpha in alpha_grid:
                            # Create task for each seed batch
                            for sb in seed_batches:
                                tasks.append((
                                    scenario, K, T, c1, alpha,
                                    args.base_seed, sb,
                                    bool(args.permute_mu),
                                    args.base_perm_seed,
                                    c2
                                ))
                    
                    # Run
                    it = pool.imap_unordered(_worker_seed_batch, tasks)
                    if tqdm:
                        it = tqdm(it, total=len(tasks), desc=f"{scenario} K={K} T={T}", leave=False)
                    
                    for res in it:
                        all_rows.append(res)
                        
                        # Incremental write to prevent data loss
                        pd.DataFrame([res]).to_csv(batch_file, mode='a', header=False, index=False)
                    
                    # Analyze and Plot for this Scenario/K/T
                    # We should filter all_rows for current env to make sure we include previously restored info
                    df_scen = pd.DataFrame(all_rows)
                    if not df_scen.empty:
                        # Filter strictly
                        mask = (
                            (df_scen["scenario"] == str(scenario)) & 
                            (df_scen["K"] == int(K)) & 
                            (df_scen["T"] == int(T))
                        )
                        sub_df = df_scen[mask]
                        if not sub_df.empty:
                            summ = aggregate_batches(sub_df.to_dict('records'))
                            summ.to_csv(sub_dir / "summary.csv", index=False)
                            plot_heatmap(summ, scenario, K, T, sub_dir)
                            
                            log(f"[Done] {scenario} K={K} T={T} | processed {len(sub_df)} batches")

    finally:
        pool.close()
        pool.join()
        log_fp.close()
        
        # Final cleanup
        # Save full dataset
        pd.DataFrame(all_rows).to_csv(run_dir / "final_all_batches.csv", index=False)
        log(f"Complete. Saved to {run_dir}")

if __name__ == "__main__":
    main()
