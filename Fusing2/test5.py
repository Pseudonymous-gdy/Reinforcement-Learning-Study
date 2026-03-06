import os
import sys
import math
import argparse
import datetime as dt
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Fusing2.env import Environment
from Fusing2.combined_sequential_halving import CombinedSequentialHalving
from Fusing2.FSH import FusionSequentialHalving_Final


SCENARIOS = [
    "fusing_bump",
    "uniform_gap",
    "uniform_tiny_gap",
    "unique_best_tiny",
    "unique_best",
    "gaps_increasing_tiny",
    "gaps_increasing",
    "gaps_decreasing_tiny",
    "gaps_decreasing",
    "random",
]

def _scenario_rng_seed(scenario: str, K: int) -> int:
    s = sum((i + 1) * ord(c) for i, c in enumerate(scenario))
    return int((s * 1000003 + 17 * int(K)) % (2**31 - 1))

def _make_sorted_values(scenario: str, K: int) -> np.ndarray:
    K = int(K)
    if K <= 0: return np.array([], dtype=float)
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
    rng = np.random.default_rng(int(base_perm_seed + 100000 * int(K) + int(seed)))
    perm = rng.permutation(int(K)).astype(int)
    mu_p = np.asarray(mu_sorted, dtype=float)[perm]
    return mu_p, perm

def alpha0_from_zeta(zeta: float) -> float:
    z = float(zeta)
    if z <= 0.0: return 0.5
    if z >= 1.0: return 0.0
    rho = 2.0 * z / (1.0 - z)
    disc = rho * rho + 6.0 * rho + 1.0
    a0 = (3.0 + rho - math.sqrt(disc)) / 4.0
    return float(min(0.5, max(0.0, a0)))

def _worker_seed_batch(task: Tuple) -> List[Dict[str, Any]]:
    (scenario, K, C, wR, z, base_seed, seed_list, permute_flag, base_perm_seed) = task
    
    agg = {
        "FSH": {"n": 0, "acc_sum": 0, "spent_sum": 0.0},
        "CSH": {"n": 0, "acc_sum": 0, "spent_sum": 0.0}
    }
    
    mu_sorted = _make_sorted_values(str(scenario), K)
    
    for rep_seed in seed_list:
        rep_seed = int(rep_seed)
        if permute_flag:
            means, _ = permute_mu(mu_sorted, seed=rep_seed, K=K, base_perm_seed=base_perm_seed)
        else:
            means = mu_sorted.copy()
            
        true_best = int(np.argmax(means))
        env_seed = int(base_seed + 10007 * rep_seed + 11)
        base_algo_seed = int(base_seed + 20011 * rep_seed + 29)
        
        T_v = int(C / (z * wR + 1.0 - z))
        
        # FSH
        env1 = Environment(number_of_bandits=K, distribution="bernoulli", seed=env_seed)
        env1.set_means(means.tolist())
        try:
            a_req = alpha0_from_zeta(z)
            algo1 = FusionSequentialHalving_Final(env=env1, total_budget=T_v, zeta=z, alpha=a_req, seed=base_algo_seed+1)
            algo1.run()
            c1 = algo1.candidate_set if hasattr(algo1, "candidate_set") else list(algo1.C)
            ans1 = int(c1[0]) if len(c1)>0 else -1
            s1 = algo1.spent_budget
        except Exception:
            ans1, s1 = -1, 0
        
        k_fsh = "FSH"
        agg[k_fsh]["n"] += 1
        if ans1 == true_best: agg[k_fsh]["acc_sum"] += 1
        agg[k_fsh]["spent_sum"] += float(s1)
        
        # CSH
        env2 = Environment(number_of_bandits=K, distribution="bernoulli", seed=env_seed)
        env2.set_means(means.tolist())
        try:
            algo2 = CombinedSequentialHalving(env=env2, total_budget=T_v, alpha=z, seed=base_algo_seed+2)
            algo2.run()
            c2 = algo2.get_candidate_set() if hasattr(algo2, "get_candidate_set") else algo2.candidate_set
            ans2 = int(c2[0]) if len(c2)>0 else -1
            s2 = algo2.spent_budget
        except Exception:
            ans2, s2 = -1, 0
        
        k_csh = "CSH"
        agg[k_csh]["n"] += 1
        if ans2 == true_best: agg[k_csh]["acc_sum"] += 1
        agg[k_csh]["spent_sum"] += float(s2)

    res_list = []
    for algo, st in agg.items():
        if st["n"] > 0:
            res_list.append({
                "scenario": scenario, "K": K, "C": C, 
                "omega_R": float(wR), "zeta": float(z), "algo": algo,
                "n": st["n"], "acc_sum": st["acc_sum"], "spent_sum": st["spent_sum"],
                "permute_mu": int(permute_flag), "base_perm_seed": base_perm_seed
            })
    return res_list

def plot_fixed_cost_zeta(df: pd.DataFrame, scenario: str, K: int, C: int, w_R: float, outdir: Path):
    sub = df[(df["scenario"] == scenario) & (df["K"] == K) & (df["C"] == C) & (np.isclose(df["omega_R"], w_R))].copy()
    if sub.empty: return

    fsh_sub = sub[sub["algo"] == "FSH"].sort_values("zeta")
    csh_sub = sub[sub["algo"] == "CSH"].sort_values("zeta")

    plt.figure(figsize=(8,6))
    plt.plot(fsh_sub["zeta"], fsh_sub["acc_mean"], label="FSH", marker='o')
    plt.plot(csh_sub["zeta"], csh_sub["acc_mean"], label="CSH", marker='s')

    pd_fsh = fsh_sub[np.isclose(fsh_sub["zeta"], 0.0)]
    pd_val = pd_fsh["acc_mean"].values[0] if not pd_fsh.empty else np.nan
    plt.axhline(pd_val, color="red", linestyle="--", label="Pure Dueling")

    pr_fsh = fsh_sub[np.isclose(fsh_sub["zeta"], 1.0)]
    pr_val = pr_fsh["acc_mean"].values[0] if not pr_fsh.empty else np.nan
    plt.axhline(pr_val, color="green", linestyle=":", label="Pure Reward")

    plt.xlabel(r"$\zeta = T_R / (T_R + T_D)$")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.02)
    plt.title(f"{scenario} | K={K} C={C} | $\omega_R={w_R}$")
    plt.legend()
    plt.grid(True)
    outpath = outdir / f"{scenario}__K{K}__C{C}__wR_{w_R}__plot1_zeta.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_fixed_cost_omegaR(df: pd.DataFrame, scenario: str, K: int, C: int, outdir: Path):
    sub = df[(df["scenario"] == scenario) & (df["K"] == K) & (df["C"] == C)].copy()
    if sub.empty: return

    wRs = sorted(sub["omega_R"].unique())
    fsh_best = []
    csh_best = []
    pd_vals = []
    pr_vals = []

    for wR in wRs:
        sub_wR = sub[np.isclose(sub["omega_R"], wR)]
        fsh_wR = sub_wR[sub_wR["algo"] == "FSH"]
        csh_wR = sub_wR[sub_wR["algo"] == "CSH"]
        
        fsh_best.append(fsh_wR["acc_mean"].max() if not fsh_wR.empty else np.nan)
        csh_best.append(csh_wR["acc_mean"].max() if not csh_wR.empty else np.nan)
        
        pd_val = fsh_wR[np.isclose(fsh_wR["zeta"], 0.0)]["acc_mean"].values
        pd_vals.append(pd_val[0] if len(pd_val)>0 else np.nan)
        
        pr_val = fsh_wR[np.isclose(fsh_wR["zeta"], 1.0)]["acc_mean"].values
        pr_vals.append(pr_val[0] if len(pr_val)>0 else np.nan)

    plt.figure(figsize=(8,6))
    plt.plot(wRs, fsh_best, label="FSH (Best over $\zeta$)", marker='o')
    plt.plot(wRs, csh_best, label="CSH (Best over $\zeta/\\alpha$)", marker='s')
    plt.plot(wRs, pd_vals, label="Pure Dueling", linestyle="--", color="red", marker='^')
    plt.plot(wRs, pr_vals, label="Pure Reward", linestyle=":", color="green", marker='v')
    
    plt.xlabel(r"$\omega_R$")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.02)
    plt.title(f"{scenario} | K={K} C={C}")
    plt.legend()
    plt.grid(True)
    outpath = outdir / f"{scenario}__K{K}__C{C}__plot2_omegaR.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="./log")
    p.add_argument("--n_seeds", type=int, default=5000)
    p.add_argument("--base_seed", type=int, default=12345)
    p.add_argument("--n_cpu", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--scenarios", type=str, default=",".join(SCENARIOS))
    p.add_argument("--Ks", type=str, default="32")
    p.add_argument("--Cs", type=str, default="10000,20000,40000")
    p.add_argument("--permute_mu", type=int, default=1)
    p.add_argument("--base_perm_seed", type=int, default=77777)
    return p.parse_args()

def main():
    args = parse_args()
    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]
    Cs = [int(x) for x in args.Cs.split(",") if x.strip()]
    scen_list = [x.strip() for x in args.scenarios.split(",") if x.strip() != ""]

    omega_Rs = list(np.linspace(0.5, 1.5, 11))
    zetas = list(np.linspace(0.0, 1.0, 21))

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_test5")
    run_dir = Path(args.logdir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting run in {run_dir}")
    
    seeds = list(range(args.n_seeds))
    n_cpu = max(1, min(args.n_cpu, mp.cpu_count()))
    batch_size = int(math.ceil(len(seeds) / n_cpu))
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=n_cpu)
    
    all_rows = []
    
    try:
        for scen in scen_list:
            for K in Ks:
                for C in Cs:
                    sub_dir = run_dir / f"{scen}__K{K}__C{C}"
                    sub_dir.mkdir(parents=True, exist_ok=True)
                    
                    seed_batches = [seeds[i:i + batch_size] for i in range(0, len(seeds), batch_size)]
                    tasks = []
                    for wR in omega_Rs:
                        for z in zetas:
                            for sb in seed_batches:
                                tasks.append((
                                    scen, K, C, wR, z,
                                    args.base_seed, sb,
                                    bool(args.permute_mu), args.base_perm_seed
                                ))
                    
                    it = pool.imap_unordered(_worker_seed_batch, tasks)
                    if tqdm:
                        it = tqdm(it, total=len(tasks), desc=f"{scen} K={K} C={C}")
                    
                    scen_rows = []
                    batch_file = sub_dir / "all__batches_inc.csv"
                    for res_list in it:
                        scen_rows.extend(res_list)
                        pd.DataFrame(res_list).to_csv(batch_file, mode='a', header=not batch_file.exists(), index=False)
                        
                    if not scen_rows: continue
                    
                    # Aggregate over all batches for this scenario/K/C
                    df_raw = pd.DataFrame(scen_rows)
                    g = df_raw.groupby(["scenario", "K", "C", "omega_R", "zeta", "algo"], as_index=False).sum()
                    g["acc_mean"] = g["acc_sum"] / g["n"].clip(lower=1)
                    
                    g.to_csv(sub_dir / "summary.csv", index=False)
                    all_rows.extend(g.to_dict('records'))
                    
                    # Plots
                    # Plot 1: fix w_R in {0.8, 1.0, 1.3}
                    for wR in [0.8, 1.0, 1.3]:
                         # Due to float precision
                         nearest_wR = min(omega_Rs, key=lambda x: abs(x - wR))
                         plot_fixed_cost_zeta(g, scen, K, C, nearest_wR, sub_dir)
                    
                    # Plot 2: x-axis is w_R
                    plot_fixed_cost_omegaR(g, scen, K, C, sub_dir)

    finally:
        pool.close()
        pool.join()
        
        if all_rows:
            overall_df = pd.DataFrame(all_rows)
            overall_df.to_csv(run_dir / "final_all_batches.csv", index=False)
            print(f"Done! Saved to {run_dir}")

if __name__ == "__main__":
    main()
