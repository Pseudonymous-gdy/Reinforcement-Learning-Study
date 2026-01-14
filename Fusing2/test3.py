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
from Fusing2.FSH import FusionSequentialHalving_Final as FSH_Final


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

    Tiny-vs-non-tiny settings differ by <= 10x in total range.
    """
    K = int(K)
    if K <= 0:
        return np.array([], dtype=float)

    if scenario == "uniform_gap":
        # total range ~0.80
        vals = np.linspace(0.90, 0.10, K, dtype=float)

    elif scenario == "uniform_tiny_gap":
        # total range ~0.08 (10x smaller)
        vals = np.linspace(0.54, 0.46, K, dtype=float)

    elif scenario == "fusing_bump":
        vals = np.linspace(0.80, 0.20, K, dtype=float)
        vals[0] = min(1.0, vals[0] + 0.006)  # tiny bump

    elif scenario == "unique_best":
        # gap ~0.20
        vals = np.full(K, 0.60, dtype=float)
        vals[0] = 0.80

    elif scenario == "unique_best_tiny":
        # gap ~0.02 (10x smaller)
        vals = np.full(K, 0.60, dtype=float)
        vals[0] = 0.62

    elif scenario in ("gaps_increasing", "gaps_increasing_tiny"):
        top = 0.90
        total_range = 0.50 if scenario == "gaps_increasing" else 0.07  # 10x
        raw = np.linspace(1.0, float(K - 1), K - 1, dtype=float)  # increasing gaps
        deltas = raw / raw.sum() * total_range
        vals = np.empty(K, dtype=float)
        vals[0] = top
        vals[1:] = top - np.cumsum(deltas)
        vals = np.clip(vals, 0.0, 1.0)

    elif scenario in ("gaps_decreasing", "gaps_decreasing_tiny"):
        top = 0.90
        total_range = 0.50 if scenario == "gaps_decreasing" else 0.07  # 10x
        raw = np.linspace(float(K - 1), 1.0, K - 1, dtype=float)  # decreasing gaps
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
    Returns (mu_permuted, perm).

    Note:
      We do NOT explicitly build a dueling matrix.
      env.get_dueling() uses mu-differences when env.dueling_means is None,
      so permuting mu induces the same permutation on the duel structure.
    """
    rng = np.random.default_rng(int(base_perm_seed + 100000 * int(K) + int(seed)))
    perm = rng.permutation(int(K)).astype(int)
    mu_p = np.asarray(mu_sorted, dtype=float)[perm]
    return mu_p, perm


# =============================================================================
# alpha0(zeta) from dominant-balance heuristic
# =============================================================================
def alpha0_from_zeta(zeta: float) -> float:
    """
    rho = 2*T_R/T_D = 2*zeta/(1-zeta)
    alpha0 = (3 + rho - sqrt(rho^2 + 6 rho + 1))/4
    clamp into [0, 0.5]
    """
    z = float(zeta)
    if z <= 0.0:
        return 0.5
    if z >= 1.0:
        return 0.0
    rho = 2.0 * z / (1.0 - z)
    disc = rho * rho + 6.0 * rho + 1.0
    a0 = (3.0 + rho - math.sqrt(disc)) / 4.0
    return float(min(0.5, max(0.0, a0)))


# =============================================================================
# FSH_Final instantiation / running
# =============================================================================
def instantiate_fsh_final(env: Environment, total_budget: int, zeta: float, alpha: float, seed: int):
    """
    Try a few common signatures (robust to minor API differences).
    """
    kwargs_candidates = [
        dict(env=env, total_budget=total_budget, zeta=zeta, alpha=alpha, seed=seed),
        dict(env=env, total_budget=total_budget, zeta=zeta, alpha=alpha),
        dict(env=env, T=total_budget, zeta=zeta, alpha=alpha, seed=seed),
        dict(env=env, T=total_budget, zeta=zeta, alpha=alpha),
        dict(env=env, budget=total_budget, zeta=zeta, alpha=alpha, seed=seed),
        dict(env=env, budget=total_budget, zeta=zeta, alpha=alpha),
    ]
    last_err = None
    for kw in kwargs_candidates:
        try:
            return FSH_Final(**kw)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot instantiate FSH_Final with tried signatures; last error: {last_err}")


def run_fsh_and_get_arm(algo) -> int:
    """
    Return final estimated arm id.
    Prefer run()->int; fallback to candidate_set accessors.
    """
    res = None
    if hasattr(algo, "run"):
        res = algo.run()

    if isinstance(res, (int, np.integer)):
        return int(res)

    for attr in ("get_candidate_set", "candidate_set", "C", "C_final"):
        if hasattr(algo, attr):
            obj = getattr(algo, attr)
            C = obj() if callable(obj) else obj
            if isinstance(C, (list, tuple, np.ndarray)) and len(C) > 0:
                if len(C) == 1:
                    return int(C[0])
                return int(min(map(int, C)))

    raise RuntimeError("Cannot extract final arm from algo (expected run()->int or get_candidate_set()).")


def get_spent_budget(algo) -> Optional[int]:
    """
    Extract spent budget if exposed by the algorithm; else None.
    """
    for attr in ("spent_budget", "budget_spent", "t"):
        if hasattr(algo, attr):
            v = getattr(algo, attr)
            try:
                return int(v() if callable(v) else v)
            except Exception:
                pass
    for meth in ("get_spent_budget", "get_round_tot", "get_total_spent"):
        if hasattr(algo, meth):
            try:
                return int(getattr(algo, meth)())
            except Exception:
                pass
    return None


# =============================================================================
# Multiprocessing: batch-of-seeds per (scenario,K,T,zeta,alpha)
# =============================================================================
def _clamp_for_constructor(zeta: float, alpha: float, eps: float) -> Tuple[float, float]:
    """
    Only used if FSH_Final refuses endpoints.
    Keep clamp minimal, but record clamp_rate in logs.
    """
    z = float(zeta)
    a = float(alpha)

    if z <= 0.0:
        z = eps
    if z >= 1.0:
        z = 1.0 - eps
    if a <= 0.0:
        a = eps
    if a >= 0.5:
        a = 0.5 - eps
    return z, a


def _worker_seed_batch(task: Tuple) -> Dict[str, Any]:
    """
    task = (
      scenario, K, T, zeta_req, alpha_req,
      base_seed, seed_list, eps,
      permute_flag, base_perm_seed
    )

    Returns aggregated stats over that seed_list.
    """
    (scenario, K, T, zeta_req, alpha_req, base_seed, seed_list, eps, permute_flag, base_perm_seed) = task
    K = int(K)
    T = int(T)
    permute_flag = bool(permute_flag)
    base_perm_seed = int(base_perm_seed)

    mu_sorted = _make_sorted_values(str(scenario), K)

    n = 0
    acc_sum = 0
    spent_sum = 0.0
    spent_sumsq = 0.0
    clamp_cnt = 0

    for rep_seed in seed_list:
        rep_seed = int(rep_seed)

        # per-seed permutation increases randomness across runs
        if permute_flag:
            means, _ = permute_mu(mu_sorted, seed=rep_seed, K=K, base_perm_seed=base_perm_seed)
        else:
            means = mu_sorted.copy()

        # env RNG seed and algo seed (separate)
        env_seed = int(base_seed + 10007 * rep_seed + 11)
        algo_seed = int(base_seed + 20011 * rep_seed + 29)

        env = Environment(
            number_of_bandits=K,
            distribution="bernoulli",
            seed=env_seed,
            dueling_means=None,  # implicit from mu-differences
        )
        env.set_means(means.tolist())
        true_best = int(env.get_optimal_action())

        zeta_used = float(zeta_req)
        alpha_used = float(alpha_req)

        try:
            algo = instantiate_fsh_final(env, T, zeta_used, alpha_used, algo_seed)
        except Exception:
            clamp_cnt += 1
            zeta_used, alpha_used = _clamp_for_constructor(zeta_used, alpha_used, eps)
            algo = instantiate_fsh_final(env, T, zeta_used, alpha_used, algo_seed)

        chosen = run_fsh_and_get_arm(algo)
        spent = get_spent_budget(algo)
        if spent is None:
            spent = len(getattr(algo, "history", [])) if hasattr(algo, "history") else 0

        acc = 1 if int(chosen) == true_best else 0

        n += 1
        acc_sum += int(acc)
        spent_sum += float(spent)
        spent_sumsq += float(spent) * float(spent)

    return dict(
        scenario=str(scenario),
        K=int(K),
        T=int(T),
        zeta_req=float(zeta_req),
        alpha_req=float(alpha_req),
        n=int(n),
        acc_sum=int(acc_sum),
        spent_sum=float(spent_sum),
        spent_sumsq=float(spent_sumsq),
        clamp_cnt=int(clamp_cnt),
        permute_mu=int(1 if permute_flag else 0),
        base_perm_seed=int(base_perm_seed),
    )


# =============================================================================
# Aggregation & CI
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

    g = df.groupby(["scenario", "K", "T", "zeta_req", "alpha_req"], as_index=False).agg(
        n=("n", "sum"),
        acc_sum=("acc_sum", "sum"),
        spent_sum=("spent_sum", "sum"),
        spent_sumsq=("spent_sumsq", "sum"),
        clamp_cnt=("clamp_cnt", "sum"),
        permute_mu=("permute_mu", "max"),
        base_perm_seed=("base_perm_seed", "max"),
    )

    g["acc_mean"] = g["acc_sum"] / g["n"].clip(lower=1)
    p = g["acc_mean"].astype(float)
    g["acc_std"] = np.sqrt(np.clip(p * (1.0 - p), 0.0, 1.0))
    g["acc_sem"] = g["acc_std"] / np.sqrt(g["n"].clip(lower=1).astype(float))
    g["acc_ci95"] = 1.96 * g["acc_sem"]

    spent_mean, spent_std = [], []
    for _, row in g.iterrows():
        n = int(row["n"])
        m, sd = _mean_std_from_sums(n, float(row["spent_sum"]), float(row["spent_sumsq"]))
        spent_mean.append(m)
        spent_std.append(sd)
    g["spent_mean"] = spent_mean
    g["spent_std"] = spent_std
    g["spent_sem"] = g["spent_std"] / np.sqrt(g["n"].clip(lower=1).astype(float))
    g["spent_ci95"] = 1.96 * g["spent_sem"]

    g["clamp_rate"] = g["clamp_cnt"] / g["n"].clip(lower=1)
    return g


# =============================================================================
# Plotting
# =============================================================================
def plot_heatmap(df_sum: pd.DataFrame, scenario: str, K: int, T: int, outdir: Path, alpha_grid: np.ndarray):
    sub = df_sum[(df_sum["scenario"] == scenario) & (df_sum["K"] == K) & (df_sum["T"] == T)].copy()
    if sub.empty:
        return

    piv = sub.pivot(index="alpha_req", columns="zeta_req", values="acc_mean")
    if piv.empty:
        return

    alphas = np.array(piv.index.values, dtype=float)
    zetas = np.array(piv.columns.values, dtype=float)
    Z = piv.values

    plt.figure(figsize=(10, 6))
    im = plt.imshow(
        Z,
        aspect="auto",
        origin="lower",
        extent=[float(zetas.min()), float(zetas.max()), float(alphas.min()), float(alphas.max())],
    )
    plt.colorbar(im, label="acc_mean")
    plt.xlabel("zeta")
    plt.ylabel("alpha")
    plt.title(f"{scenario} | K={K} T={T}: acc_mean heatmap")

    # overlay alpha0(zeta)
    z_line = np.linspace(float(zetas.min()), float(zetas.max()), 200)
    a_line = np.array([alpha0_from_zeta(z) for z in z_line], dtype=float)
    plt.plot(z_line, a_line, linewidth=2)

    outpath = outdir / f"{scenario}__K{K}__T{T}__heatmap_acc_mean.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_acc_vs_zeta_lines(df_sum: pd.DataFrame, scenario: str, K: int, T: int, outdir: Path, alpha_grid: np.ndarray):
    sub = df_sum[(df_sum["scenario"] == scenario) & (df_sum["K"] == K) & (df_sum["T"] == T)].copy()
    if sub.empty:
        return

    zetas = np.array(sorted(sub["zeta_req"].unique()), dtype=float)

    best_mean, best_ci = [], []
    worst_mean, worst_ci = [], []
    alpha0_mean, alpha0_ci = [], []
    alpha0_alpha = []

    # prepare arrays for explicit alpha targets (nearest grid match)
    a05_near = float(alpha_grid[np.argmin(np.abs(alpha_grid - 0.5))])
    a00_near = float(alpha_grid[np.argmin(np.abs(alpha_grid - 0.0))])
    alpha05_mean, alpha05_ci = [], []
    alpha00_mean, alpha00_ci = [], []

    for z in zetas:
        zz = float(z)
        s = sub[sub["zeta_req"] == zz]
        if s.empty:
            continue

        # best/worst across alpha for this zeta
        ib = s["acc_mean"].idxmax()
        iw = s["acc_mean"].idxmin()
        b = s.loc[ib]
        w = s.loc[iw]

        best_mean.append(float(b["acc_mean"]))
        best_ci.append(float(b["acc_ci95"]))
        worst_mean.append(float(w["acc_mean"]))
        worst_ci.append(float(w["acc_ci95"]))

        # alpha0 nearest grid
        a0 = alpha0_from_zeta(zz)
        a_near = float(alpha_grid[np.argmin(np.abs(alpha_grid - a0))])
        alpha0_alpha.append(a_near)

        s2 = s[s["alpha_req"] == a_near]
        if s2.empty:
            alpha0_mean.append(float("nan"))
            alpha0_ci.append(float("nan"))
        else:
            alpha0_mean.append(float(s2.iloc[0]["acc_mean"]))
            alpha0_ci.append(float(s2.iloc[0]["acc_ci95"]))

        # alpha=0.5 (nearest grid) performance
        s05 = s[s["alpha_req"] == a05_near]
        if s05.empty:
            alpha05_mean.append(float("nan"))
            alpha05_ci.append(float("nan"))
        else:
            alpha05_mean.append(float(s05.iloc[0]["acc_mean"]))
            alpha05_ci.append(float(s05.iloc[0]["acc_ci95"]))

        # alpha=0.0 (nearest grid) performance
        s00 = s[s["alpha_req"] == a00_near]
        if s00.empty:
            alpha00_mean.append(float("nan"))
            alpha00_ci.append(float("nan"))
        else:
            alpha00_mean.append(float(s00.iloc[0]["acc_mean"]))
            alpha00_ci.append(float(s00.iloc[0]["acc_ci95"]))

    zetas_plot = zetas[: len(best_mean)]
    best_mean = np.array(best_mean, dtype=float)
    best_ci = np.array(best_ci, dtype=float)
    worst_mean = np.array(worst_mean, dtype=float)
    worst_ci = np.array(worst_ci, dtype=float)
    alpha0_mean = np.array(alpha0_mean, dtype=float)
    alpha0_ci = np.array(alpha0_ci, dtype=float)
    alpha05_mean = np.array(alpha05_mean, dtype=float)
    alpha05_ci = np.array(alpha05_ci, dtype=float)
    alpha00_mean = np.array(alpha00_mean, dtype=float)
    alpha00_ci = np.array(alpha00_ci, dtype=float)

    plt.figure(figsize=(10, 6))

    plt.plot(zetas_plot, best_mean, linewidth=2, label="best over alpha")
    plt.fill_between(
        zetas_plot,
        np.clip(best_mean - best_ci, 0, 1),
        np.clip(best_mean + best_ci, 0, 1),
        alpha=0.2,
    )

    plt.plot(zetas_plot, worst_mean, linewidth=2, label="worst over alpha")
    plt.fill_between(
        zetas_plot,
        np.clip(worst_mean - worst_ci, 0, 1),
        np.clip(worst_mean + worst_ci, 0, 1),
        alpha=0.2,
    )

    plt.plot(zetas_plot, alpha0_mean, linewidth=2, label="alpha0(zeta) (nearest grid)")
    plt.fill_between(
        zetas_plot,
        np.clip(alpha0_mean - alpha0_ci, 0, 1),
        np.clip(alpha0_mean + alpha0_ci, 0, 1),
        alpha=0.2,
    )

    # explicit alpha targets: alpha=0.5 and alpha=0.0 (nearest grid values)
    plt.plot(zetas_plot, alpha05_mean, linewidth=2, linestyle="--", label=f"alpha=0.5 (grid {a05_near:.3f})")
    plt.fill_between(
        zetas_plot,
        np.clip(alpha05_mean - alpha05_ci, 0, 1),
        np.clip(alpha05_mean + alpha05_ci, 0, 1),
        alpha=0.15,
    )

    plt.plot(zetas_plot, alpha00_mean, linewidth=2, linestyle=":", label=f"alpha=0.0 (grid {a00_near:.3f})")
    plt.fill_between(
        zetas_plot,
        np.clip(alpha00_mean - alpha00_ci, 0, 1),
        np.clip(alpha00_mean + alpha00_ci, 0, 1),
        alpha=0.15,
    )

    plt.xlabel("zeta")
    plt.ylabel("acc_mean")
    plt.ylim(0.28, 1.02)
    plt.title(f"{scenario} | K={K} T={T}: acc_mean vs zeta (best/worst/alpha0) with 95% CI")
    plt.legend()

    outpath = outdir / f"{scenario}__K{K}__T{T}__acc_mean_vs_zeta_lines.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    audit = pd.DataFrame({"zeta": zetas_plot, "alpha0_nearest": np.array(alpha0_alpha, dtype=float), "alpha_0": np.array([alpha0_from_zeta(z) for z in zetas_plot], dtype=float)})
    audit.to_csv(outdir / f"{scenario}__K{K}__T{T}__alpha0_nearest_grid.csv", index=False)


# =============================================================================
# CLI
# =============================================================================
def parse_float_list(s: str) -> np.ndarray:
    xs = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        xs.append(float(tok))
    return np.array(xs, dtype=float)


def parse_int_list(s: str) -> List[int]:
    xs = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        xs.append(int(tok))
    return xs


def parse_args():
    p = argparse.ArgumentParser()

    # output root folder (will create timestamp subfolder inside)
    p.add_argument("--logdir", type=str, default="./log")

    # multi-K, multi-T
    p.add_argument("--Ks", type=str, default="32")
    p.add_argument("--Ts", type=str, default="10000,20000,40000")

    # experiment repeats
    p.add_argument("--n_seeds", type=int, default=5000)
    p.add_argument("--base_seed", type=int, default=12345)

    # multiprocessing
    p.add_argument("--n_cpu", type=int, default=max(1, mp.cpu_count() - 1))

    # grids: either supply a comma-separated list via --zetas/--alphas,
    # or omit them and use --zeta_step/--alpha_step to auto-generate ranges.
    p.add_argument("--zetas", type=str, default=None,
                   help="Comma-separated zeta values (e.g. '0,0.025,0.05') or omit to use --zeta_step")
    p.add_argument("--zeta_step", type=float, default=0.025,
                   help="Step size when auto-generating zeta grid from 0 to 1 (used if --zetas omitted)")

    p.add_argument("--alphas", type=str, default=None,
                   help="Comma-separated alpha values (e.g. '0,0.025,0.05') or omit to use --alpha_step")
    p.add_argument("--alpha_step", type=float, default=0.0125,
                   help="Step size when auto-generating alpha grid from 0 to 0.5 (used if --alphas omitted)")

    # scenarios subset
    p.add_argument("--scenarios", type=str, default=",".join(SCENARIOS))

    # clamp epsilon (only if FSH_Final rejects endpoints)
    p.add_argument("--eps", type=float, default=1e-12)

    # per-seed permutation
    p.add_argument("--permute_mu", type=int, default=1, help="1: permute mu per seed; 0: keep fixed identities")
    p.add_argument("--base_perm_seed", type=int, default=77777)

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    Ks = parse_int_list(args.Ks)
    Ts = parse_int_list(args.Ts)

    # build zeta/alpha grids: accept explicit comma-list or auto-generate from step
    if args.zetas is None:
        step = float(args.zeta_step)
        # include endpoint 1.0 robustly
        zeta_grid = np.round(np.arange(0.0, 1.0 + step / 2.0, step), decimals=12)
    else:
        zeta_grid = parse_float_list(args.zetas)

    if args.alphas is None:
        step = float(args.alpha_step)
        # alpha in [0,0.5]
        alpha_grid = np.round(np.arange(0.0, 0.5 + step / 2.0, step), decimals=12)
    else:
        alpha_grid = parse_float_list(args.alphas)

    scen_list = [x.strip() for x in args.scenarios.split(",") if x.strip() != ""]
    for s in scen_list:
        if s not in SCENARIOS:
            raise ValueError(f"Unknown scenario in --scenarios: {s}")

    # timestamp folder under logdir
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.logdir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # decide actual cpu used (avoid nonsense)
    cpu_cap = mp.cpu_count()
    n_cpu_used = int(max(1, min(int(args.n_cpu), int(cpu_cap))))

    # log meta (all parameters)
    meta = dict(
        timestamp=str(dt.datetime.now()),
        run_dir=str(run_dir.resolve()),
        Ks=Ks,
        Ts=Ts,
        n_seeds=int(args.n_seeds),
        base_seed=int(args.base_seed),
        n_cpu_requested=int(args.n_cpu),
        n_cpu_used=int(n_cpu_used),
        cpu_cap=int(cpu_cap),
        zetas=zeta_grid.tolist(),
        alphas=alpha_grid.tolist(),
        zeta_step=float(args.zeta_step) if getattr(args, "zeta_step", None) is not None else None,
        alpha_step=float(args.alpha_step) if getattr(args, "alpha_step", None) is not None else None,
        scenarios=scen_list,
        eps=float(args.eps),
        distribution="bernoulli",
        permute_mu=bool(int(args.permute_mu)),
        base_perm_seed=int(args.base_perm_seed),
        notes=dict(
            tiny_vs_non_tiny_gap_ratio_leq_10=True,
            mp_seed_batching=True,
            per_config_tasks="(scenario,K,T,zeta,alpha) x seed-batches",
            dueling_structure="implicit from means (env.dueling_means=None)",
        ),
    )
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # concise logfile
    log_fp = open(run_dir / "run.log", "w", encoding="utf-8")

    def log(msg: str):
        print(msg)
        log_fp.write(msg + "\n")
        log_fp.flush()

    log(f"[run] {timestamp} | run_dir={run_dir.resolve()}")
    log(f"[cfg] Ks={Ks} | Ts={Ts} | n_seeds={args.n_seeds} | base_seed={args.base_seed} | n_cpu={n_cpu_used}/{cpu_cap}")
    log(f"[grid] zetas={zeta_grid.tolist()}")
    log(f"[grid] alphas={alpha_grid.tolist()}")
    log(f"[scenarios] {scen_list}")
    log(f"[permute_mu] enabled={bool(int(args.permute_mu))} | base_perm_seed={int(args.base_perm_seed)}")

    seeds = list(range(int(args.n_seeds)))

    # multiprocessing pool
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=n_cpu_used)

    all_batch_rows: List[Dict[str, Any]] = []

    try:
        for K in Ks:
            for T in Ts:
                for scenario in scen_list:
                    sub_dir = run_dir / f"{scenario}__K{K}__T{T}"
                    sub_dir.mkdir(parents=True, exist_ok=True)

                    # seed batches (no chunksize; batching reduces IPC)
                    batch_size = int(math.ceil(len(seeds) / max(1, n_cpu_used)))
                    seed_batches = [seeds[i:i + batch_size] for i in range(0, len(seeds), batch_size)]

                    tasks: List[Tuple] = []
                    for z in zeta_grid:
                        for a in alpha_grid:
                            for sb in seed_batches:
                                tasks.append((
                                    scenario, int(K), int(T),
                                    float(z), float(a),
                                    int(args.base_seed), sb,
                                    float(args.eps),
                                    int(args.permute_mu), int(args.base_perm_seed),
                                ))

                    it = pool.imap_unordered(_worker_seed_batch, tasks)
                    if tqdm is not None:
                        it = tqdm(it, total=len(tasks), desc=f"{scenario} K={K} T={T}", leave=False)

                    batch_rows = []
                    for row in it:
                        batch_rows.append(row)

                    all_batch_rows.extend(batch_rows)

                    df_sum = aggregate_batches(batch_rows)
                    df_sum.to_csv(sub_dir / "summary.csv", index=False)

                    cols_keep = [
                        "scenario", "K", "T", "zeta_req", "alpha_req",
                        "n", "acc_mean", "acc_std", "acc_ci95",
                        "spent_mean", "spent_std", "spent_ci95",
                        "clamp_rate",
                        "permute_mu", "base_perm_seed",
                    ]
                    if not df_sum.empty:
                        df_sum[cols_keep].to_csv(sub_dir / "summary_slim.csv", index=False)

                    # plots
                    if not df_sum.empty:
                        plot_heatmap(df_sum, scenario, int(K), int(T), sub_dir, alpha_grid=alpha_grid)
                        plot_acc_vs_zeta_lines(df_sum, scenario, int(K), int(T), sub_dir, alpha_grid=alpha_grid)

                        # concise summary line
                        ib = df_sum["acc_mean"].idxmax()
                        iw = df_sum["acc_mean"].idxmin()
                        best = df_sum.loc[ib]
                        worst = df_sum.loc[iw]
                        log(
                            f"[done] {scenario} | K={K} T={T} | "
                            f"best acc_mean={best['acc_mean']:.3f} @ (zeta={best['zeta_req']:.3f}, alpha={best['alpha_req']:.3f}) | "
                            f"worst acc_mean={worst['acc_mean']:.3f} @ (zeta={worst['zeta_req']:.3f}, alpha={worst['alpha_req']:.3f}) | "
                            f"clamp_rate_best={best['clamp_rate']:.3f}"
                        )
                    else:
                        log(f"[done] {scenario} | K={K} T={T} | EMPTY SUMMARY (check failures)")

        # overall combined summaries
        df_all_batches = pd.DataFrame(all_batch_rows)
        df_all_batches.to_csv(run_dir / "all__batches.csv", index=False)

        df_all = aggregate_batches(all_batch_rows)
        df_all.to_csv(run_dir / "all__summary.csv", index=False)

        # scenario-level best/worst per (scenario,K,T)
        scen_stats = []
        if not df_all.empty:
            for (scenario, K, T), sub in df_all.groupby(["scenario", "K", "T"], as_index=False):
                # sub is a DataFrame with its own index labels; use loc with idxmax/idxmin
                ib = sub["acc_mean"].idxmax()
                iw = sub["acc_mean"].idxmin()
                b = sub.loc[ib]
                w = sub.loc[iw]
                scen_stats.append(dict(
                    scenario=str(scenario),
                    K=int(K),
                    T=int(T),
                    best_acc_mean=float(b["acc_mean"]),
                    best_zeta=float(b["zeta_req"]),
                    best_alpha=float(b["alpha_req"]),
                    worst_acc_mean=float(w["acc_mean"]),
                    worst_zeta=float(w["zeta_req"]),
                    worst_alpha=float(w["alpha_req"]),
                    clamp_rate_best=float(b["clamp_rate"]),
                    clamp_rate_worst=float(w["clamp_rate"]),
                    permute_mu=int(b["permute_mu"]),
                    base_perm_seed=int(b["base_perm_seed"]),
                ))
        pd.DataFrame(scen_stats).to_csv(run_dir / "scenario_KT__best_worst.csv", index=False)

        log(f"[saved] run_dir={run_dir.resolve()}")

    finally:
        pool.close()
        pool.join()
        log_fp.close()


if __name__ == "__main__":
    main()
