# test.py
# ─────────────────────────────────────────────────────────────────────────────
# Parallel by CONFIG (each worker runs all seeds for one config)
# - Same seeds shared across all configs (same perm / env_seed / algo_seed per seed)
# - Log folder: log/<timestamp>/<config_id>/md/seed_XXX.md (trace all rounds)
# - Per-config integration: per_seed.csv + summary.json
# - Global integration: log/<timestamp>/integration/summary.csv + plots
# - Progress bar (multiprocessing-safe) using Queue + tqdm in main
# - Baseline: SAME schedule as FSH, NO elimination, final = argmax[(1-alpha)*borda + alpha*mu_hat]
# ─────────────────────────────────────────────────────────────────────────────

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
from Fusing2.sequential_halving import FusionSequentialHalving


# -----------------------------------------------------------------------------
# Paper setting (K=16): fixed MU + fixed NU (from your screenshot)
# -----------------------------------------------------------------------------
K_PAPER = 16
MU_PAPER = np.array(
    [0.86, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50,
     0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10],
    dtype=float
)

NU_PAPER = np.array([
    [0.50, 0.54, 0.57, 0.60, 0.63, 0.65, 0.69, 0.71, 0.73, 0.76, 0.78, 0.82, 0.86, 0.91, 0.95, 0.98],
    [0.46, 0.50, 0.54, 0.58, 0.61, 0.64, 0.67, 0.70, 0.74, 0.76, 0.79, 0.81, 0.84, 0.87, 0.89, 0.92],
    [0.43, 0.46, 0.50, 0.54, 0.58, 0.60, 0.63, 0.66, 0.69, 0.72, 0.76, 0.79, 0.83, 0.85, 0.88, 0.91],
    [0.40, 0.42, 0.46, 0.50, 0.54, 0.58, 0.61, 0.64, 0.66, 0.69, 0.72, 0.76, 0.79, 0.82, 0.85, 0.88],
    [0.37, 0.39, 0.42, 0.46, 0.50, 0.54, 0.56, 0.59, 0.63, 0.66, 0.69, 0.72, 0.76, 0.78, 0.82, 0.86],
    [0.35, 0.36, 0.40, 0.42, 0.46, 0.50, 0.54, 0.57, 0.59, 0.63, 0.67, 0.70, 0.73, 0.76, 0.79, 0.82],
    [0.31, 0.33, 0.37, 0.39, 0.44, 0.46, 0.50, 0.54, 0.58, 0.61, 0.64, 0.68, 0.71, 0.72, 0.75, 0.79],
    [0.29, 0.30, 0.34, 0.36, 0.41, 0.43, 0.46, 0.50, 0.54, 0.57, 0.59, 0.62, 0.65, 0.68, 0.72, 0.76],
    [0.27, 0.26, 0.31, 0.34, 0.37, 0.41, 0.42, 0.46, 0.50, 0.54, 0.58, 0.61, 0.63, 0.66, 0.69, 0.73],
    [0.24, 0.24, 0.28, 0.31, 0.34, 0.37, 0.39, 0.43, 0.46, 0.50, 0.54, 0.56, 0.59, 0.62, 0.66, 0.69],
    [0.22, 0.21, 0.24, 0.28, 0.31, 0.33, 0.36, 0.41, 0.42, 0.46, 0.50, 0.54, 0.56, 0.58, 0.61, 0.65],
    [0.18, 0.19, 0.21, 0.24, 0.28, 0.30, 0.32, 0.38, 0.39, 0.44, 0.46, 0.50, 0.54, 0.57, 0.58, 0.62],
    [0.14, 0.16, 0.17, 0.21, 0.24, 0.27, 0.29, 0.35, 0.37, 0.41, 0.44, 0.46, 0.50, 0.54, 0.56, 0.59],
    [0.09, 0.13, 0.15, 0.18, 0.22, 0.24, 0.28, 0.32, 0.34, 0.38, 0.42, 0.43, 0.46, 0.50, 0.54, 0.56],
    [0.05, 0.11, 0.12, 0.15, 0.18, 0.21, 0.25, 0.28, 0.31, 0.34, 0.39, 0.42, 0.44, 0.46, 0.50, 0.54],
    [0.02, 0.08, 0.09, 0.12, 0.14, 0.18, 0.21, 0.24, 0.27, 0.31, 0.35, 0.38, 0.41, 0.44, 0.46, 0.50],
], dtype=float)

assert MU_PAPER.shape == (K_PAPER,)
assert NU_PAPER.shape == (K_PAPER, K_PAPER)


# -----------------------------------------------------------------------------
# Config definitions
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AlgoConfig:
    algo: str                 # "FSH" or "FSH_NO_ELIM_MIX"
    total_budget: int
    alpha: float
    gamma: Optional[float] = None
    episode_keep: Optional[float] = None
    use_paper_dueling: bool = True

    def config_id(self) -> str:
        if self.algo == "FSH":
            return f"FSH_duel{int(self.use_paper_dueling)}_g{self.gamma:.6f}_k{self.episode_keep:.6f}"
        if self.algo == "FSH_NO_ELIM_MIX":
            return f"FSH_NO_ELIM_MIX_duel{int(self.use_paper_dueling)}"
        raise ValueError(f"Unknown algo={self.algo}")

    def folder_name(self) -> str:
        # file-system safe
        return self.config_id().replace(".", "p")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def permute_mu_nu(mu: np.ndarray, nu: np.ndarray, perm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu_p = mu[perm]
    nu_p = nu[np.ix_(perm, perm)]
    return mu_p, nu_p


def _force_set_means_no_sort(env: Environment, mu: np.ndarray) -> None:
    """
    Your Environment.set_means() sorts means, which breaks permutation experiments.
    So we force-override env.bandit_means directly to preserve arm identity.
    """
    if not hasattr(env, "bandit_means"):
        raise RuntimeError("Environment must expose bandit_means attribute to force-override.")
    env.bandit_means = np.array(mu, dtype=float)
    # standard_deviations exists for normal; keep size consistent
    if hasattr(env, "standard_deviations"):
        sd = getattr(env, "standard_deviations")
        if sd is None or len(np.array(sd)) != len(mu):
            env.standard_deviations = np.zeros(len(mu), dtype=float)
    # keep optimal_mean consistent (used by get_optimal_action)
    if hasattr(env, "optimal_mean"):
        env.optimal_mean = float(np.max(mu))


def _force_set_dueling(env: Environment, nu: Optional[np.ndarray]) -> None:
    if nu is None:
        env.dueling_means = None
        return
    env.dueling_means = np.array(nu, dtype=float)


def make_env_fixed(env_seed: int, mu: np.ndarray, nu: Optional[np.ndarray]) -> Environment:
    env = Environment(
        number_of_bandits=len(mu),
        distribution="bernoulli",
        seed=int(env_seed),
        dueling_means=nu if nu is not None else None,
    )
    # force means (preserve order)
    _force_set_means_no_sort(env, mu)
    # force dueling matrix if provided
    _force_set_dueling(env, nu)

    # safety check
    arr = np.array(env.bandit_means, dtype=float)
    if not np.allclose(arr, mu, atol=1e-12):
        raise RuntimeError("Failed to force means without reordering; check your Environment implementation.")
    return env


def _duel_winner(env: Environment, rng: np.random.Generator, i: int, j: int) -> int:
    """Return winner index i or j."""
    if hasattr(env, "duel"):
        w = int(env.duel(i, j))
        if w not in (i, j):
            raise ValueError(f"env.duel must return i or j, got {w}")
        return w
    if hasattr(env, "get_dueling"):
        out = float(env.get_dueling(i, j))
        if out == 1.0:
            return i
        if out == 0.0:
            return j
        return int(rng.choice([i, j]))  # tie
    # fallback
    ri = float(env.get_reward(i))
    rj = float(env.get_reward(j))
    return i if ri >= rj else j


def _empirical_borda_from_nu_hat(nu_hat: np.ndarray) -> np.ndarray:
    K = nu_hat.shape[0]
    denom = max(1, K - 1)
    b = np.zeros(K, dtype=float)
    for k in range(K):
        s = 0.0
        for j in range(K):
            if j == k:
                continue
            s += float(nu_hat[k, j])
        b[k] = s / denom
    return b


def run_fsh(env: Environment, cfg: AlgoConfig, algo_seed: int) -> Tuple[int, Dict[str, Any]]:
    """
    Runs your FusionSequentialHalving as-is (with elimination).
    Tries to enable round_trace if supported (trace_enabled=True).
    """
    import inspect
    sig = inspect.signature(FusionSequentialHalving.__init__)

    kwargs = dict(
        env=env,
        total_budget=int(cfg.total_budget),
        alpha=float(cfg.alpha),
        gamma=float(cfg.gamma),
        episode_keep=float(cfg.episode_keep),
    )
    if "seed" in sig.parameters:
        kwargs["seed"] = int(algo_seed)
    if "trace_enabled" in sig.parameters:
        kwargs["trace_enabled"] = True

    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    algo = FusionSequentialHalving(**kwargs)
    algo.run()

    final = int(algo.get_candidate_set()[0]) if hasattr(algo, "get_candidate_set") else -1

    # derive duels/rewards from history if available
    duels = rewards = spent = 0
    history = None
    if hasattr(algo, "get_history"):
        history = algo.get_history()
    elif hasattr(algo, "history"):
        history = getattr(algo, "history")
    if history is not None:
        duels = sum(1 for h in history if len(h) > 0 and h[0] == "D")
        rewards = sum(1 for h in history if len(h) > 0 and h[0] == "R")
        spent = duels + rewards

    round_trace = getattr(algo, "round_trace", None)
    rounds_executed = getattr(algo, "round", None)

    extra = {
        "spent_budget": int(getattr(algo, "spent_budget", spent if spent > 0 else cfg.total_budget)),
        "duels": int(duels),
        "rewards": int(rewards),
        "rounds_executed": int(rounds_executed) if rounds_executed is not None else None,
        "round_trace": round_trace,  # list[dict] if supported
        "baseline_score": None,
        "baseline_components": None,
    }
    return final, extra


def run_fsh_no_elim_mix(env: Environment, total_budget: int, alpha: float, algo_seed: int) -> Tuple[int, Dict[str, Any]]:
    """
    Baseline: same schedule as FSH (rounds + duel/reward split),
    but never eliminate arms. Final decision:
        score_k = (1-alpha)*borda_k + alpha*mu_hat_k
    where borda_k computed from empirical nu_hat accumulated over all duels.
    """
    rng = np.random.default_rng(int(algo_seed))
    K = int(env.number_of_bandits)
    C = list(range(K))

    num_rounds = 1 if K <= 1 else int(math.ceil(math.log2(K)))
    budget_per_round = int(math.ceil(int(total_budget) / num_rounds))

    # reward stats
    N = np.zeros(K, dtype=np.int64)
    reward_sum = np.zeros(K, dtype=np.float64)
    mu_hat = np.zeros(K, dtype=np.float64)

    # duel stats
    M = np.zeros((K, K), dtype=np.int64)
    W = np.zeros((K, K), dtype=np.int64)
    nu_hat = np.full((K, K), 0.5, dtype=np.float64)
    np.fill_diagonal(nu_hat, 0.5)

    spent = 0
    duels = 0
    rewards = 0
    round_trace: List[Dict[str, Any]] = []
    r = 0

    while spent < total_budget:
        r += 1
        remaining = total_budget - spent
        T_r = min(budget_per_round, remaining)

        nD = int(math.floor(T_r * (1.0 - float(alpha))))
        nR = int(T_r - nD)

        # duel phase
        for _ in range(nD):
            i = int(rng.choice(C))
            j = int(rng.choice([x for x in C if x != i]))
            winner = _duel_winner(env, rng, i, j)

            M[i, j] += 1
            M[j, i] += 1
            if winner == i:
                W[i, j] += 1
            else:
                W[j, i] += 1

            nu_hat[i, j] = W[i, j] / M[i, j]
            nu_hat[j, i] = W[j, i] / M[j, i]
            duels += 1

        # reward phase
        for _ in range(nR):
            a = int(rng.choice(C))
            rwd = float(env.get_reward(a))
            N[a] += 1
            reward_sum[a] += rwd
            mu_hat[a] = reward_sum[a] / N[a]
            rewards += 1

        spent += T_r

        # logging (top3 by current borda, top3 by mu)
        borda = _empirical_borda_from_nu_hat(nu_hat)
        top3_borda = sorted([(int(i), float(borda[i])) for i in range(K)], key=lambda x: x[1], reverse=True)[:3]
        top3_mu = sorted([(int(i), float(mu_hat[i])) for i in range(K)], key=lambda x: x[1], reverse=True)[:3]

        round_trace.append({
            "round": int(r),
            "T_r": int(T_r),
            "n_r": int(K),
            "nD": int(nD),
            "nR": int(nR),
            "candidate_pre": C,
            "C_tilde": C,
            "candidate_post": C,
            "top3_borda": top3_borda,
            "top3_mu_hat": top3_mu,
            "spent_budget_after": int(spent),
        })

    # final combined score
    borda_final = _empirical_borda_from_nu_hat(nu_hat)
    score = (1.0 - float(alpha)) * borda_final + float(alpha) * mu_hat
    final = int(np.argmax(score))

    # for md: show top5 final score
    top_score = sorted([(int(i), float(score[i])) for i in range(K)], key=lambda x: x[1], reverse=True)[:5]

    extra = {
        "spent_budget": int(spent),
        "duels": int(duels),
        "rewards": int(rewards),
        "rounds_executed": int(r),
        "round_trace": round_trace,
        "baseline_score": top_score,
        "baseline_components": {
            "alpha": float(alpha),
            "mu_hat": mu_hat.tolist(),
            "borda_hat": borda_final.tolist(),
        },
    }
    return final, extra


def write_seed_md(
    md_path: Path,
    *,
    cfg: AlgoConfig,
    seed: int,
    perm: np.ndarray,
    best_arm: int,
    final_arm: int,
    correct: int,
    env_seed: int,
    algo_seed: int,
    extra: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Run Log")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- config_id: `{cfg.config_id()}`")
    lines.append(f"- algo: `{cfg.algo}`")
    lines.append(f"- total_budget: {cfg.total_budget}")
    lines.append(f"- alpha: {cfg.alpha}")
    lines.append(f"- gamma: {cfg.gamma}")
    lines.append(f"- episode_keep: {cfg.episode_keep}")
    lines.append(f"- use_paper_dueling: {int(cfg.use_paper_dueling)}")
    lines.append("")
    lines.append("## Seed / Repro")
    lines.append(f"- seed: {seed}")
    lines.append(f"- env_seed: {env_seed}")
    lines.append(f"- algo_seed: {algo_seed}")
    lines.append(f"- perm (applied to MU and NU): `{','.join(map(str, perm.tolist()))}`")
    lines.append("")
    lines.append("## Result")
    lines.append(f"- best_arm (after permutation): {best_arm}")
    lines.append(f"- final_arm: {final_arm}")
    lines.append(f"- correct: {correct}")
    lines.append(f"- spent_budget: {extra.get('spent_budget')}")
    lines.append(f"- duels: {extra.get('duels')}")
    lines.append(f"- rewards: {extra.get('rewards')}")
    if extra.get("rounds_executed") is not None:
        lines.append(f"- rounds_executed: {extra.get('rounds_executed')}")
    lines.append("")

    if cfg.algo == "FSH_NO_ELIM_MIX":
        lines.append("## Baseline final score (top5)")
        top_score = extra.get("baseline_score", None)
        if isinstance(top_score, list):
            lines.append(f"- score = (1-alpha)*borda + alpha*mu_hat, alpha={cfg.alpha}")
            lines.append(f"- top5_score: {top_score}")
        else:
            lines.append("- (unavailable)")
        lines.append("")

    # Round trace
    rt = extra.get("round_trace", None)
    lines.append("## Round-by-round trace")
    lines.append("")
    if isinstance(rt, list) and len(rt) > 0:
        for rlog in rt:
            r = rlog.get("round", "?")
            lines.append(f"### Round {r}")
            for key in ["T_r", "n_r", "nD", "nR", "candidate_pre", "C_tilde", "candidate_post", "spent_budget_after"]:
                if key in rlog:
                    lines.append(f"- {key}: {rlog[key]}")
            if "top3_borda" in rlog:
                lines.append(f"- top3_borda: {rlog['top3_borda']}")
            if "top3_mu_hat" in rlog:
                lines.append(f"- top3_mu_hat: {rlog['top3_mu_hat']}")
            if "borda" in rlog and isinstance(rlog["borda"], dict):
                items = [(int(k), float(v)) for k, v in rlog["borda"].items()]
                items.sort(key=lambda x: x[1], reverse=True)
                lines.append(f"- top3_borda(dict): {items[:3]}")
            if "mu_hat_snapshot_on_C_tilde" in rlog and isinstance(rlog["mu_hat_snapshot_on_C_tilde"], dict):
                items = [(int(k), float(v)) for k, v in rlog["mu_hat_snapshot_on_C_tilde"].items()]
                items.sort(key=lambda x: x[1], reverse=True)
                lines.append(f"- mu_hat_on_C_tilde(sorted): {items}")
            lines.append("")
    else:
        lines.append("- (trace unavailable: your algorithm may not expose round_trace)")
        lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def plot_heatmap(summary_df: pd.DataFrame, out_png: Path, title: str) -> None:
    sub = summary_df[summary_df["algo"] == "FSH"].copy()
    if sub.empty:
        return
    piv = sub.pivot_table(index="episode_keep", columns="gamma", values="acc_mean", aggfunc="mean")
    piv = piv.sort_index(axis=0).sort_index(axis=1)

    plt.figure()
    plt.imshow(piv.values, aspect="auto", interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(len(piv.columns)), [f"{x:.3f}" for x in piv.columns], rotation=45, ha="right")
    plt.yticks(range(len(piv.index)), [f"{y:.3f}" for y in piv.index])
    plt.xlabel("gamma")
    plt.ylabel("episode_keep")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Worker: one config (loops over ALL seeds)
# -----------------------------------------------------------------------------
def run_one_config_worker(args_tuple: Tuple[AlgoConfig, List[int], Dict[str, int], str, Any]) -> Dict[str, Any]:
    cfg, seeds, base_seeds, outdir, q = args_tuple

    config_dir = Path(outdir) / cfg.folder_name()
    config_dir.mkdir(parents=True, exist_ok=True)
    md_dir = config_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict[str, Any]] = []
    t_cfg0 = time.time()

    try:
        for seed in seeds:
            env_seed = base_seeds["env"] + seed
            algo_seed = base_seeds["algo"] + seed
            perm_seed = base_seeds["perm"] + seed

            # same permutation for this seed across ALL configs
            perm_rng = np.random.default_rng(int(perm_seed))
            perm = perm_rng.permutation(K_PAPER).astype(int)
            mu_p, nu_p = permute_mu_nu(MU_PAPER, NU_PAPER, perm)
            best_arm = int(np.argmax(mu_p))

            # choose dueling matrix
            # use_paper_dueling=True => use permuted nu; else None => env default (mu-diff model)
            nu_used = nu_p if cfg.use_paper_dueling else None

            env = make_env_fixed(env_seed=env_seed, mu=mu_p, nu=nu_used)

            t0 = time.time()
            if cfg.algo == "FSH":
                final_arm, extra = run_fsh(env, cfg, algo_seed=algo_seed)
            elif cfg.algo == "FSH_NO_ELIM_MIX":
                final_arm, extra = run_fsh_no_elim_mix(
                    env,
                    total_budget=cfg.total_budget,
                    alpha=cfg.alpha,
                    algo_seed=algo_seed,
                )
            else:
                raise ValueError(f"Unknown algo: {cfg.algo}")
            t1 = time.time()

            correct = int(final_arm == best_arm)

            # markdown for this (config, seed)
            md_path = md_dir / f"seed_{seed:03d}.md"
            write_seed_md(
                md_path,
                cfg=cfg,
                seed=seed,
                perm=perm,
                best_arm=best_arm,
                final_arm=final_arm,
                correct=correct,
                env_seed=env_seed,
                algo_seed=algo_seed,
                extra=extra,
            )

            per_seed_rows.append({
                "seed": int(seed),
                "env_seed": int(env_seed),
                "algo_seed": int(algo_seed),
                "perm": ",".join(map(str, perm.tolist())),
                "best_arm": int(best_arm),
                "final_arm": int(final_arm),
                "correct": int(correct),
                "wall_time_sec": float(t1 - t0),
                "spent_budget": int(extra.get("spent_budget", cfg.total_budget)),
                "duels": int(extra.get("duels", 0)),
                "rewards": int(extra.get("rewards", 0)),
                "rounds_executed": extra.get("rounds_executed", None),
                "seed_md": str(md_path),
                "baseline_top5_score": json.dumps(extra.get("baseline_score", None), ensure_ascii=False),
            })

            # progress update: one seed done under this config
            if q is not None:
                q.put(1)

    finally:
        # signal config done
        if q is not None:
            q.put("DONE")

    # per-config integration
    per_seed_df = pd.DataFrame(per_seed_rows).sort_values("seed")
    per_seed_csv = config_dir / "per_seed.csv"
    per_seed_df.to_csv(per_seed_csv, index=False, encoding="utf-8-sig")

    acc_mean = float(per_seed_df["correct"].mean()) if len(per_seed_df) else float("nan")
    acc_std = float(per_seed_df["correct"].std()) if len(per_seed_df) else float("nan")
    n = int(len(per_seed_df))

    t_cfg1 = time.time()
    summary = {
        "config_id": cfg.config_id(),
        "folder": str(config_dir),
        "algo": cfg.algo,
        "use_paper_dueling": int(cfg.use_paper_dueling),
        "total_budget": int(cfg.total_budget),
        "alpha": float(cfg.alpha),
        "gamma": None if cfg.gamma is None else float(cfg.gamma),
        "episode_keep": None if cfg.episode_keep is None else float(cfg.episode_keep),
        "n_seeds": n,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "wall_time_total_sec": float(t_cfg1 - t_cfg0),
        "per_seed_csv": str(per_seed_csv),
    }

    (config_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_budget", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--seeds", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--base_env_seed", type=int, default=12345)
    parser.add_argument("--base_algo_seed", type=int, default=67890)
    parser.add_argument("--base_perm_seed", type=int, default=24680)
    parser.add_argument("--outdir", type=str, default="", help="Default: ./log/<timestamp>")
    args = parser.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir.strip() or os.path.join(os.path.dirname(__file__), "log", ts)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    integ_dir = outdir_path / "integration"
    integ_dir.mkdir(parents=True, exist_ok=True)

    # configs: grid + baseline
    gammas = [0.55, 0.60, 0.65, 1 / math.sqrt(2), 0.75, 0.80, 0.85, 0.90]
    keeps  = [0.50, 0.60, 1 / math.sqrt(2), 0.75]

    configs: List[AlgoConfig] = []

    # FSH grid (duel on/off)
    for use_paper_dueling in [True, False]:
        for g in gammas:
            for k in keeps:
                if k <= g:
                    configs.append(AlgoConfig(
                        algo="FSH",
                        total_budget=int(args.total_budget),
                        alpha=float(args.alpha),
                        gamma=float(g),
                        episode_keep=float(k),
                        use_paper_dueling=bool(use_paper_dueling),
                    ))

    # Baseline: same schedule, no elimination, final = (1-alpha)*borda + alpha*reward
    for use_paper_dueling in [True, False]:
        configs.append(AlgoConfig(
            algo="FSH_NO_ELIM_MIX",
            total_budget=int(args.total_budget),
            alpha=float(args.alpha),
            gamma=None,
            episode_keep=None,
            use_paper_dueling=bool(use_paper_dueling),
        ))

    seeds = list(range(int(args.seeds)))
    base_seeds = {"env": int(args.base_env_seed), "algo": int(args.base_algo_seed), "perm": int(args.base_perm_seed)}

    (outdir_path / "config.json").write_text(json.dumps({
        "timestamp": ts,
        "K": K_PAPER,
        "mu": MU_PAPER.tolist(),
        "nu_shape": list(NU_PAPER.shape),
        "total_budget": args.total_budget,
        "alpha": args.alpha,
        "seeds": args.seeds,
        "workers": args.workers,
        "base_seeds": base_seeds,
        "num_configs": len(configs),
        "config_ids": [c.config_id() for c in configs],
        "gammas": gammas,
        "keeps": keeps,
        "baseline": "FSH_NO_ELIM_MIX: final = (1-alpha)*borda + alpha*mu_hat",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # multiprocessing by config
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    q = manager.Queue()

    tasks = [(cfg, seeds, base_seeds, str(outdir_path), q) for cfg in configs]

    total_units = len(configs) * len(seeds)

    pool = ctx.Pool(processes=int(args.workers))
    async_results = [pool.apply_async(run_one_config_worker, (t,)) for t in tasks]
    pool.close()

    done_cfg = 0
    done_units = 0

    if tqdm is not None:
        pbar = tqdm(total=total_units, desc="Running (config x seed)", dynamic_ncols=True)
    else:
        pbar = None
        print(f"[INFO] total work units = {total_units}")

    try:
        while done_cfg < len(configs):
            msg = q.get()
            if msg == "DONE":
                done_cfg += 1
            else:
                inc = int(msg)
                done_units += inc
                if pbar is not None:
                    pbar.update(inc)
                else:
                    if done_units % max(1, total_units // 20) == 0:
                        print(f"[PROGRESS] {done_units}/{total_units}")
    finally:
        if pbar is not None:
            pbar.close()

    pool.join()

    # collect per-config summaries
    summaries: List[Dict[str, Any]] = [ar.get() for ar in async_results]

    summary_df = pd.DataFrame(summaries).sort_values(["algo", "use_paper_dueling", "gamma", "episode_keep"])
    summary_csv = integ_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # heatmaps for FSH only
    for duel_flag in [1, 0]:
        sub = summary_df[(summary_df["algo"] == "FSH") & (summary_df["use_paper_dueling"] == duel_flag)].copy()
        if not sub.empty:
            out_png = integ_dir / f"heatmap_acc_FSH_usePaperDuel{duel_flag}.png"
            plot_heatmap(sub, out_png, title=f"FSH Acc heatmap (use_paper_dueling={duel_flag})")

    # baseline plot
    base = summary_df[summary_df["algo"] == "FSH_NO_ELIM_MIX"].copy()
    if not base.empty:
        plt.figure()
        xs = [f"usePaperDuel={int(x)}" for x in base["use_paper_dueling"].tolist()]
        ys = base["acc_mean"].tolist()
        plt.plot(xs, ys, marker="o")
        plt.ylim(0.0, 1.0)
        plt.ylabel("acc_mean")
        plt.title("Baseline: FSH_NO_ELIM_MIX accuracy")
        plt.tight_layout()
        plt.savefig(integ_dir / "baseline_fsh_no_elim_mix.png", dpi=220)
        plt.close()

    print(f"[OK] log dir      : {outdir_path}")
    print(f"[OK] integration : {integ_dir}")
    print(f"[OK] summary.csv : {summary_csv}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
