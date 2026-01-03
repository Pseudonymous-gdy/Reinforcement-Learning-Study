# test2.py
# ─────────────────────────────────────────────────────────────────────────────
# Extended experiment runner for K=32/64 with multiple mean-structure scenarios.
# Parallelization: BY CONFIG (each worker runs all seeds for one config)
#
# Adds Plotly interactive views DIRECTLY from this script:
# - 3D scatter (alpha,gamma,keep) colored by acc_mean
# - Small-multiples heatmaps: gamma × keep for each alpha (one HTML)
# - Best-over-keep heatmap (alpha × gamma) for max acc, and argmax keep heatmap
# - Parallel coordinates (alpha,gamma,keep -> acc)
# - Baseline acc vs alpha (interactive line)
# - index.html linking all interactive outputs
#
# Key constraints:
# - For a fixed seed: env_seed/algo_seed/mu_perm_seed are shared across ALL configs
# - MU is generated per (scenario,K,seed) deterministically, then permuted per (seed,K)
# - No paper dueling (dueling_means=None), env uses its internal Bernoulli-based dueling
# - Baseline: same round schedule as FSH, no elimination, final score:
#       score_k = (1 - alpha) * borda_k + alpha * mu_hat_k
#
# IMPORTANT:
# - Your Environment.set_means() sorts means; that breaks permutation experiments.
#   This script FORCE-OVERRIDES env.bandit_means directly to preserve identity.
# - Extreme params alpha/gamma/keep ∈ {0,1} may break your class impl. We fallback to
#   a reference implementation inferred from DuelThenReward semantics.
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
import json
import math
import time
import argparse
import datetime as dt
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# progress bar (main process only)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# headless matplotlib plots (optional; still useful for quick PNG exports)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Plotly (interactive HTML)
PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except Exception:
    PLOTLY_AVAILABLE = False


# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
if __package__ in (None, ""):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Fusing2.env import Environment
from Fusing2.sequential_halving import FusionSequentialHalving


# -----------------------------------------------------------------------------
# Scenario generation
# -----------------------------------------------------------------------------
SCENARIOS = [
    "fusing_bump",         # equally spaced with tiny bump above base top
    "uniform_gap",         # equally spaced, larger range
    "uniform_tiny_gap",    # equally spaced, tiny gaps (hard)
    "unique_best",         # μ1 > μ2=...=μK
    "gaps_increasing",     # Δ_{i,i+1} < Δ_{i+1,i+2}
    "gaps_decreasing",     # Δ_{i,i+1} > Δ_{i+1,i+2}
    "random",              # random generation, sorted descending
]


def _clip01(mu: np.ndarray) -> np.ndarray:
    return np.clip(mu, 0.0, 1.0)


def generate_mu(scenario: str, K: int, seed: int, base_mu_seed: int) -> np.ndarray:
    """
    Returns mu sorted descending (ranked arms), then we will permute indices.
    Deterministic given (scenario, K, seed, base_mu_seed).
    """
    sid = SCENARIOS.index(scenario)
    rng = np.random.default_rng(int(base_mu_seed + 100000 * K + 1000 * sid + seed))

    if scenario == "fusing_bump":
        # base grid top at 0.85, then tiny bump above it
        hi = 0.85
        lo = 0.10
        eps = 0.001
        mu = np.linspace(hi, lo, K)
        mu[0] = min(0.999, hi + eps)

    elif scenario == "uniform_gap":
        hi = 0.90
        lo = 0.05
        mu = np.linspace(hi, lo, K)

    elif scenario == "uniform_tiny_gap":
        center = 0.60
        delta = 0.0015 if K == 64 else 0.0025
        mu = np.array([center - i * delta for i in range(K)], dtype=float)
        mu = np.maximum(mu, 0.05)

    elif scenario == "unique_best":
        mu1 = 0.70
        mu_rest = 0.68
        mu = np.array([mu1] + [mu_rest] * (K - 1), dtype=float)

    elif scenario == "gaps_increasing":
        hi = 0.90
        lo = 0.05
        total_drop = hi - lo
        a = 2.0 * total_drop / (K * (K - 1))  # sum_{i=1..K-1} a*i = total_drop
        mu = np.zeros(K, dtype=float)
        mu[0] = hi
        for i in range(1, K):
            mu[i] = mu[i - 1] - a * i

    elif scenario == "gaps_decreasing":
        hi = 0.90
        lo = 0.05
        total_drop = hi - lo
        a = 2.0 * total_drop / (K * (K - 1))  # sum_{i=1..K-1} a*(K-i) = total_drop
        mu = np.zeros(K, dtype=float)
        mu[0] = hi
        for i in range(1, K):
            mu[i] = mu[i - 1] - a * (K - i)

    elif scenario == "random":
        lo, hi = 0.05, 0.95
        mu = rng.uniform(lo, hi, size=K)
        mu.sort()
        mu = mu[::-1]
        mu[0] = min(0.999, mu[0] + 1e-3)  # ensure unique best

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    mu = _clip01(mu)
    mu = np.sort(mu)[::-1]
    return mu


def permute_mu(mu_sorted: np.ndarray, seed: int, K: int, base_perm_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply deterministic permutation per (seed, K).
    Returns (mu_permuted, perm).
    """
    rng = np.random.default_rng(int(base_perm_seed + 100000 * K + seed))
    perm = rng.permutation(K).astype(int)
    mu_p = mu_sorted[perm]
    return mu_p, perm


# -----------------------------------------------------------------------------
# Config definitions
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AlgoConfig:
    scenario: str
    K: int
    algo: str  # "FSH" or "BASELINE_NO_ELIM_MIX"
    base_budget_16: int
    alpha: float
    gamma: Optional[float] = None
    episode_keep: Optional[float] = None
    use_class_impl: bool = True

    def total_budget(self) -> int:
        # linear budget scaling with K/16
        scale = self.K / 16.0
        return int(round(self.base_budget_16 * scale))

    def config_id(self) -> str:
        if self.algo == "FSH":
            return (f"{self.scenario}_K{self.K}_FSH_"
                    f"a{self.alpha:.3f}_g{float(self.gamma):.3f}_k{float(self.episode_keep):.3f}_"
                    f"impl{int(self.use_class_impl)}")
        if self.algo == "BASELINE_NO_ELIM_MIX":
            return f"{self.scenario}_K{self.K}_BASELINE_NO_ELIM_MIX_a{self.alpha:.3f}"
        raise ValueError(f"Unknown algo={self.algo}")

    def folder_name(self) -> str:
        return self.config_id().replace(".", "p")


# -----------------------------------------------------------------------------
# Environment forcing (avoid mean sorting)
# -----------------------------------------------------------------------------
def _force_set_means_no_sort(env: Environment, mu: np.ndarray) -> None:
    """
    Your Environment.set_means() sorts means -> breaks arm identity.
    Force override bandit_means to preserve order.
    """
    if not hasattr(env, "bandit_means"):
        raise RuntimeError("Environment must expose bandit_means.")
    env.bandit_means = np.array(mu, dtype=float)

    if hasattr(env, "optimal_mean"):
        env.optimal_mean = float(np.max(mu))

    if hasattr(env, "standard_deviations"):
        sd = getattr(env, "standard_deviations")
        if sd is None or len(np.array(sd)) != len(mu):
            env.standard_deviations = np.zeros(len(mu), dtype=float)


def make_env_fixed(env_seed: int, K: int, mu: np.ndarray) -> Environment:
    # No paper dueling: dueling_means=None
    env = Environment(
        number_of_bandits=int(K),
        distribution="bernoulli",
        seed=int(env_seed),
        dueling_means=None,
    )
    _force_set_means_no_sort(env, mu)

    arr = np.array(env.bandit_means, dtype=float)
    if not np.allclose(arr, mu, atol=1e-12):
        raise RuntimeError("Failed to force means without reordering.")
    return env


# -----------------------------------------------------------------------------
# Duel utilities + Borda
# -----------------------------------------------------------------------------
def _duel_winner(env: Environment, rng: np.random.Generator, i: int, j: int) -> int:
    if hasattr(env, "duel"):
        w = int(env.duel(i, j))
        if w not in (i, j):
            raise ValueError(f"env.duel must return i or j, got {w}")
        return w
    out = float(env.get_dueling(i, j))
    if out == 1.0:
        return i
    if out == 0.0:
        return j
    return int(rng.choice([i, j]))  # tie


def _borda_on_set(nu_hat: np.ndarray, C: List[int]) -> Dict[int, float]:
    denom = max(1, len(C) - 1)
    borda: Dict[int, float] = {}
    for i in C:
        s = 0.0
        for j in C:
            if j == i:
                continue
            s += float(nu_hat[i, j])
        borda[int(i)] = float(s / denom)
    return borda


def _topk_by_score(
    items: List[int],
    primary: Dict[int, float],
    secondary: Optional[Dict[int, float]] = None,
    k: int = 1
) -> List[int]:
    """
    Deterministic tie-break:
    - primary desc
    - secondary desc (if provided)
    - index asc
    """
    if secondary is None:
        secondary = {int(i): 0.0 for i in items}

    def key(i: int):
        return (-float(primary.get(i, 0.0)), -float(secondary.get(i, 0.0)), int(i))

    out = sorted([int(x) for x in items], key=key)
    return out[:max(1, int(k))]


# -----------------------------------------------------------------------------
# Reference FSH runner (robust at extreme params)
# Inferred semantics:
# - Each round budget: budget_per_round = ceil(total_budget / ceil(log2 K))
# - Duel phase uses nD=floor(T_r*(1-alpha)) and keeps ceil(gamma*|C|)
# - Reward phase uses nR and keeps floor(episode_keep*|C_tilde|)
# - if no reward samples, rank by borda
# -----------------------------------------------------------------------------
def run_fsh_reference(
    env: Environment,
    K: int,
    total_budget: int,
    alpha: float,
    gamma: float,
    episode_keep: float,
    algo_seed: int,
) -> Tuple[int, Dict[str, Any]]:
    rng = np.random.default_rng(int(algo_seed))

    alpha = float(alpha)
    gamma = float(gamma)
    episode_keep = float(episode_keep)

    # duel stats
    M = np.zeros((K, K), dtype=np.int64)
    W = np.zeros((K, K), dtype=np.int64)
    nu_hat = np.full((K, K), 0.5, dtype=float)
    np.fill_diagonal(nu_hat, 0.5)

    # reward stats
    N = np.zeros(K, dtype=np.int64)
    S = np.zeros(K, dtype=float)
    mu_hat = np.zeros(K, dtype=float)

    spent = 0
    duels = 0
    rewards = 0

    C = list(range(K))
    round_trace: List[Dict[str, Any]] = []

    num_rounds = int(math.ceil(math.log2(max(2, K))))
    budget_per_round = int(math.ceil(total_budget / max(1, num_rounds)))

    rounds = 0
    while spent < total_budget and len(C) > 1 and rounds < (num_rounds + 8):
        rounds += 1
        remaining = total_budget - spent
        T_r = min(budget_per_round, remaining)

        nD = int(math.floor(T_r * (1.0 - alpha)))
        nR = int(T_r - nD)

        C_pre = list(C)

        # duel phase
        if nD <= 0:
            C_tilde = list(C_pre)
        else:
            for _ in range(nD):
                i = int(rng.choice(C_pre))
                j = int(rng.choice([x for x in C_pre if x != i]))
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

            borda_pre = _borda_on_set(nu_hat, C_pre)
            keep_duel = max(1, int(math.ceil(gamma * len(C_pre))))
            C_tilde = _topk_by_score(C_pre, primary=borda_pre, secondary=None, k=keep_duel)

        # reward phase
        if nR > 0:
            for _ in range(nR):
                a = int(rng.choice(C_tilde))
                rwd = float(env.get_reward(a))
                N[a] += 1
                S[a] += rwd
                mu_hat[a] = S[a] / N[a]
                rewards += 1

            borda_tilde = _borda_on_set(nu_hat, C_tilde)
            keep_reward = max(1, int(math.floor(episode_keep * len(C_tilde))))
            C_post = _topk_by_score(
                C_tilde,
                primary={i: float(mu_hat[i]) for i in C_tilde},
                secondary=borda_tilde,
                k=keep_reward
            )
        else:
            borda_tilde = _borda_on_set(nu_hat, C_tilde)
            keep_reward = max(1, int(math.floor(episode_keep * len(C_tilde))))
            C_post = _topk_by_score(C_tilde, primary=borda_tilde, secondary=None, k=keep_reward)

        spent += T_r

        borda_for_log = _borda_on_set(nu_hat, C_pre)
        top3_borda = sorted([(i, borda_for_log[i]) for i in borda_for_log], key=lambda x: x[1], reverse=True)[:3]

        mu_snapshot = {int(i): float(mu_hat[i]) for i in C_tilde if N[i] > 0}
        mu_sorted = sorted(mu_snapshot.items(), key=lambda x: x[1], reverse=True)

        round_trace.append({
            "round": int(rounds),
            "T_r": int(T_r),
            "n_r": int(len(C_pre)),
            "nD": int(nD),
            "nR": int(nR),
            "candidate_pre": C_pre,
            "C_tilde": C_tilde,
            "candidate_post": C_post,
            "top3_borda": top3_borda,
            "mu_hat_on_C_tilde(sorted)": mu_sorted,
            "spent_budget_after": int(spent),
        })

        C = list(C_post)

    # final
    if len(C) == 1:
        final = int(C[0])
    else:
        borda_final = _borda_on_set(nu_hat, C)
        final = _topk_by_score(
            C,
            primary={i: float(mu_hat[i]) for i in C},
            secondary=borda_final,
            k=1
        )[0]

    extra = {
        "impl_used": "ref",
        "spent_budget": int(spent),
        "duels": int(duels),
        "rewards": int(rewards),
        "rounds_executed": int(rounds),
        "round_trace": round_trace,
    }
    return final, extra


# -----------------------------------------------------------------------------
# Baseline: same schedule, no elimination; final score=(1-alpha)*borda + alpha*mu_hat
# -----------------------------------------------------------------------------
def run_baseline_no_elim_mix(
    env: Environment,
    K: int,
    total_budget: int,
    alpha: float,
    algo_seed: int,
) -> Tuple[int, Dict[str, Any]]:
    rng = np.random.default_rng(int(algo_seed))
    alpha = float(alpha)

    # duel stats
    M = np.zeros((K, K), dtype=np.int64)
    W = np.zeros((K, K), dtype=np.int64)
    nu_hat = np.full((K, K), 0.5, dtype=float)
    np.fill_diagonal(nu_hat, 0.5)

    # reward stats
    N = np.zeros(K, dtype=np.int64)
    S = np.zeros(K, dtype=float)
    mu_hat = np.zeros(K, dtype=float)

    num_rounds = int(math.ceil(math.log2(max(2, K))))
    budget_per_round = int(math.ceil(total_budget / max(1, num_rounds)))

    spent = 0
    duels = 0
    rewards = 0
    rounds = 0
    round_trace: List[Dict[str, Any]] = []
    C = list(range(K))

    while spent < total_budget and rounds < num_rounds:
        rounds += 1
        remaining = total_budget - spent
        T_r = min(budget_per_round, remaining)

        nD = int(math.floor(T_r * (1.0 - alpha)))
        nR = int(T_r - nD)

        for _ in range(nD):
            i = int(rng.integers(0, K))
            j = int(rng.integers(0, K - 1))
            if j >= i:
                j += 1
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

        for _ in range(nR):
            a = int(rng.integers(0, K))
            rwd = float(env.get_reward(a))
            N[a] += 1
            S[a] += rwd
            mu_hat[a] = S[a] / N[a]
            rewards += 1

        spent += T_r

        borda = _borda_on_set(nu_hat, C)
        top3_borda = sorted([(i, borda[i]) for i in borda], key=lambda x: x[1], reverse=True)[:3]
        top3_mu = sorted([(i, float(mu_hat[i])) for i in range(K)], key=lambda x: x[1], reverse=True)[:3]

        round_trace.append({
            "round": int(rounds),
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

    borda_final = np.array([_borda_on_set(nu_hat, C)[i] for i in range(K)], dtype=float)
    score = (1.0 - alpha) * borda_final + alpha * mu_hat
    final = int(np.argmax(score))

    top5 = sorted([(i, float(score[i])) for i in range(K)], key=lambda x: x[1], reverse=True)[:5]

    extra = {
        "impl_used": "baseline",
        "spent_budget": int(spent),
        "duels": int(duels),
        "rewards": int(rewards),
        "rounds_executed": int(rounds),
        "round_trace": round_trace,
        "baseline_top5_score": top5,
    }
    return final, extra


# -----------------------------------------------------------------------------
# Try class implementation, fallback to reference
# -----------------------------------------------------------------------------
def run_fsh_class_or_fallback(
    env: Environment,
    cfg: AlgoConfig,
    algo_seed: int,
) -> Tuple[int, Dict[str, Any]]:
    K = cfg.K
    total_budget = cfg.total_budget()
    alpha = float(cfg.alpha)
    gamma = float(cfg.gamma)
    episode_keep = float(cfg.episode_keep)

    if not cfg.use_class_impl:
        return run_fsh_reference(env, K, total_budget, alpha, gamma, episode_keep, algo_seed)

    try:
        import inspect
        sig = inspect.signature(FusionSequentialHalving.__init__)
        kwargs = dict(
            env=env,
            total_budget=int(total_budget),
            alpha=float(alpha),
            gamma=float(gamma),
            episode_keep=float(episode_keep),
        )
        if "seed" in sig.parameters:
            kwargs["seed"] = int(algo_seed)
        if "trace_enabled" in sig.parameters:
            kwargs["trace_enabled"] = True
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        algo = FusionSequentialHalving(**kwargs)
        algo.run()

        final = int(algo.get_candidate_set()[0]) if hasattr(algo, "get_candidate_set") else -1

        round_trace = getattr(algo, "round_trace", None)
        rounds_executed = getattr(algo, "round", None)

        duels = rewards = 0
        hist = None
        if hasattr(algo, "get_history"):
            hist = algo.get_history()
        elif hasattr(algo, "history"):
            hist = getattr(algo, "history")
        if hist is not None:
            duels = sum(1 for h in hist if len(h) > 0 and h[0] == "D")
            rewards = sum(1 for h in hist if len(h) > 0 and h[0] == "R")

        extra = {
            "impl_used": "class",
            "spent_budget": int(getattr(algo, "spent_budget", total_budget)),
            "duels": int(duels),
            "rewards": int(rewards),
            "rounds_executed": int(rounds_executed) if rounds_executed is not None else None,
            "round_trace": round_trace if isinstance(round_trace, list) else [{"round": None, "note": "class impl had no round_trace"}],
        }
        return final, extra

    except Exception as e:
        final, extra = run_fsh_reference(env, K, total_budget, alpha, gamma, episode_keep, algo_seed)
        extra["fallback_reason"] = repr(e)
        return final, extra


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def write_seed_md(
    md_path: Path,
    *,
    cfg: AlgoConfig,
    seed: int,
    env_seed: int,
    algo_seed: int,
    perm: np.ndarray,
    mu_perm: np.ndarray,
    best_arm: int,
    final_arm: int,
    correct: int,
    extra: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# Run Log")
    lines.append("")
    lines.append("## Config")
    lines.append(f"- config_id: `{cfg.config_id()}`")
    lines.append(f"- scenario: `{cfg.scenario}`")
    lines.append(f"- K: {cfg.K}")
    lines.append(f"- algo: `{cfg.algo}`")
    lines.append(f"- total_budget: {cfg.total_budget()}")
    lines.append(f"- alpha: {cfg.alpha}")
    lines.append(f"- gamma: {cfg.gamma}")
    lines.append(f"- episode_keep: {cfg.episode_keep}")
    lines.append(f"- impl_used: `{extra.get('impl_used')}`")
    if "fallback_reason" in extra:
        lines.append(f"- fallback_reason: `{extra['fallback_reason']}`")
    lines.append("")
    lines.append("## Seed / Repro")
    lines.append(f"- seed: {seed}")
    lines.append(f"- env_seed: {env_seed}")
    lines.append(f"- algo_seed: {algo_seed}")
    lines.append(f"- perm(seed,K): `{','.join(map(str, perm.tolist()))}`")
    lines.append("")
    lines.append("## MU (permuted arms)")
    mu_str = ",".join([f"{x:.6f}" for x in mu_perm.tolist()])
    lines.append(f"- mu: `{mu_str}`")
    lines.append("")
    lines.append("## Result")
    lines.append(f"- best_arm (argmax mu): {best_arm}")
    lines.append(f"- final_arm: {final_arm}")
    lines.append(f"- correct: {correct}")
    lines.append(f"- spent_budget: {extra.get('spent_budget')}")
    lines.append(f"- duels: {extra.get('duels')}")
    lines.append(f"- rewards: {extra.get('rewards')}")
    if extra.get("rounds_executed") is not None:
        lines.append(f"- rounds_executed: {extra.get('rounds_executed')}")
    lines.append("")

    if cfg.algo == "BASELINE_NO_ELIM_MIX":
        lines.append("## Baseline final score (top5)")
        top5 = extra.get("baseline_top5_score", None)
        if isinstance(top5, list):
            lines.append(f"- score = (1-alpha)*borda + alpha*mu_hat, alpha={cfg.alpha}")
            lines.append(f"- top5_score: {top5}")
        lines.append("")

    lines.append("## Round-by-round trace")
    lines.append("")
    rt = extra.get("round_trace", None)
    if isinstance(rt, list) and len(rt) > 0:
        for rlog in rt:
            r = rlog.get("round", "?")
            lines.append(f"### Round {r}")
            for key in ["T_r", "n_r", "nD", "nR", "candidate_pre", "C_tilde", "candidate_post",
                        "top3_borda", "top3_mu_hat",
                        "mu_hat_on_C_tilde(sorted)", "spent_budget_after", "note"]:
                if key in rlog:
                    lines.append(f"- {key}: {rlog[key]}")
            lines.append("")
    else:
        lines.append("- (trace unavailable)")
        lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Matplotlib plotting (optional)
# -----------------------------------------------------------------------------
def plot_static_heatmaps_by_alpha(df_fsh: pd.DataFrame, out_dir: Path, title_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas = sorted(df_fsh["alpha"].unique().tolist())
    for a in alphas:
        df_a = df_fsh[df_fsh["alpha"] == a].copy()
        if df_a.empty:
            continue
        piv = df_a.pivot_table(index="episode_keep", columns="gamma", values="acc_mean", aggfunc="mean")
        piv = piv.sort_index(axis=0).sort_index(axis=1)

        plt.figure()
        plt.imshow(piv.values, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
        plt.colorbar()
        plt.xticks(range(len(piv.columns)), [f"{x:.2f}" for x in piv.columns], rotation=45, ha="right")
        plt.yticks(range(len(piv.index)), [f"{y:.2f}" for y in piv.index])
        plt.xlabel("gamma")
        plt.ylabel("episode_keep")
        plt.title(f"{title_prefix} | alpha={a:.2f}")
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmap_alpha_{a:.2f}.png", dpi=220)
        plt.close()


def plot_static_3d_scatter(df_fsh: pd.DataFrame, out_png: Path, title: str) -> None:
    if df_fsh.empty:
        return
    xs = df_fsh["alpha"].astype(float).values
    ys = df_fsh["gamma"].astype(float).values
    zs = df_fsh["episode_keep"].astype(float).values
    cs = df_fsh["acc_mean"].astype(float).values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(xs, ys, zs, c=cs)
    fig.colorbar(sc)
    ax.set_xlabel("alpha")
    ax.set_ylabel("gamma")
    ax.set_zlabel("episode_keep")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# -----------------------------------------------------------------------------
# Plotly interactive outputs
# -----------------------------------------------------------------------------
def _write_plotly_html(fig: "go.Figure", out_html: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    # self-contained HTML (no external CDN dependency)
    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)


def fig_plotly_3d_scatter(df_fsh: pd.DataFrame, scenario: str, K: int) -> "go.Figure":
    sub = df_fsh[(df_fsh["scenario"] == scenario) & (df_fsh["K"] == K)].copy()
    sub = sub.dropna(subset=["gamma", "episode_keep"])
    fig = px.scatter_3d(
        sub,
        x="alpha", y="gamma", z="episode_keep",
        color="acc_mean",
        hover_data={"acc_mean":":.4f", "alpha":":.3f", "gamma":":.3f", "episode_keep":":.3f"},
        title=f"FSH 3D scatter | scenario={scenario}, K={K}",
    )
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(scene=dict(
        xaxis_title="alpha",
        yaxis_title="gamma",
        zaxis_title="episode_keep",
    ))
    return fig


def fig_plotly_small_multiples_heatmap(df_fsh: pd.DataFrame, scenario: str, K: int) -> "go.Figure":
    """
    One HTML: multiple panels, each panel is heatmap (gamma x keep) at a fixed alpha.
    This is usually more readable than 3D.
    """
    sub = df_fsh[(df_fsh["scenario"] == scenario) & (df_fsh["K"] == K)].copy()
    sub = sub.dropna(subset=["gamma", "episode_keep"])
    alphas = sorted(sub["alpha"].unique().tolist())
    gammas = sorted(sub["gamma"].unique().tolist())
    keeps = sorted(sub["episode_keep"].unique().tolist())

    n = len(alphas)
    cols = 3
    rows = int(math.ceil(n / cols))

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"alpha={a:.2f}" for a in alphas],
        horizontal_spacing=0.06, vertical_spacing=0.12,
    )

    # consistent colorbar across all
    # build each alpha heatmap as a matrix (keeps x gammas)
    for idx, a in enumerate(alphas):
        df_a = sub[sub["alpha"] == a].copy()
        piv = df_a.pivot_table(index="episode_keep", columns="gamma", values="acc_mean", aggfunc="mean")
        piv = piv.reindex(index=keeps, columns=gammas)

        r = idx // cols + 1
        c = idx % cols + 1

        fig.add_trace(
            go.Heatmap(
                z=piv.values,
                x=[float(x) for x in gammas],
                y=[float(y) for y in keeps],
                coloraxis="coloraxis",
                hovertemplate="alpha=%{meta:.2f}<br>gamma=%{x:.2f}<br>keep=%{y:.2f}<br>acc=%{z:.4f}<extra></extra>",
                meta=float(a),
            ),
            row=r, col=c
        )

        fig.update_xaxes(title_text="gamma", row=r, col=c)
        fig.update_yaxes(title_text="episode_keep", row=r, col=c)

    fig.update_layout(
        title=f"FSH heatmaps by alpha | scenario={scenario}, K={K}",
        coloraxis=dict(colorscale="Viridis", cmin=0.0, cmax=1.0, colorbar=dict(title="acc")),
        height=320 * rows + 140,
        width=1100,
    )
    return fig


def fig_plotly_best_over_keep(df_fsh: pd.DataFrame, scenario: str, K: int) -> Tuple["go.Figure", "go.Figure"]:
    """
    Two heatmaps:
    - best_acc(alpha,gamma) = max_keep acc
    - best_keep(alpha,gamma) = argmax_keep acc
    """
    sub = df_fsh[(df_fsh["scenario"] == scenario) & (df_fsh["K"] == K)].copy()
    sub = sub.dropna(subset=["gamma", "episode_keep"])
    if sub.empty:
        return go.Figure(), go.Figure()

    rows = []
    for (a, g), grp in sub.groupby(["alpha", "gamma"]):
        idx = grp["acc_mean"].idxmax()
        rows.append({
            "alpha": float(a),
            "gamma": float(g),
            "best_acc": float(grp.loc[idx, "acc_mean"]),
            "best_keep": float(grp.loc[idx, "episode_keep"]),
        })
    agg = pd.DataFrame(rows)

    pivot_acc = agg.pivot(index="alpha", columns="gamma", values="best_acc").sort_index().sort_index(axis=1)
    pivot_keep = agg.pivot(index="alpha", columns="gamma", values="best_keep").sort_index().sort_index(axis=1)

    fig_acc = go.Figure(
        data=go.Heatmap(
            z=pivot_acc.values,
            x=[float(x) for x in pivot_acc.columns],
            y=[float(y) for y in pivot_acc.index],
            colorscale="Viridis",
            zmin=0.0, zmax=1.0,
            hovertemplate="alpha=%{y:.2f}<br>gamma=%{x:.2f}<br>best_acc=%{z:.4f}<extra></extra>",
        )
    )
    fig_acc.update_layout(
        title=f"Best acc over keep | scenario={scenario}, K={K}",
        xaxis_title="gamma",
        yaxis_title="alpha",
        height=650,
        width=900,
    )

    fig_keep = go.Figure(
        data=go.Heatmap(
            z=pivot_keep.values,
            x=[float(x) for x in pivot_keep.columns],
            y=[float(y) for y in pivot_keep.index],
            colorscale="Plasma",
            hovertemplate="alpha=%{y:.2f}<br>gamma=%{x:.2f}<br>argmax_keep=%{z:.2f}<extra></extra>",
        )
    )
    fig_keep.update_layout(
        title=f"Argmax keep (achieving best acc) | scenario={scenario}, K={K}",
        xaxis_title="gamma",
        yaxis_title="alpha",
        height=650,
        width=900,
    )

    return fig_acc, fig_keep


def fig_plotly_parallel_coords(df_fsh: pd.DataFrame, scenario: str, K: int) -> "go.Figure":
    sub = df_fsh[(df_fsh["scenario"] == scenario) & (df_fsh["K"] == K)].copy()
    sub = sub.dropna(subset=["gamma", "episode_keep"])
    if sub.empty:
        return go.Figure()

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=sub["acc_mean"], colorscale="Viridis", cmin=0.0, cmax=1.0, showscale=True),
            dimensions=[
                dict(label="alpha", values=sub["alpha"]),
                dict(label="gamma", values=sub["gamma"]),
                dict(label="episode_keep", values=sub["episode_keep"]),
                dict(label="acc_mean", values=sub["acc_mean"]),
            ],
        )
    )
    fig.update_layout(
        title=f"FSH parallel coordinates | scenario={scenario}, K={K}",
        height=700,
        width=1100,
    )
    return fig


def fig_plotly_baseline_acc_vs_alpha(df_base: pd.DataFrame, scenario: str, K: int) -> "go.Figure":
    sub = df_base[(df_base["scenario"] == scenario) & (df_base["K"] == K)].copy()
    if sub.empty:
        return go.Figure()
    sub = sub.sort_values("alpha")
    fig = px.line(
        sub,
        x="alpha",
        y="acc_mean",
        markers=True,
        title=f"Baseline (no elim, mix score) acc vs alpha | scenario={scenario}, K={K}",
    )
    fig.update_layout(yaxis=dict(range=[0.0, 1.0]))
    return fig


def write_plotly_index(index_path: Path, entries: List[Tuple[str, Path]]) -> None:
    """
    entries: list of (title, html_path)
    """
    lines = []
    lines.append("<html><head><meta charset='utf-8'><title>Plotly Outputs</title></head><body>")
    lines.append("<h1>Interactive Outputs (Plotly)</h1>")
    lines.append("<ul>")
    for title, path in entries:
        rel = os.path.relpath(str(path), str(index_path.parent))
        lines.append(f"<li><a href='{rel}' target='_blank'>{title}</a></li>")
    lines.append("</ul>")
    lines.append("</body></html>")
    index_path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Worker: one config (loops over all seeds)
# -----------------------------------------------------------------------------
def run_one_config_worker(args_tuple: Tuple[AlgoConfig, List[int], Dict[str, int], str, Any]) -> Dict[str, Any]:
    cfg, seeds, base_seeds, outdir, q = args_tuple

    root = Path(outdir) / f"{cfg.scenario}_K{cfg.K}"
    config_dir = root / cfg.folder_name()
    md_dir = config_dir / "md"
    md_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict[str, Any]] = []
    t0_cfg = time.time()
    total_budget = cfg.total_budget()

    try:
        for seed in seeds:
            env_seed = base_seeds["env"] + seed
            algo_seed = base_seeds["algo"] + seed

            mu_sorted = generate_mu(cfg.scenario, cfg.K, seed, base_seeds["mu"])
            mu_perm, perm = permute_mu(mu_sorted, seed, cfg.K, base_seeds["perm"])
            best_arm = int(np.argmax(mu_perm))

            env = make_env_fixed(env_seed=env_seed, K=cfg.K, mu=mu_perm)

            t0 = time.time()
            if cfg.algo == "FSH":
                final_arm, extra = run_fsh_class_or_fallback(env, cfg, algo_seed=algo_seed)
            elif cfg.algo == "BASELINE_NO_ELIM_MIX":
                final_arm, extra = run_baseline_no_elim_mix(
                    env, K=cfg.K, total_budget=total_budget, alpha=cfg.alpha, algo_seed=algo_seed
                )
            else:
                raise ValueError(f"Unknown algo: {cfg.algo}")
            t1 = time.time()

            correct = int(final_arm == best_arm)

            md_path = md_dir / f"seed_{seed:03d}.md"
            write_seed_md(
                md_path,
                cfg=cfg,
                seed=seed,
                env_seed=env_seed,
                algo_seed=algo_seed,
                perm=perm,
                mu_perm=mu_perm,
                best_arm=best_arm,
                final_arm=final_arm,
                correct=correct,
                extra=extra,
            )

            per_seed_rows.append({
                "seed": int(seed),
                "scenario": cfg.scenario,
                "K": int(cfg.K),
                "algo": cfg.algo,
                "alpha": float(cfg.alpha),
                "gamma": None if cfg.gamma is None else float(cfg.gamma),
                "episode_keep": None if cfg.episode_keep is None else float(cfg.episode_keep),
                "total_budget": int(total_budget),
                "env_seed": int(env_seed),
                "algo_seed": int(algo_seed),
                "perm": ",".join(map(str, perm.tolist())),
                "best_arm": int(best_arm),
                "final_arm": int(final_arm),
                "correct": int(correct),
                "impl_used": extra.get("impl_used"),
                "fallback_reason": extra.get("fallback_reason", None),
                "spent_budget": int(extra.get("spent_budget", total_budget)),
                "duels": int(extra.get("duels", 0)),
                "rewards": int(extra.get("rewards", 0)),
                "rounds_executed": extra.get("rounds_executed", None),
                "wall_time_sec": float(t1 - t0),
                "seed_md": str(md_path),
            })

            if q is not None:
                q.put(1)

    finally:
        if q is not None:
            q.put("DONE")

    per_seed_df = pd.DataFrame(per_seed_rows).sort_values("seed")
    config_dir.mkdir(parents=True, exist_ok=True)
    per_seed_csv = config_dir / "per_seed.csv"
    per_seed_df.to_csv(per_seed_csv, index=False, encoding="utf-8-sig")

    acc_mean = float(per_seed_df["correct"].mean()) if len(per_seed_df) else float("nan")
    acc_std = float(per_seed_df["correct"].std()) if len(per_seed_df) else float("nan")
    n = int(len(per_seed_df))
    fallback_rate = float((per_seed_df["impl_used"] == "ref").mean()) if "impl_used" in per_seed_df.columns else 0.0

    t1_cfg = time.time()
    summary = {
        "config_id": cfg.config_id(),
        "folder": str(config_dir),
        "scenario": cfg.scenario,
        "K": int(cfg.K),
        "algo": cfg.algo,
        "total_budget": int(total_budget),
        "alpha": float(cfg.alpha),
        "gamma": None if cfg.gamma is None else float(cfg.gamma),
        "episode_keep": None if cfg.episode_keep is None else float(cfg.episode_keep),
        "n_seeds": n,
        "acc_mean": acc_mean,
        "acc_std": acc_std,
        "fallback_rate": fallback_rate,
        "wall_time_total_sec": float(t1_cfg - t0_cfg),
        "per_seed_csv": str(per_seed_csv),
    }
    (config_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_budget_16", type=int, default=2000, help="Budget at K=16; scaled linearly for K=32/64.")
    parser.add_argument("--Ks", type=str, default="32,64", help="Comma-separated K list, e.g. 32,64")
    parser.add_argument("--scenarios", type=str, default="all", help=f"Comma-separated scenarios or 'all'. Options: {SCENARIOS}")

    parser.add_argument("--alpha_points", type=int, default=5, help="linspace points for alpha in [0,1] (includes extremes)")
    parser.add_argument("--gamma_points", type=int, default=5, help="linspace points for gamma in [0,1] (includes extremes)")
    parser.add_argument("--keep_points", type=int, default=5, help="linspace points for episode_keep in [0,1] (includes extremes)")

    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))

    parser.add_argument("--base_env_seed", type=int, default=12345)
    parser.add_argument("--base_algo_seed", type=int, default=67890)
    parser.add_argument("--base_perm_seed", type=int, default=24680)
    parser.add_argument("--base_mu_seed", type=int, default=13579)

    parser.add_argument("--use_class_impl", action="store_true", help="Try FusionSequentialHalving class first (fallback to ref).")

    # outputs
    parser.add_argument("--outdir", type=str, default="", help="Default: ./log/<timestamp>")
    parser.add_argument("--no_matplotlib", action="store_true", help="Disable static PNG outputs.")
    parser.add_argument("--plotly", action="store_true", help="Enable Plotly interactive HTML outputs.")

    args = parser.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir.strip() or os.path.join(os.path.dirname(__file__), "log", ts)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    integ_dir = outdir_path / "integration"
    plots_dir = integ_dir / "plots"
    plots_plotly_dir = integ_dir / "plots_plotly"
    integ_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_plotly_dir.mkdir(parents=True, exist_ok=True)

    Ks = [int(x) for x in args.Ks.split(",") if x.strip()]

    if args.scenarios.strip().lower() == "all":
        scenarios = list(SCENARIOS)
    else:
        scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
        for s in scenarios:
            if s not in SCENARIOS:
                raise ValueError(f"Unknown scenario '{s}'. Options: {SCENARIOS}")

    alpha_grid = np.linspace(0.0, 1.0, int(args.alpha_points))
    gamma_grid = np.linspace(0.0, 1.0, int(args.gamma_points))
    keep_grid = np.linspace(0.0, 1.0, int(args.keep_points))

    seeds = list(range(int(args.seeds)))
    base_seeds = {
        "env": int(args.base_env_seed),
        "algo": int(args.base_algo_seed),
        "perm": int(args.base_perm_seed),
        "mu": int(args.base_mu_seed),
    }

    # Build configs
    configs: List[AlgoConfig] = []
    for scenario in scenarios:
        for K in Ks:
            # FSH grid (alpha,gamma,keep)
            for a in alpha_grid:
                for g in gamma_grid:
                    for k in keep_grid:
                        configs.append(AlgoConfig(
                            scenario=scenario,
                            K=int(K),
                            algo="FSH",
                            base_budget_16=int(args.base_budget_16),
                            alpha=float(a),
                            gamma=float(g),
                            episode_keep=float(k),
                            use_class_impl=bool(args.use_class_impl),
                        ))
            # Baseline varies only with alpha
            for a in alpha_grid:
                configs.append(AlgoConfig(
                    scenario=scenario,
                    K=int(K),
                    algo="BASELINE_NO_ELIM_MIX",
                    base_budget_16=int(args.base_budget_16),
                    alpha=float(a),
                    gamma=None,
                    episode_keep=None,
                    use_class_impl=False,
                ))

    # snapshot
    (outdir_path / "config.json").write_text(json.dumps({
        "timestamp": ts,
        "Ks": Ks,
        "scenarios": scenarios,
        "base_budget_16": args.base_budget_16,
        "seeds": args.seeds,
        "workers": args.workers,
        "base_seeds": base_seeds,
        "alpha_grid": alpha_grid.tolist(),
        "gamma_grid": gamma_grid.tolist(),
        "keep_grid": keep_grid.tolist(),
        "use_class_impl": bool(args.use_class_impl),
        "num_configs": len(configs),
        "plotly_enabled": bool(args.plotly),
        "matplotlib_enabled": (not bool(args.no_matplotlib)),
        "plotly_available": bool(PLOTLY_AVAILABLE),
        "note": "No paper dueling used (dueling_means=None).",
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
    summary_df = pd.DataFrame(summaries)

    # global summary csv
    summary_csv = integ_dir / "summary.csv"
    summary_df.sort_values(["scenario", "K", "algo", "alpha", "gamma", "episode_keep"], na_position="last") \
              .to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # Split for plotting
    df_fsh = summary_df[summary_df["algo"] == "FSH"].copy()
    df_base = summary_df[summary_df["algo"] == "BASELINE_NO_ELIM_MIX"].copy()

    # ─────────────────────────────────────────────────────────────
    # Static PNG plots (optional)
    # ─────────────────────────────────────────────────────────────
    if not args.no_matplotlib:
        for scenario in scenarios:
            for K in Ks:
                sub_fsh = df_fsh[(df_fsh["scenario"] == scenario) & (df_fsh["K"] == K)].copy()
                if not sub_fsh.empty:
                    hm_dir = plots_dir / f"{scenario}_K{K}" / "heatmaps"
                    plot_static_heatmaps_by_alpha(sub_fsh, hm_dir, title_prefix=f"{scenario} K={K} FSH")

                    out3d = plots_dir / f"{scenario}_K{K}" / "scatter_3d.png"
                    out3d.parent.mkdir(parents=True, exist_ok=True)
                    plot_static_3d_scatter(sub_fsh, out3d, title=f"{scenario} K={K} FSH: acc_mean over (alpha,gamma,keep)")

                sub_base = df_base[(df_base["scenario"] == scenario) & (df_base["K"] == K)].copy()
                if not sub_base.empty:
                    outb = plots_dir / f"{scenario}_K{K}" / "baseline_acc_vs_alpha.png"
                    outb.parent.mkdir(parents=True, exist_ok=True)
                    plt.figure()
                    sub_base = sub_base.sort_values("alpha")
                    plt.plot(sub_base["alpha"].values, sub_base["acc_mean"].values, marker="o")
                    plt.ylim(0.0, 1.0)
                    plt.xlabel("alpha")
                    plt.ylabel("acc_mean")
                    plt.title(f"{scenario} K={K} Baseline acc vs alpha")
                    plt.tight_layout()
                    plt.savefig(outb, dpi=220)
                    plt.close()

    # ─────────────────────────────────────────────────────────────
    # Plotly interactive HTML outputs (recommended for analysis)
    # ─────────────────────────────────────────────────────────────
    plotly_entries: List[Tuple[str, Path]] = []

    if args.plotly:
        if not PLOTLY_AVAILABLE:
            print("[WARN] Plotly is not installed. Install with: pip install plotly")
        else:
            for scenario in scenarios:
                for K in Ks:
                    sub_fsh = df_fsh[(df_fsh["scenario"] == scenario) & (df_fsh["K"] == K)].copy()
                    sub_base = df_base[(df_base["scenario"] == scenario) & (df_base["K"] == K)].copy()

                    tag_dir = plots_plotly_dir / f"{scenario}_K{K}"
                    tag_dir.mkdir(parents=True, exist_ok=True)

                    if not sub_fsh.empty:
                        # 3D scatter
                        f3d = fig_plotly_3d_scatter(df_fsh, scenario, K)
                        p3d = tag_dir / "fsh_scatter_3d.html"
                        _write_plotly_html(f3d, p3d)
                        plotly_entries.append((f"[FSH] 3D scatter | {scenario} K={K}", p3d))

                        # small-multiples heatmaps by alpha
                        fhm = fig_plotly_small_multiples_heatmap(df_fsh, scenario, K)
                        phm = tag_dir / "fsh_heatmaps_by_alpha.html"
                        _write_plotly_html(fhm, phm)
                        plotly_entries.append((f"[FSH] Heatmaps by alpha | {scenario} K={K}", phm))

                        # best-over-keep + argmax keep
                        facc, fkeep = fig_plotly_best_over_keep(df_fsh, scenario, K)
                        pacc = tag_dir / "fsh_best_acc_over_keep.html"
                        pkeep = tag_dir / "fsh_argmax_keep.html"
                        _write_plotly_html(facc, pacc)
                        _write_plotly_html(fkeep, pkeep)
                        plotly_entries.append((f"[FSH] Best acc over keep | {scenario} K={K}", pacc))
                        plotly_entries.append((f"[FSH] Argmax keep | {scenario} K={K}", pkeep))

                        # parallel coordinates
                        fpc = fig_plotly_parallel_coords(df_fsh, scenario, K)
                        ppc = tag_dir / "fsh_parallel_coords.html"
                        _write_plotly_html(fpc, ppc)
                        plotly_entries.append((f"[FSH] Parallel coords | {scenario} K={K}", ppc))

                    if not sub_base.empty:
                        fb = fig_plotly_baseline_acc_vs_alpha(df_base, scenario, K)
                        pb = tag_dir / "baseline_acc_vs_alpha.html"
                        _write_plotly_html(fb, pb)
                        plotly_entries.append((f"[Baseline] acc vs alpha | {scenario} K={K}", pb))

            # index
            index_html = plots_plotly_dir / "index.html"
            write_plotly_index(index_html, plotly_entries)
            print(f"[OK] Plotly index: {index_html}")

    print(f"[OK] log dir      : {outdir_path}")
    print(f"[OK] summary.csv : {summary_csv}")
    print(f"[OK] static plots: {plots_dir} (enabled={not args.no_matplotlib})")
    print(f"[OK] plotly plots : {plots_plotly_dir} (enabled={bool(args.plotly)}; available={bool(PLOTLY_AVAILABLE)})")
    print("[INFO] No paper dueling used (dueling_means=None).")


if __name__ == "__main__":
    mp.freeze_support()
    main()
