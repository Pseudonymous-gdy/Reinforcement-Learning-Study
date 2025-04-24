# Test.py  – compare UCB‑Tuned, UCB‑2 (α=0.001) and Epsilon‑Greedy (c=0.1)
import numpy as np
import matplotlib.pyplot as plt
import System
import UCB_Tuned
import UCB_2
import Epsilon_Greedy
import UCB_Normal   # optional – in case you want to add it later

# ---------------- environments -----------------
envs = [
    System.System(2,  [0.9, 0.6]),
    System.System(2,  [0.9, 0.8]),
    System.System(2,  [0.55, 0.45]),
    System.System(10, [0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]),
    System.System(10, [0.9, 0.8, 0.8, 0.8, 0.7, 0.7, 0.7, 0.6, 0.6, 0.6]),
    System.System(10, [0.9, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]),
    System.System(10, [0.55, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]),
]

# ---------------- experiment grid -----------------
X      = np.linspace(1, 5, 100)              # 10¹ … 10⁵ plays
horizon_max = int(10 ** X[-1])
plays   = (10 ** X).astype(int)
runs    = 100                              # Monte‑Carlo averages

# ---------------- helper to gather a path -----------------
def get_path(agent, horizon=horizon_max):
    """Return regret path, opt‑pull path sampled at `plays`."""
    if hasattr(agent, "simulate_path"):
        reg, opt = agent.simulate_path(horizon=horizon, runs=runs)
        return reg[plays - 1], opt[plays - 1]
    # fallback – simulate many horizons incrementally (slower, for UCB_Tuned)
    Y_reg, Y_opt = [], []
    for H in plays:
        r, o = agent.simulate(H, runs=1)     # UCB_Tuned already averages 100 seeds
        Y_reg.append(r)
        Y_opt.append(o)
    return np.array(Y_reg), np.array(Y_opt)

# ---------------- main loop -----------------
for idx, env in enumerate(envs, 1):
    print(f"\n=== Environment {idx}/{len(envs)}  n={env.n}  p={env.p} ===")

    algos = {
        "UCB‑Tuned":      UCB_Tuned.UCB_Tuned(env),
        "UCB‑2 α=0.001":  UCB_2.UCB2(env, alpha=0.01),
        "ε‑Greedy c=0.1": Epsilon_Greedy.EpsilonGreedy(env, c=0.1),
        "UCB‑Normal":     UCB_Normal.UCBNormal(env)
    }

    paths_reg, paths_opt = {}, {}
    for name, algo in algos.items():
        print(f"  ↳ running {name} …", end="", flush=True)
        reg, opt = get_path(algo)
        paths_reg[name] = reg
        paths_opt[name] = opt
        print(" done.")

    # ---------- Plot: cumulative regret ----------
    plt.figure(figsize=(7, 4))
    for name, y in paths_reg.items():
        plt.semilogx(plays, y, label=name)
    plt.title(f"Environment #{idx} · cumulative regret")
    plt.xlabel("plays (log scale)")
    plt.ylabel("average cumulative regret")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- Plot: optimal‑arm pull fraction ----------
    plt.figure(figsize=(7, 4))
    for name, y in paths_opt.items():
        plt.semilogx(plays, y, label=name)
    plt.title(f"Environment #{idx} · best‑arm pull fraction")
    plt.xlabel("plays (log scale)")
    plt.ylabel("fraction of pulls on optimal arm")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
