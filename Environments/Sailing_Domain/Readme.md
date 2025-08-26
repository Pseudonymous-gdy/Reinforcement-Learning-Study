Awesome—here’s a tight, “what-to-code” scaffold for reproducing the **sailing domain** experiments and plugging in your tree-search (e.g., UCT) or your new algorithm.

# 0) Repo layout (minimal)

```
sailing/
  env/
    sailing_env.py         # Grid, wind/tack state, 7 actions, step()
    wind_model.py          # Stochastic wind dynamics
    cost_model.py          # Action-costs in [1, 8.6], forbid opposite-wind move
  planners/
    uct.py                 # UCT/your planner, episodic rollouts
    baselines.py           # Uniform MC, (optional) ARTDP/PG-ID stubs
  eval/
    vstar_qstar.py         # Offline value iteration → V*, Q*
    rollout_value.py       # V̂(x) = (1+ε(x))·V*(x) with ε~U[-0.1,0.1]
    metrics.py             # avg error: Q*(s,π(s)) - V*(s)
  experiments/
    run_experiment.py      # Orchestrates training/eval across grid sizes
    config.yaml            # Grid sizes, seeds, ε, bias scales, etc.
  utils/
    rng.py, logging_utils.py, serialize.py
  tests/
    test_env.py, test_planner.py, test_eval.py
```

# 1) Environment

**`SailingEnv`**

* **State**: `(x, y, wind_dir, tack)`; wind\_dir ∈ {0..7}, tack ∈ {0,1,2}. (State space ≈ `24 × grid_cells`.)
* **Actions (7)**: neighbor moves; **action opposite to wind is forbidden**. **Cost ∈ \[1, 8.6]**, depends on action vs. wind.
* **API**:

  * `reset(start=None, goal=None) -> state`
  * `step(state, action) -> (next_state, cost, done, info)`
  * `valid_actions(state) -> List[int]`
* **Wind/Tack update**: delegate to `WindModel`.

**`WindModel`**

* `sample_next(wind_dir) -> wind_dir'` (e.g., simple Markov or iid model)
* Optional: parameters for gustiness, directional persistence.

**`CostModel`**

* `cost(action, wind_dir, tack) -> float` in **\[1, 8.6]**; forbid opposite-wind action.

# 2) Optimal values for evaluation

**`vstar_qstar.py`**

* **Goal**: compute **V\*** and **Q\*** offline (value iteration for SSP).
* API:

  * `compute_vq_star(env, goal, gamma=1.0) -> (V_star, Q_star)`
* **Usage**: provides ground truth to score planners: average of **`Q*(s, a_rec) - V*(s)`** over 1000 random states.

**`rollout_value.py`**

* Build rollout evaluator **V̂** for simulations:
  `V_hat(x) = (1 + ε(x)) · V_star(x)`, with ε(x) \~ U\[-0.1, 0.1], fixed per run (seeded).

# 3) Planner(s)

**`uct.py`** (use this as a template for your algorithm too)

* Data: `N(s)`, `N(s,a)`, `Q(s,a)`; UCB/selection rule.
* **Stopping rule per episode**: stop with probability **1 / N\_s(t)** to emulate iterative deepening.
* **Bias scaling**: allow multiplier (e.g., **×10** as in the paper) via config.
* API:

  * `plan(state0, budget_calls) -> action` (returns best root action)
  * `run_episode(state0) -> (total_cost, calls_used)`
  * `select(s) -> a` (UCB/your policy)
  * `expand/evaluate(s) -> leaf_value` (uses V̂)
  * `backup(path, returns)`
* **Bookkeeping**: return **number of simulator calls** to enable fair comparisons. (In these studies, “samples” = simulator calls.)

**`baselines.py`**

* `UniformMCPlanner` (uniform action sampling).
* (Optional) stubs/wrappers for ARTDP, PG-ID if you’ll compare.

# 4) Evaluation & Experiment harness

**`metrics.py`**

* `avg_qgap(policy, V_star, Q_star, states) = mean[ Q*(s, π(s)) - V*(s) ]` over **1000** random states.

**`run_experiment.py`**

* Loop grid sizes (e.g., **2×2 to 40×40**), random seeds, planners.
* For each setting:

  1. build env, compute **V\***, **Q\*** once
  2. generate fixed set of 1000 random states
  3. for each planner: run with budget (or until target error), log:

     * avg error, **simulator calls** to reach error threshold (e.g., 0.1)
* Save CSVs + plots.

# 5) Tests (fast sanity checks)

* `test_env.py`: forbidden-action check; cost bounds; terminal handling.
* `test_vstar.py`: tiny grid (e.g., 2×2) where V\* is hand-checkable.
* `test_planner.py`: Q-table updates monotone, stop rule stats ≈ 1/N.

---

## Minimal function skeletons (Python-ish)

```python
# env/sailing_env.py
class SailingEnv:
    def __init__(self, width, height, wind_model, cost_model, start, goal):
        ...

    def reset(self, start=None, goal=None):
        ...

    def valid_actions(self, s):
        ...

    def step(self, s, a):
        # returns s', cost, done, info
        ...

# eval/vstar_qstar.py
def compute_vq_star(env, tol=1e-6):
    # value iteration for SSP -> V*, Q*
    return V_star, Q_star

# eval/rollout_value.py
def make_Vhat(V_star, rng):
    # fix ε(x) ~ U[-0.1, 0.1] per state
    ...

# planners/uct.py
class UCT:
    def __init__(self, env, C, bias_scale=10.0, rng=None):
        ...

    def plan(self, s0, budget_calls):
        ...

    def run_episode(self, s0):
        # stop with prob 1/N_s(t); count simulator calls
        ...

    def select(self, s):
        # argmax_a Q(s,a) + UCB(s,a)
        ...
```

That’s it—this is the smallest set of parts your students need to write to replicate the sailing-domain benchmarks and drop in your new tree-search. If you want, I can turn this into a ready-to-run cookiecutter with stub files.
