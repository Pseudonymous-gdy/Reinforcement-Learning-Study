# Coding Agent Task: Pilot Validation for Phased Joint-MLE Mixed-XY BAI

## 1. Mission

Implement and run a **pilot validation experiment** for the second algorithm only:

> **Phased Joint-MLE Mixed-XY BAI with Efficient Rounding**

The goal is not to produce a full paper-level experiment. The goal is to determine whether the fusion algorithm is worth continuing.

The pilot must answer:

1. Can the fusion algorithm stop under fixed confidence?
2. Is the fusion algorithm's success rate close to the requested confidence level?
3. Is the fusion algorithm's weighted cost better than both reward-only and duel-only in at least some representative cases?

The key success criterion is:

\[
C_f(\omega_R)<\min\{C_R(\omega_R), C_D(\omega_R)\}.
\]

Where:

\[
C_R(\omega_R)=\omega_R T_R,
\]

\[
C_D(\omega_R)=T_D,
\]

\[
C_f(\omega_R)=\omega_R T_r^{(f)}+T_d^{(f)}.
\]

Use:

\[
\omega_D=1,
\qquad
\omega_R\in\{1,2,4,8,16\}.
\]

Important: **Do not implement a cost-aware version yet.** In this pilot, \(\omega_R\) is used only for post-hoc cost evaluation. The algorithm itself should not change when \(\omega_R\) changes.

---

## 2. Fixed Environment Parameters

Use the following data-generation model for all experiments.

Reward feedback:

\[
Y_i^{(R)}=x_i^\top\theta^\star+\epsilon,
\qquad
\epsilon\sim\mathcal N(0,1).
\]

Dueling feedback:

\[
Y_{ij}^{(D)}\sim \mathrm{Bernoulli}
\left(
\sigma((x_i-x_j)^\top\theta^\star)
\right),
\]

where:

\[
\sigma(u)=\frac{1}{1+e^{-u}}.
\]

Fixed parameters:

```text
sigma_R = 1
gamma = 1
omega_D = 1
omega_R_values = [1, 2, 4, 8, 16]
delta_values = [0.1, 0.05, 0.01]
seed_count = 500
seeds = 0, 1, ..., 499
T_max = 100000
```

The confidence level \(\delta\) is an algorithm input and must be rerun for each value. The cost weights \(\omega_R\) are not algorithm inputs in this pilot and should be evaluated post-hoc from saved \(T_r,T_d\).

---

## 3. Feedback Regimes

For every instance and every \(\delta\), run three regimes.

### 3.1 Reward-only

Only reward queries are allowed.

\[
\mathcal Q=\mathcal Q_R,
\qquad
\mathcal Q_D=\varnothing.
\]

Record:

\[
T_R=T_r.
\]

### 3.2 Duel-only

Only duel queries are allowed.

\[
\mathcal Q=\mathcal Q_D,
\qquad
\mathcal Q_R=\varnothing.
\]

Record:

\[
T_D=T_d.
\]

### 3.3 Fusion

Both reward and duel queries are allowed.

\[
\mathcal Q=\mathcal Q_R\cup\mathcal Q_D.
\]

Record:

\[
T_f=T_r^{(f)}+T_d^{(f)}.
\]

Also record the fusion duel fraction:

\[
p_D=\frac{T_d^{(f)}}{T_r^{(f)}+T_d^{(f)}}.
\]

If \(p_D\approx 0\), fusion effectively degenerates to reward-only. If \(p_D\approx 1\), fusion effectively degenerates to duel-only.

---

## 4. Pilot Instances

Implement exactly these three instance families.

### 4.1 Case 1: Unit-vector instance

Purpose: stochastic-bandit-like extreme case.

Use:

\[
K=d=n=10,
\qquad
x_i=e_i.
\]

Set:

\[
\theta^\star=(0,-0.35,-0.45,-1.5,\ldots,-1.5)\in\mathbb R^{10}.
\]

So:

```text
theta_star[0] = 0
theta_star[1] = -0.35
theta_star[2] = -0.45
theta_star[3:] = -1.5
```

The optimal arm is arm 0 if using zero-based indexing.

This case has two near-optimal competitors and several clearly suboptimal arms.

---

### 4.2 Case 2: Special linear-geometry instance

Purpose: test whether the algorithm works beyond unit vectors and can exploit linear geometry.

Use:

\[
d=10.
\]

The arm set is:

\[
X=\{e_1,\ldots,e_d,x_{d+1},x_{d+2}\}.
\]

Define:

\[
x_{d+1}=\cos(0.55)e_1+\sin(0.55)e_2,
\]

\[
x_{d+2}=\cos(0.65)e_1+\sin(0.65)e_3.
\]

Set:

\[
\theta^\star=e_1.
\]

The optimal arm is \(e_1\), i.e. arm 0 in zero-based indexing.

The two near-optimal arms are \(x_{d+1}\) and \(x_{d+2}\), with gaps approximately:

\[
1-\cos(0.55)\approx 0.147,
\]

\[
1-\cos(0.65)\approx 0.204.
\]

---

### 4.3 Case 3: Controlled general random instance

Purpose: test whether the algorithm works on non-handcrafted random geometry.

Use:

\[
K=20,
\qquad
 d=5.
\]

For each seed, generate an instance as follows:

1. Sample arms:

\[
x_i\sim\mathcal N(0,I_d),
\qquad
x_i\leftarrow \frac{x_i}{\|x_i\|_2}.
\]

2. Sample parameter:

\[
\theta^\star\sim\mathcal N(0,I_d),
\qquad
\theta^\star\leftarrow \frac{\theta^\star}{\|\theta^\star\|_2}.
\]

3. Compute:

\[
i^\star=\arg\max_i x_i^\top\theta^\star.
\]

4. Compute gaps:

\[
\Delta_i=(x_{i^\star}-x_i)^\top\theta^\star,
\qquad i\ne i^\star.
\]

5. Accept the instance only if:

\[
\Delta_{\min}\in[0.12,0.25],
\]

and:

\[
|\{i\ne i^\star:\Delta_i\le 0.35\}|\ge 3.
\]

If the sampled instance does not satisfy these filters, resample using the same master seed stream until it does. Save the number of attempts in the raw output.

Important: for a fixed `seed`, use the same generated general instance for reward-only, duel-only, and fusion. This makes the regimes comparable.

---

## 5. Algorithm to Implement

Implement the practical pilot version of **Phased Joint-MLE Mixed-XY BAI with Efficient Rounding**.

### 5.1 Feasible query sets

For an arm set \(\{x_i\}_{i=1}^K\):

Reward query set:

\[
\mathcal Q_R=\{(R,i):i\in\mathcal I_R\}.
\]

Duel query set:

\[
\mathcal Q_D=\{(D,i,j):(i,j)\in E_D\}.
\]

For this pilot, use full availability before applying the regime filter:

```text
I_R = all arms
E_D = all unordered pairs (i, j), i < j
```

Then apply the regime:

```text
reward-only: Q = Q_R
duel-only:   Q = Q_D
fusion:      Q = Q_R union Q_D
```

### 5.2 Raw measurement direction

For every query \(a\in\mathcal Q\), define:

\[
v(a)=
\begin{cases}
x_i, & a=(R,i),\\
x_i-x_j, & a=(D,i,j).
\end{cases}
\]

### 5.3 Mixed measurement set and target directions

Define:

\[
\mathcal X_{\rm mix}=\{x_i:i\in\mathcal I_R\}\cup\{x_i-x_j:(i,j)\in E_D\}.
\]

Define:

\[
\mathcal Y=\{x_i-x_j:i\ne j\}.
\]

Define:

\[
\mathcal V=\operatorname{span}(\mathcal X_{\rm mix}),
\]

\[
\mathcal Y^{\rm id}=\mathcal Y\cap \mathcal V.
\]

Practical implementation: determine membership in \(\mathcal V\) numerically using projection residual tolerance `1e-7`. Since the pilot uses full reward/duel availability within each regime, most target directions should be identifiable. Still compute and log `identifiable_ratio = len(Y_id) / len(Y)`.

### 5.4 Log-likelihood and MLE

The phased version uses a joint MLE over \(\theta\). Each observation has a type \(\rho_s\in\{R,D\}\), direction \(v_s\), and scalar outcome \(Y_s\).

Use log-likelihood:

\[
\ell(\theta)=\sum_s \left[Y_s v_s^\top\theta-b_{\rho_s}(v_s^\top\theta)\right],
\]

where:

\[
b_R(u)=\frac{u^2}{2},
\qquad
b_D(u)=\log(1+e^u).
\]

For reward observations, store the observed Gaussian reward value as \(Y_s\). This gives the term:

\[
Y_s v_s^\top\theta - \frac{(v_s^\top\theta)^2}{2},
\]

which is equivalent to Gaussian regression with variance 1 up to constants.

For duel observations, store \(Y_s\in\{0,1\}\), where \(Y_s=1\) means the first arm in \((i,j)\) wins.

Implementation requirements:

- Use stable `logaddexp(0, u)` for \(\log(1+e^u)\).
- Use `scipy.optimize.minimize` on the negative log-likelihood.
- Default optimizer: `L-BFGS-B`.
- Warm-start MLE from previous phase's \(\hat\theta\).
- Add tiny ridge regularization only for numerical stability if needed:

\[
\ell_{\rm reg}(\theta)=\ell(\theta)-\frac{\lambda}{2}\|\theta\|_2^2,
\qquad
\lambda=10^{-6}.
\]

Log optimizer status and objective value.

### 5.5 Fisher increments

For each action \(a\), define \(J_m(a)\) at current \(\hat\theta\).

Reward:

\[
J_m(R,i)=x_i x_i^\top.
\]

Duel:

\[
J_m(D,i,j)=w((x_i-x_j)^\top\hat\theta)(x_i-x_j)(x_i-x_j)^\top,
\]

where:

\[
w(u)=\sigma(u)(1-\sigma(u)).
\]

### 5.6 Design matrix

For a distribution \(q\in\Delta(\mathcal Q)\):

\[
B_m(q)=\sum_{a\in\mathcal Q}q(a)J_m(a).
\]

Use pseudo-inverse with a stable tolerance:

```python
np.linalg.pinv(B, rcond=1e-8)
```

### 5.7 Phase schedule

At phase \(m=1,2,\ldots\):

\[
\delta_m=\frac{6\delta}{\pi^2m^2},
\]

\[
\bar x_m=d\log 5+\log\left(\frac{\pi^2m^2}{3\delta}\right),
\]

\[
\varepsilon_m=2^{-m}.
\]

Use:

\[
C_{\rm safe}=32e^3(1+\zeta).
\]

Set:

```text
zeta = 0.1
r_zeta = d^2
```

If phase sizes explode, expose `r_zeta` and safety constants in config.

### 5.8 Leverage and target objectives

Define:

\[
L_m(q)=\max_{v\in\mathcal X_{\rm mix}}v^\top B_m(q)^\dagger v,
\]

\[
T_m(q)=\max_{y\in\mathcal Y^{\rm id}}y^\top B_m(q)^\dagger y.
\]

### 5.9 Phase allocation

Compute:

\[
q_m\in\arg\min_{q\in\Delta(\mathcal Q)}
\max\left\{
C_{\rm safe}\bar x_m L_m(q),
128e^3\bar x_m\frac{T_m(q)}{\varepsilon_m^2}
\right\}.
\]

Practical implementation options:

1. First implement SLSQP over simplex constraints.
2. If SLSQP is too slow or unstable, implement a greedy fallback:
   - start from uniform \(q\),
   - repeatedly add small mass to the action that most reduces the current max objective,
   - project back to simplex.
3. Log solver status, objective value, and elapsed time.

Do not silently ignore optimizer failures. If optimizer fails, use the fallback and set `design_solver_status = fallback`.

### 5.10 Phase size

Set:

\[
n_m=\left\lceil
\max\left\{
r(\zeta),
(1+\zeta)C_{\rm safe}\bar x_mL_m(q_m),
(1+\zeta)128e^3\bar x_m\frac{T_m(q_m)}{\varepsilon_m^2}
\right\}
\right\rceil.
\]

Before executing a phase, check whether current total samples plus \(n_m\) would exceed `T_max`. If yes, execute only up to `T_max`, mark `stopped = False`, and terminate the run.

### 5.11 Rounding

For the pilot, exact Fiez-style efficient rounding is not required. Implement deterministic integer rounding:

1. Compute raw counts:

\[
\tilde n_a=n_m q_m(a).
\]

2. Set:

\[
n_a=\lfloor \tilde n_a\rfloor.
\]

3. Distribute the remaining samples to actions with largest fractional parts.

4. Create a fixed phase sequence with exactly \(n_m\) actions.

5. Shuffle the phase sequence using the run RNG to avoid ordering artifacts.

Log the rounding error:

\[
\|n/n_m-q_m\|_1.
\]

### 5.12 Executing an action

Reward action \((R,i)\):

```python
y = x_i @ theta_star + rng.normal(0, 1)
```

Store observation:

```text
rho = R
v = x_i
y = reward_value
```

Duel action \((D,i,j)\):

```python
p = sigmoid((x_i - x_j) @ theta_star)
y = rng.binomial(1, p)
```

Store observation:

```text
rho = D
v = x_i - x_j
y = 1 if i wins over j else 0
```

### 5.13 Empirical Fisher and stopping rule

After every phase, recompute \(\hat\theta_m\) and empirical Fisher:

\[
\widehat H_m
=
\sum_{s:\rho_s=R}v_sv_s^\top
+
\sum_{s:\rho_s=D}w(v_s^\top\hat\theta_m)v_sv_s^\top.
\]

Let:

\[
\hat i_m=\arg\max_i x_i^\top\hat\theta_m.
\]

Use:

\[
\widehat\beta_m=4e^{3/2}\sqrt{2\bar x_m}.
\]

Stop if for all \(j\ne \hat i_m\):

\[
(x_{\hat i_m}-x_j)^\top\hat\theta_m
>
\widehat\beta_m
\sqrt{(x_{\hat i_m}-x_j)^\top\widehat H_m^\dagger(x_{\hat i_m}-x_j)}.
\]

Return \(\hat i_m\).

If total samples reach `T_max` before stopping, return the current \(\hat i_m\), but mark:

```text
stopped = False
stop_fail = True
```

---

## 6. Burn-in

The algorithm needs a burn-in procedure so that the initial MLE and Fisher matrix are numerically stable.

Implement a simple pilot burn-in:

1. Compute all allowed action directions \(v(a)\).
2. Greedily select actions that increase the rank of the direction matrix until the rank reaches the span dimension of the allowed directions.
3. Query each selected action `burnin_repeats` times.
4. Use:

```text
burnin_repeats = 3
```

If the allowed action set cannot span all of \(\mathbb R^d\), that is acceptable. Work on the identifiable span, use pseudo-inverses, and log the rank.

Record burn-in counts separately:

```text
T_r_burn
T_d_burn
T_r_main
T_d_main
```

Total counts:

```text
T_r = T_r_burn + T_r_main
T_d = T_d_burn + T_d_main
T_total = T_r + T_d
```

---

## 7. Experiment Matrix

Run:

```text
cases = [unit, special, general]
regimes = [reward_only, duel_only, fusion]
delta_values = [0.1, 0.05, 0.01]
seeds = 0..499
```

Total number of runs:

```text
3 cases * 3 regimes * 3 deltas * 500 seeds = 13500 runs
```

Before the full run, support a smoke-test mode:

```text
seeds = 0..19
```

Smoke-test total:

```text
3 * 3 * 3 * 20 = 540 runs
```

Run the smoke test first. Only run 500 seeds after the smoke test produces reasonable results.

---

## 8. Required Outputs

Create an output directory:

```text
outputs/fusion_pilot/
```

### 8.1 Raw results

Save:

```text
outputs/fusion_pilot/raw_results.csv
```

One row per run.

Required columns:

```text
case_name
regime
delta
seed
instance_attempts
K
d
i_star
i_hat
success
stopped
stop_fail
T_r
T_d
T_total
T_r_burn
T_d_burn
T_r_main
T_d_main
final_phase
final_loglik
mle_success
mle_status
design_solver_status
identifiable_ratio
p_D
```

For non-fusion regimes, define:

```text
p_D = 0 for reward-only
p_D = 1 for duel-only
```

### 8.2 Post-hoc cost table

Create:

```text
outputs/fusion_pilot/cost_by_omega.csv
```

For each raw run and each \(\omega_R\in\{1,2,4,8,16\}\), compute:

\[
C(\omega_R)=\omega_R T_r+T_d.
\]

Columns:

```text
case_name
regime
delta
seed
omega_R
omega_D
cost
T_r
T_d
T_total
success
stopped
```

### 8.3 Fusion cost-ratio table

Create:

```text
outputs/fusion_pilot/fusion_cost_ratio.csv
```

For each:

```text
case_name, delta, seed, omega_R
```

match reward-only, duel-only, and fusion runs with the same case, delta, seed.

Compute:

\[
\text{ratio}=\frac{C_f}{\min(C_R,C_D)}.
\]

Also compute:

```text
fusion_wins = ratio < 1
```

Columns:

```text
case_name
delta
seed
omega_R
C_R
C_D
C_f
ratio
fusion_wins
success_R
success_D
success_f
stopped_R
stopped_D
stopped_f
T_R
T_D
T_r_f
T_d_f
T_f
p_D_f
```

### 8.4 Summary tables

Create:

```text
outputs/fusion_pilot/summary_by_case_regime_delta.csv
```

Group by:

```text
case_name, regime, delta
```

Report:

```text
success_rate
stop_fail_rate
median_T_r
q25_T_r
q75_T_r
median_T_d
q25_T_d
q75_T_d
median_T_total
q25_T_total
q75_T_total
median_p_D
q25_p_D
q75_p_D
```

Create:

```text
outputs/fusion_pilot/summary_ratio_by_case_delta_omega.csv
```

Group by:

```text
case_name, delta, omega_R
```

Report:

```text
median_ratio
q25_ratio
q75_ratio
fusion_win_rate
median_C_R
median_C_D
median_C_f
success_rate_fusion
stop_fail_rate_fusion
median_p_D_fusion
```

---

## 9. Required Figures

Create figures under:

```text
outputs/fusion_pilot/figures/
```

### 9.1 Cost-ratio heatmaps

Create one heatmap per case:

```text
heatmap_ratio_unit.png
heatmap_ratio_special.png
heatmap_ratio_general.png
```

Rows:

```text
delta = 0.1, 0.05, 0.01
```

Columns:

```text
omega_R = 1, 2, 4, 8, 16
```

Cell value:

```text
median ratio = median(C_f / min(C_R, C_D))
```

Annotate each cell with the numeric value rounded to 2 decimals.

Use a visual threshold at ratio = 1. The plot should make it obvious where fusion wins.

### 9.2 Representative result table as Markdown

Create:

```text
outputs/fusion_pilot/representative_delta005_omega4.md
```

This table should use:

```text
delta = 0.05
omega_R = 4
omega_D = 1
```

Rows:

```text
case_name x regime
```

Columns:

```text
success_rate
stop_fail_rate
median_T_r
median_T_d
median_T_total
median_cost
median_p_D
```

### 9.3 Optional success sanity plot

Create:

```text
success_by_delta.png
```

For each case, show fusion success rate versus \(\delta\), with reference line \(1-\delta\).

This is optional but useful.

---

## 10. Acceptance Criteria

The pilot is considered successful if the following are true.

### 10.1 Basic correctness

For fusion:

```text
stop_fail_rate is close to 0
success_rate is close to 1 - delta
```

Some statistical fluctuation is acceptable, but severe underperformance must be reported.

### 10.2 Fusion usefulness

At least in the special case or general case, there should be some meaningful region where:

\[
\frac{C_f}{\min(C_R,C_D)}<1.
\]

Especially inspect:

```text
delta = 0.05
omega_R = 4 or 8
```

### 10.3 Fusion should not silently degenerate

Inspect \(p_D\). It is acceptable if fusion leans toward one feedback type, but report if:

```text
p_D < 0.05 or p_D > 0.95
```

for most fusion runs.

---

## 11. Implementation Plan

### Phase 0: Project setup

Create a module, for example:

```text
experiments/fusion_pilot/
  config.py
  instances.py
  feedback.py
  mle.py
  design.py
  rounding.py
  algorithm.py
  run_pilot.py
  analyze.py
  plot.py
  tests/
```

### Phase 1: Instance generators

Implement:

```python
make_unit_instance()
make_special_instance()
make_general_instance(seed)
```

Each should return:

```python
X: np.ndarray shape (K, d)
theta_star: np.ndarray shape (d,)
i_star: int
metadata: dict
```

### Phase 2: Feedback simulator

Implement:

```python
sample_reward(i, X, theta_star, rng)
sample_duel(i, j, X, theta_star, rng)
```

Use stable sigmoid.

### Phase 3: Core algorithm

Implement:

```python
run_phased_mixed_xy_bai(X, theta_star, delta, regime, seed, config) -> RunResult
```

This function should:

1. Build feasible queries.
2. Run burn-in.
3. Iterate phases.
4. Compute MLE.
5. Compute Fisher increments.
6. Solve design.
7. Round to phase sequence.
8. Execute phase.
9. Check stopping rule.
10. Stop at success or `T_max`.

### Phase 4: Experiment runner

Implement CLI:

```bash
python -m experiments.fusion_pilot.run_pilot --seeds 500 --out outputs/fusion_pilot
```

Also support smoke test:

```bash
python -m experiments.fusion_pilot.run_pilot --seeds 20 --out outputs/fusion_pilot_smoke
```

Parallelization is allowed. Use deterministic seeding so that reruns are reproducible.

### Phase 5: Analysis

Implement:

```bash
python -m experiments.fusion_pilot.analyze --input outputs/fusion_pilot/raw_results.csv
```

This should produce all CSV summary files and Markdown tables.

### Phase 6: Plotting

Implement:

```bash
python -m experiments.fusion_pilot.plot --summary outputs/fusion_pilot/summary_ratio_by_case_delta_omega.csv
```

This should produce the heatmaps and optional success plot.

---

## 12. Reproducibility Requirements

Use deterministic seeding.

Recommended seed scheme:

```python
base_seed = seed
instance_seed = hash((case_name, seed, "instance"))
run_seed = hash((case_name, regime, delta, seed, "run"))
```

The same `case_name` and `seed` should produce the same instance across regimes and deltas. The observation randomness may differ by regime and delta, but must be reproducible.

Save the full config as:

```text
outputs/fusion_pilot/config.json
```

---

## 13. Diagnostics to Print During Smoke Test

For the 20-seed smoke test, print:

```text
case, regime, delta, success_rate, stop_fail_rate, median_T_total, median_T_r, median_T_d
```

Also print warnings if:

```text
stop_fail_rate > 0.1
success_rate < 1 - delta - 0.15
median_T_total is near T_max
MLE frequently fails
Design optimizer frequently falls back
```

---

## 14. Important Notes

1. This pilot does not prove theory. It only checks whether the algorithm is worth further development.
2. Do not tune parameters separately for reward-only, duel-only, and fusion. The comparison should be fair.
3. Do not rerun the algorithm for each \(\omega_R\), unless a cost-aware design is explicitly implemented later.
4. Do rerun the algorithm for each \(\delta\), because \(\delta\) affects the phase schedule and stopping rule.
5. If fusion does not beat both baselines, report it honestly. In that case, inspect whether fusion is sample-efficient but not cost-efficient. That would motivate a later cost-aware version.
6. Exact efficient rounding is not required for this pilot. Deterministic largest-fraction rounding is acceptable, but document the approximation.
7. If exact design optimization is too slow, use the fallback and report how often fallback was used.
