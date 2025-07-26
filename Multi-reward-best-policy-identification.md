$generated\ by\ ChatGPT$.
### 1  Background (in short)

Designing *one* good reward is already tricky in reinforcement learning (RL); many practical tasks demand **testing or simultaneously satisfying several rewards** (e.g., coverage vs. throughput in radio networks, or different goal positions in robotics). The paper therefore studies how to explore once and still learn policies that are optimal for **every reward in a given set**.&#x20;

---

### 2  Problem setting (mathematical)

* **MDP:** $M=(\mathcal S,\mathcal A,P,\gamma)$ with finite state/action spaces and discount $\gamma\in(0,1)$.
* **Reward set:** a *known, finite* collection $R=\{r_i:\mathcal S\times\mathcal A\to[0,1]\}$.&#x20;
* **Objective (δ‑PC):** Design an algorithm that stops in finite time $\tau$ and outputs a policy $\hat\pi_{\,\tau,r}$ such that

  \[
     \Pr\bigl[\forall r\in R,\; \hat\pi_{\,\tau,r}\text{ is optimal for }r\bigr]\;\ge 1-\delta .
  \]

  The goal is to **minimise $\mathbb E[\tau]$**.&#x20;

---

### 3  Main contributions: algorithms

| Level            | Algorithm                                         | Idea                                                                                                                                                                                                                  |
| ---------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tabular MDPs** | **MR‑NaS** (Multi‑Reward *Navigate‑and‑Stop*)     | Track the stationary distribution $\omega^\star$ that minimises a convex “relaxed” exploration cost $U(\omega;M)$; mix it with forced exploration and stop when a likelihood‑ratio test is confident.                 |
| **Deep RL**      | **DBMR‑BPI** (Deep Bootstrapped Multi‑Reward BPI) | Extends MR‑NaS using an ensemble of Q‑networks to approximate uncertainty and a bootstrap trick to sample exploratory policies; picks which reward to explore with probability $\propto 1/\widehat{\Delta}_r^{\,2}$.  |

---

### 4  Key theoretical results

* **Instance‑specific lower bound (Theorem 3.1).** Any δ‑PC algorithm needs

  \[
     \displaystyle\mathbb E[\tau]\;\ge\;\frac{\log(1/\delta)}{T^\star(M)} ,
  \]

  where $T^\star(M)$ is the solution of a zero‑sum game between the learner’s visitation frequencies $\omega$ and an adversarial “confusing” MDP $M'$.&#x20;
* **Asymptotic optimality of MR‑NaS.** With an appropriate stopping rule, MR‑NaS satisfies

  \[
     \displaystyle\limsup_{\delta\to0}\frac{\tau}{\log(1/\delta)}\;\le\;U^\star(M)\quad\text{a.s.},
  \]

  matching the relaxed characteristic time $U^\star(M)$.&#x20;

---

### 5  Conclusions, limitations & future work

The study **generalises best‑policy identification from one reward to many**, supplies algorithms that **empirically beat reward‑free and unsupervised baselines** on hard‑exploration benchmarks, and shows promising transfer to *unseen* rewards.&#x20;

*Key limitations* include: theory restricted to tabular MDPs, per‑step recomputation of $\omega^\star_t$ can be expensive, and deep‑RL variant currently assumes finite action spaces and careful reward scaling. &#x20;

*Future directions* suggested by the authors: extending theory to continuous domains, scalable approximations for large reward sets, and handling multiple optimal policies or ε‑optimality gaps.&#x20;
