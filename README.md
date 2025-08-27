# Reinforcement-Learning-Study

This respiratory is for learning algorithms in reinforcement learning (and related fields). Sections below would be included into the respiratory.

## MAB Problems
The multi-armed bandit problem suppose you have $K$ arms to pull in a bandit, and you are needed to optimize your choice. To be clearer, there are two branches to focus on.
### Regret Minimization
Regret means the difference between the rewards after pulling the arms and the optimal rewards. Studies have conducted an asymptotic $O(\log{n}/n)$ cumulative regret. This is outlined in *Finite Time Analysis on Multi-Armed Bandit* (Auer et al., 2002) and *Bandit based Monte-Carlo Planning* (2006). We contain related experiments like:

- 02 Finite Time Analysis on MAB
### Best Arm Identification

Best Arm Identification focuses more on figuring out the best arm after all the simulations. Hence, this series of problems indicates a gap compared to Regret Minimization, since Regret Minimization puts more effort on exploitation (pulling (near-)optimal arms).

One branch is by select $\varepsilon$-optimal arms, which is actually a set of optimal or near-optimal arms with little difference ($\varepsilon$ difference on mean).
