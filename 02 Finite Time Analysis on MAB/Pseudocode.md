## UCB1 Algorithm

Initialization: Play each machine once.
Loop:
    Select the machine j that maximizes the Upper Confidence Bound (UCB):
        UCB_j = X̄_j + √(2 ln n / n_j)
    Play machine j.
    Update n_j and X̄_j.

## UCB2 Algorithm

Parameters: 0 < α < 1
Initialization: r_j = 0, play each machine once.
Loop:
    1. Select the machine j that maximizes the index:
        Index = X̄_j + a_{n, r_j}, where a_{n, r} = √[(1+α) ln(en/τ(r)) / (2τ(r))]
        τ(r) = ⌈(1+α)^r⌉
    2. Play machine j τ(r_j + 1) - τ(r_j) times.
    3. r_j += 1.

## $\epsilon_n$-Greedy Algorithm

Parameters: c > 0, 0 < d < 1
Define ε_n = min{1, cK/(d²n)}
Loop:
    1. Select the machine z with the highest current average reward.
    2. Play z with probability 1 - ε_n; otherwise, randomly select a machine.

## UCB1-Normal Algorithm

Initialization: Play each machine at least ⌈8 log n⌉ times.
Loop:
    Select the machine j that maximizes the index:
        Index = X̄_j + √[16 * (Q_j - n_j X̄_j²)/(n_j - 1) * (ln(n - 1)/n_j)]
    Play machine j.
    Update X̄_j and Q_j (cumulative squared reward).
