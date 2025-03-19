## UCB1 Algorithm

**Initialization:**  
- Play each machine once.

**Loop:**  
- Compute for each machine $j$ the Upper Confidence Bound:
  
  $$
  \text{UCB}_j = \bar{X}_j + \sqrt{\frac{2 \ln n}{n_j}}
  $$

- Select the machine $j$ that maximizes $\text{UCB}_j$.
- Play machine $j$.
- Update $n_j$ and $\bar{X}_j$.

## UCB2 Algorithm

**Parameters:**  
- $0 < \alpha < 1$

**Initialization:**  
- Set $r_j = 0$ for all machines.
- Play each machine once.

**Loop:**  
1. For each machine $j$, compute:
  
   $$ \text{Index} = \bar{X}_j + a_{n, r_j} $$
  
   where

   $$ a_{n, r} = \sqrt{\frac{(1+\alpha) \ln\left(\frac{e n}{\tau(r)}\right)}{2 \tau(r)}} $$
  
   and
  
   $$ \tau(r) = \lceil (1+\alpha)^r \rceil. $$

2. Select the machine $j$ with the highest index.
3. Play machine $j$ for
  
   $$ \tau(r_j+1) - \tau(r_j) $$
  
   times.
4. Increment $r_j$ by 1.

## $\varepsilon_n$-Greedy Algorithm

**Parameters:**  
- $c > 0$, $0 < d < 1$

**Define:**  
  
$$ \varepsilon_n = \min\{1, \frac{cK}{d^2 n}\} $$

**Loop:**  
1. Identify the machine $z$ with the highest current average reward.
2. With probability $1-\varepsilon_n$, play machine $z$; otherwise, randomly select a machine to play.

## UCB1-Normal Algorithm

**Initialization:**  
- Play each machine at least
  
  $$ \lceil 8 \ln n \rceil $$
  
  times.

**Loop:**  
- For each machine $j$, compute the index:
  
  $$ \text{Index} = \bar{X}_j + \sqrt{\frac{16\left(Q_j - n_j \bar{X}_j^2\right)}{n_j - 1} \cdot \frac{\ln(n-1)}{n_j}} $$

- Select the machine $j$ that maximizes the index.
- Play machine $j$.
- Update $\bar{X}_j$ and $Q_j$ (the cumulative squared reward).

