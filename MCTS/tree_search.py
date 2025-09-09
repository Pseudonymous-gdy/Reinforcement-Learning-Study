"""Monte-Carlo Tree Search utilities.

This module implements a lightweight MCTS controller (`MCTS`) and an
abstract `Node` interface that concrete game states should implement.

The comments added here aim to explain the main data structures and
selection policies (UCT, OCBA, AOAP) used by the controller.
"""

from __future__ import annotations

import numpy
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, Dict, Set, List, Any, Optional
from numpy import log, sqrt


class MCTS:
    """Monte-Carlo Tree Search controller with several selection policies.

    The MCTS object stores statistics per `Node` and exposes methods to
    run a single rollout (`do_rollout`) and to pick a deterministic best
    child (`choose`) using the configured selection policy.

    Key per-node statistics kept in defaultdicts keyed by Node objects:
        - Q: cumulative sum of rewards observed for the node
        - N: number of times the node was visited
        - all_Q: raw list of sampled returns (for variance estimates)
        - ave_Q: sample mean of returns (Q / N)
        - std: sample standard deviation estimate of returns
        - pv, pm: posterior variance and mean used by Bayesian AOAP/OCBA

    Supported policies: 'uct', 'ocba', 'AOAP', 'selfpolicy'. The code
    uses small helper methods for selection logic for each policy.
    """

    def __init__(
        self,
        exploration_weight: float = 2,
        policy: str = 'uct',
        budget: int = 1000,
        n0: int = 4,
        opp_policy: str = 'random',
        muu_0: float = 2,
        sigmaa_0: float = 2,
        sigma_0: float = 1,
        GAMMA: float = 0.9,
        epsilon: float = 0.1,
        delta: float = 0.3,
        gamma: float = 0.05
    ) -> None:
        # Node -> aggregated total reward
        self.Q: DefaultDict["Node", float] = defaultdict(float)
        # placeholders (unused in current file but kept for compatibility)
        self.V_bar: DefaultDict["Node", float] = defaultdict(float)
        self.V_hat: DefaultDict["Node", float] = defaultdict(float)
        # Node -> visit count
        self.N: DefaultDict["Node", int] = defaultdict(int)
        # Node -> set of child Nodes
        self.children: DefaultDict["Node", Set["Node"]] = defaultdict(set)
        self.exploration_weight: float = exploration_weight
        # Node -> list of raw rollout returns
        self.all_Q: DefaultDict["Node", List[float]] = defaultdict(list)
        assert policy in {'uct', 'ocba', 'AOAP', 'selfpolicy'}
        self.policy: str = policy
        # Node -> sample std dev of returns
        self.std: DefaultDict["Node", float] = defaultdict(float)
        # Node -> average reward (Q / N)
        self.ave_Q: DefaultDict["Node", float] = defaultdict(float)
        # Posterior variance and mean used by AOAP/ocba
        self.pv: DefaultDict["Node", float] = defaultdict(float)
        self.pm: DefaultDict["Node", float] = defaultdict(float)
        self.budget: int = budget
        self.n0: int = n0
        # counts how many times a node has been selected as a leaf for simulation
        self.leaf_cnt: DefaultDict["Node", int] = defaultdict(int)
        self.opp_policy: str = opp_policy
        # Hyperparameters for Bayesian updates
        self.muu_0: float = muu_0
        self.sigmaa_0: float = sigmaa_0
        self.sigma_0: float = sigma_0
        self.GAMMA: float = GAMMA
        # Hyperparameters for self-defined policy
        self.epsilon: float = epsilon
        self.delta: float = delta
        self.gamma: float = gamma
        self.root_select_round: int = 0

    def choose(self, node: "Node") -> "Node":
        """Choose the best child of `node` according to the configured policy.

        Args:
            node: a non-terminal Node whose children have been populated.

        Returns:
            The selected child Node (deterministic based on stored stats).

        Raises:
            RuntimeError: if called on a terminal node.
        """
        # Choosing from a terminal node is a logic error in callers.
        if node.is_terminal():
            raise RuntimeError("choose called on terminal node {node}")

        # If we haven't cached the children set for this node yet,
        # fall back to a single random child (non-deterministic).
        if node not in self.children:
            # When returning a random child, try to ensure the child's
            # `depth` attribute is set so downstream algorithms that
            # rely on node depth will behave consistently. We do this
            # lazily to avoid forcing callers to change all concrete
            # Node implementations immediately.
            child = node.find_random_child()
            self._ensure_depth(child, node)
            return child

        # Deterministic scoring used by `choose` (no exploration term):
        # - Unvisited nodes get -inf so they are not chosen deterministically
        #   (they should be explored during normal rollouts instead).
        def score(n: "Node") -> float:
            if self.N[n] == 0:
                return float("-inf")
            if self.policy == 'uct':
                # For UCT choose the child with highest empirical mean.
                return self.ave_Q[n]
            elif self.policy == 'AOAP':
                # For AOAP choose by the posterior mean estimate.
                return self.pm[n]
            elif self.policy == 'selfpolicy':
                # For self-defined policy choose by lower confidence bound (LCB).
                # Compute an adaptive confidence radius C matching the
                # C_{delta/n} used in `_self_policy_add` and return ave - C.
                # Use parent's visit count as t and number of children as n.
                try:
                    children_count = len(self.children[node]) if node in self.children else 1
                except Exception:
                    children_count = 1
                t = max(1, self.N[node])
                denom = max(self.delta / max(children_count, 1), 1e-12)
                C = 0.0
                try:
                    inner = numpy.log2(2 * t) / denom
                    if inner > 1.0:
                        C = self.exploration_weight * sqrt(log(inner) / t)
                except Exception:
                    C = 0.0
                return self.ave_Q[n] - C
            else:
                # OCBA and fallback also use the empirical mean here.
                return self.ave_Q[n]

        # return the child with maximum score
        return max(self.children[node], key=score)

    def do_rollout(self, node: "Node") -> None:
        """Perform one MCTS rollout starting from `node`.

        The rollout runs selection, expansion (if applicable), simulation
        and backpropagation, updating the internal statistics.
        """
        # 1) Selection / expansion: pick a leaf to simulate from
        path = self._select(node)
        leaf = path[-1]

        # 2) Simulation: run a (stochastic) playout from the leaf
        self.leaf_cnt[leaf] += 1
        sim_reward = self._simulate(leaf)

        # 3) Backpropagation: propagate the sampled return up the path
        self._backpropagate(path, sim_reward)

    def _select(self, node: "Node") -> List["Node"]:
        """Selection + expansion phase: traverse from `node` to a leaf.

        Returns:
            path: list of nodes from root (`node`) to the selected leaf (inclusive).
        """
        # Traverse down the tree until we hit a leaf to simulate from.
        # We return the full path from the provided root node down to the
        # selected leaf (inclusive).
        path: List["Node"] = []
        while True:
            path.append(node)
            # increment the visit count immediately; this count is used by
            # selection formulas (UCT/OCBA/AOAP) and by expansion checks.
            self.N[node] += 1
            if node.terminal:
                return path

            # Lazily populate the cached children set. Concrete `Node`
            # implementations provide `find_children` which enumerates legal
            # successor states from this state.
            if len(self.children[node]) == 0:
                children_found = node.find_children()
                # Ensure every returned child has a valid `depth` attribute.
                # If concrete Node implementations already set depth, this
                # does nothing; otherwise we fill it using the parent's
                # depth (default 0 when missing) to keep consistency.
                for c in children_found:
                    self._ensure_depth(c, node)
                self.children[node] = children_found

            # opponent move handling (turn == -1 indicates opponent)
            # If the current node represents an opponent move (turn == -1),
            # we handle opponent policy separately. Opponent can be random or
            # use a simple UCT-like selection. This keeps opponent behaviour
            # configurable.
            if node.turn == -1:
                if self.opp_policy == 'random':
                    node = node.find_random_child()
                    # keep depth consistent for opponent-sampled moves
                    self._ensure_depth(node, path[-1] if len(path)>0 else None)
                if self.opp_policy == 'uct':
                    # Prefer to expand unvisited opponent children first
                    expandable = [n for n in self.children[node] if self.N[n] < 1]
                    if expandable:
                        node = expandable.pop()
                    else:
                        # Otherwise choose the child minimizing the UCT-style
                        # value (we use ave_Q - exploration to model an
                        # adversarial/opponent choice).
                        log_N_vertex = log(sum([self.N[c] for c in self.children[node]]))
                        node = min(
                            self.children[node],
                            key=lambda n: self.ave_Q[n]
                            - self.exploration_weight * sqrt(2 * log_N_vertex / self.N[n]),
                        )
                continue

            # Agent's turn: if there are children that haven't been sufficiently
            # explored (N < n0) expand one of them; otherwise use the configured
            # selection policy to pick the next child to descend to.
            expandable: List["Node"] = [n for n in self.children[node] if self.N[n] < self.n0]
            if expandable:
                a = self._expand(node)
                # Ensure newly expanded node has its children cached for the
                # next steps of the algorithm.
                if len(self.children[a]) == 0:
                    children_found = a.find_children()
                    for c in children_found:
                        self._ensure_depth(c, a)
                    self.children[a] = children_found
                path.append(a)
                self.N[a] += 1
                return path
            else:
                # Use the configured selection policy to pick a child to descend.
                if self.policy == 'uct':
                    a = self._uct_select(node)
                elif self.policy == 'AOAP':
                    a = self._AOAP_select(node)
                elif self.policy == 'ocba':
                    a = self._ocba_select(node)
                else:
                    a = self._self_policy_add(node)
                node = a

    def _expand(self, node: "Node", path_reward: Optional[float] = None) -> "Node":
        # Return one child that has been visited fewer than `n0` times.
        explored_once: List["Node"] = [n for n in self.children[node] if self.N[n] < self.n0]
        child = explored_once.pop()
        # make sure depth is set for the expanded child
        self._ensure_depth(child, node)
        return child

    def _simulate(self, node):
        """Simulate a random playout from `node` until a terminal state.

        Returns:
            The simulation reward (float) obtained at the terminal node.
        """
        # Do a random playout until a terminal state is reached. Concrete
        # Node implementations should make `find_random_child` and `reward`
        # behave consistently for their game domain.
        while True:
            if not node.is_terminal():
                parent = node
                node = node.find_random_child()
                # ensure depth propagated during simulation as well
                self._ensure_depth(node, parent)
            if node.terminal:
                return node.reward()

    def _backpropagate(self, path, r):
        """Backpropagate simulation result `r` along `path`.

        Updates running totals, sample statistics and Bayesian posterior
        parameters used by selection policies.

        Args:
            path: list of nodes from root to simulated leaf (inclusive).
            r: scalar reward returned by the simulation.
        """
        # Walk back from the leaf to the root and update statistics.
        # Note: rewards may be discounted (GAMMA) when propagating up so the
        # effective contribution of deeper rewards is reduced for ancestors.
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            # accumulate total reward and record the sample for variance calc
            self.Q[node] += r
            self.all_Q[node].append(r)

            old_ave_Q: float = self.ave_Q[node]
            # update sample mean (total / count)
            self.ave_Q[node] = self.Q[node] / self.N[node]
            # approximate online update of sample std (Welford-like)
            self.std[node] = sqrt(
                ((self.N[node] - 1) * self.std[node] ** 2 + (r - old_ave_Q) * (r - self.ave_Q[node])) / self.N[node]
            )

            # avoid zero std which would break Bayesian updates
            if self.std[node] == 0:
                self.std[node] = self.sigma_0

            # Bayesian posterior variance/mean used by AOAP selection. These
            # formulas assume a Gaussian likelihood with known variance
            # approximated by `self.std[node]` and a conjugate Gaussian prior.
            self.pv[node] = 1 / (1 / self.sigmaa_0 + self.N[node] / (self.std[node]) ** 2)
            self.pm[node] = self.pv[node] * (
                self.muu_0 / self.sigmaa_0 + self.N[node] * self.ave_Q[node] / (self.std[node]) ** 2
            )

            # Discount and add the immediate reward of the node itself when
            # moving up the tree. This lets deeper rewards have reduced
            # influence on ancestors (typical in discounted MDPs).
            r *= self.GAMMA
            r += node.reward()

    def _uct_select(self, node: "Node") -> "Node":
        """Select a child using the UCT formula.

        Returns:
            The child Node maximizing UCT = ave_Q + c * sqrt(2 * ln(N_parent) / N_child)
        """
        assert all(n in self.children for n in self.children[node])

        # Precompute log of parent visit count used in UCT formula.
        log_N_vertex = log(sum([self.N[c] for c in self.children[node]]))

        def uct(n: "Node") -> float:
            # Classic UCT: empirical mean + exploration bonus.
            return self.ave_Q[n] + self.exploration_weight * sqrt(2 * log_N_vertex / self.N[n])

        return max(self.children[node], key=uct)

    def _self_policy_add(self, node: "Node") -> "Node":
        """Select a child using the self-defined policy.
        
        Returns:
            The child Node maximizing the self-defined policy.
        
        Policy description:
        - If we're searching not from the root node (depth 0), conduct _uct_select.
        - If we're at the root node (depth 0), conduct the following strategy:
            1. Set Empirically Good Arms: G = {i: ave_Q[i] >= max(ave_Q) - epsilon}
            2. Calculate U_t = max(ave_Q) + C_{delta/n} - epsilon - gamma
                and L_t = max(ave_Q) - C_{delta/n} - epsilon
            3. Set Known Children K as the set of children with Upper Confidence under 
                L_t or Lower Confidence above U_t
            4. Pull the arm in G \ K with the lowest LCB (Lower Confidence Bound)
            5. Pull the arm not in G∪K with the highest UCB (Upper Confidence Bound)
            6. Pull the global empirically best arm with highest UCB

        Note:
        - C_{delta} = exploration_weight * sqrt(log(log_2(2t)/delta)/t), which is the confidence radius in this policy to calculate UCB or LCB
        - t here is the total number of selected times of the node to choose from the root
        - gamma is different from GAMMA, it's a small positive constant to avoid over-exploitation
        """
        # Ensure children are populated and depths assigned (backwards compatible)
        if len(self.children[node]) == 0:
            children_found = node.find_children()
            for c in children_found:
                self._ensure_depth(c, node)
            self.children[node] = children_found

        # If node isn't the root (depth != 0) fall back to classic UCT selection
        node_depth = getattr(node, "depth", 0)
        if node_depth != 0:
            return self._uct_select(node)

        # Now node is root (depth == 0). Implement the self-defined arm selection.
        children = list(self.children[node])
        n = max(1, len(children))

        # Safely compute t: use visit count of the node (at least 1 to avoid div0)
        t = max(1, self.N[node])

        # Empirical means for children (default 0 for unvisited)
        ave = {c: self.ave_Q[c] for c in children}
        max_ave = max(ave.values())

        # 1. Empirically good arms G
        G = {c for c, v in ave.items() if v >= max_ave - self.epsilon}

        # 2. Confidence radius C_{delta/n}
        # C = exploration_weight * sqrt( log( log2(2t)/(delta/n) ) / t )
        C = 0.0
        try:
            denom = max(self.delta / n, 1e-12)
            inner = numpy.log2(2 * t) / denom
            if inner > 1.0:
                C = self.exploration_weight * sqrt(log(inner) / t)
            else:
                C = 0.0
        except Exception:
            C = 0.0

        U_t = max_ave + C - self.epsilon - self.gamma
        L_t = max_ave - C - self.epsilon

        # compute UCB and LCB for each child
        UCB = {c: ave[c] + C for c in children}
        LCB = {c: ave[c] - C for c in children}

        # 3. Known children K: UCB < L_t OR LCB > U_t
        K = {c for c in children if (UCB[c] < L_t) or (LCB[c] > U_t)}

        # 4. Prefer arm in G \ K with lowest LCB
        if self.root_select_round == 0:
            self.root_select_round += 1
            cand = list(G - K)
            if cand:
                # choose min LCB; tie-break by highest N (more explored) then arbitrary
                return min(cand, key=lambda x: (LCB[x], -self.N[x]))

        # 5. Pull arm not in G ∪ K with highest UCB
        if self.root_select_round == 1:
            self.root_select_round += 1
            cand = [c for c in children if (c not in G) and (c not in K)]
            if cand:
                return max(cand, key=lambda x: (UCB[x], self.N[x]))

        # 6. Otherwise pull the global empirically best arm with highest UCB
        self.root_select_round = 0
        return max(children, key=lambda x: (UCB[x], self.N[x]))

    def _ensure_depth(self, child: Optional["Node"], parent: Optional["Node"]) -> None:
        """Ensure `child.depth` exists and is an int.

        Behavior:
        - If `child` is None do nothing.
        - If `child` already has a numeric `depth`, leave it untouched.
        - Otherwise set `child.depth = (parent.depth if available else 0) + 1`.

        This keeps the MCTS implementation backwards compatible: concrete
        Node classes don't need to be changed immediately, and MCTS will
        populate depth lazily when it discovers children or samples random
        moves. Downstream algorithms that rely on `Node.depth` can therefore
        assume the field is present for nodes encountered by MCTS.
        """
        if child is None:
            return
        try:
            # if depth already exists and is an int-like value, don't overwrite
            existing = getattr(child, "depth", None)
            if isinstance(existing, int):
                return
        except Exception:
            # be conservative: if attribute access fails, attempt to set it
            existing = None

        parent_depth = 0
        if parent is not None:
            parent_depth = getattr(parent, "depth", 0) or 0

        try:
            setattr(child, "depth", parent_depth + 1)
        except Exception:
            # If child doesn't allow attribute setting (unlikely for game
            # node objects), we silently ignore to preserve backward
            # compatibility; callers should prefer Node implementations
            # that expose writable attributes.
            pass
    
    def _ocba_select(self, node: "Node") -> "Node":
        assert all(n in self.children for n in self.children[node])
        assert len(self.children[node])>0
        
        if len(self.children[node]) == 1:
            return list(self.children[node])[0]
        
        all_actions = self.children[node] 
        b = max(all_actions, key=lambda n: self.ave_Q[n]) 
        best_Q = self.ave_Q[b] 
        suboptimals_set, best_actions_set, select_actions_set = set(), set(), set() 
        for k in all_actions:
            if self.ave_Q[k] == best_Q:
                best_actions_set.add(k) 
            else:
                suboptimals_set.add(k)
        
        if len(suboptimals_set) == 0:
            return min(self.children[node], key=lambda n: self.N[n]) 
        
        if len(best_actions_set) != 1:
            b = max(best_actions_set, key=lambda n : (self.std[node])**2 / self.N[n]) 
            
        for k in all_actions:
            if self.ave_Q[k] != best_Q:
                select_actions_set.add(k)
        select_actions_set.add(b)
        
        delta = defaultdict(float) 
        for k in select_actions_set:
            delta[k] = self.ave_Q[k] - best_Q 
        
        ref = next(iter(suboptimals_set)) 

        para = defaultdict(float)
        ref_std_delta = self.std[ref]/delta[ref] 
        para_sum = 0
        for k in suboptimals_set:
            para[k] = ((self.std[k]/delta[k])/(ref_std_delta))**2 
               
        para[b] = sqrt(sum((self.std[b]*para[c]/self.std[c])**2 for c in suboptimals_set))

        para_sum = sum(para.values()) 
        para[ref] = 1
       
        totalBudget = sum([self.N[c] for c in select_actions_set])+1
        ref_sol = (totalBudget)/para_sum
        
        return max(select_actions_set, key=lambda n:para[n]*ref_sol - self.N[n])

    def _AOAP_select(self, node: "Node") -> "Node":
        
        assert all(n in self.children for n in self.children[node])
        assert len(self.children[node]) > 0
        
        if len(self.children[node]) == 1:
            return list(self.children[node])[0] 
        
        all_actions = self.children[node]  
        b = max(all_actions, key=lambda n: self.pm[n])  
        best_Q = self.pm[b]  
        suboptimals_set, best_actions_set, select_actions_set = set(), set(), set()  
        
        for k in all_actions:
            if self.pm[k] == best_Q:
                best_actions_set.add(k)           
            else:
                suboptimals_set.add(k)
                
        if len(suboptimals_set) == 0:
            return min(self.children[node], key=lambda n: self.N[n]) 
        
        if len(best_actions_set) != 1:
            b = max(best_actions_set, key=lambda n : self.pv[n] / self.N[n]) 
            
        for k in all_actions:
            if self.pm[k] != best_Q:
                select_actions_set.add(k)
        select_actions_set.add(b)
            
        M = defaultdict(int)
        V = defaultdict(float)
        W = defaultdict(float)
        nv = defaultdict(float)
        
        
        for k in all_actions:
            nv[k] = self.pv[k]
            M[k] = self.N[k]
        
        for k in select_actions_set:
            M[k] += 1
            nv[k] = 1 / (1/self.sigmaa_0 + M[k]/(self.std[k])**2)  
            for i in suboptimals_set:
                W[i] = (best_Q - self.pm[i])**2 / (nv[b] + nv[i])
            V[k] = min(W.values()) 
            M[k] -= 1
            nv[k] = self.pv[k]
                  
        return max(select_actions_set, key=lambda n:V[n])
    
class Node(ABC):

    # depth is filled lazily by MCTS when missing; concrete implementations
    # can provide their own depth if desired.
    depth: int


    @abstractmethod
    def find_children(self) -> Set["Node"]:
        """Return a set of child nodes from this node."""
        return set()
    # expected attributes on concrete Node implementations
    terminal: bool
    turn: int
    # common game node attributes used by callers
    state: Any
    winner: int
    space: int

    @abstractmethod
    def find_random_child(self) -> "Node":
        """Return a randomly chosen child node (used for simulation)."""
        return None  # type: ignore[return-value]

    @abstractmethod
    def is_terminal(self) -> bool:
        return True

    @abstractmethod
    def reward(self) -> float:
        return -1e-2