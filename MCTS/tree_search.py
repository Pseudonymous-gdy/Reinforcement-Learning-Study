from __future__ import annotations

import numpy
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, Dict, Set, List, Any, Optional
from numpy import log, sqrt

class MCTS:
    """Monte-Carlo Tree Search controller with several selection policies.

    This class maintains per-node statistics and provides methods to run
    rollouts and select children according to UCT, OCBA or AOAP policies.

    Attributes (high-level):
        Q: total accumulated reward per node (float)
        N: visit count per node (int)
        children: cached child sets per node (Set[Node])
        ave_Q, std: sample mean and std of returns per node
        pv, pm: posterior variance and mean used by AOAP/OCBA
    """

    def __init__(
        self,
        exploration_weight: float = 1,
        policy: str = 'uct',
        budget: int = 1000,
        n0: int = 4,
        opp_policy: str = 'random',
        muu_0: float = 2,
        sigmaa_0: float = 2,
        sigma_0: float = 1,
        GAMMA: float = 0.9
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
        assert policy in {'uct', 'ocba', 'AOAP'}
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

    def choose(self, node: "Node") -> "Node":
        """Choose the best child of `node` according to the configured policy.

        Args:
            node: a non-terminal Node whose children have been populated.

        Returns:
            The selected child Node (deterministic based on stored stats).

        Raises:
            RuntimeError: if called on a terminal node.
        """
        if node.is_terminal():
            raise RuntimeError("choose called on terminal node {node}")

        # If the node has no cached children, fall back to a random child.
        if node not in self.children:
            return node.find_random_child()

        def score(n: "Node") -> float:
            # Unvisited nodes are deprioritized for this deterministic choose.
            if self.N[n] == 0:
                return float("-inf")
            if self.policy == 'uct':
                return self.ave_Q[n]
            elif self.policy == 'AOAP':
                return self.pm[n]
            else:
                return self.ave_Q[n]

        # return the child with maximum score
        return max(self.children[node], key=score)

    def do_rollout(self, node: "Node") -> None:
        """Perform one MCTS rollout starting from `node`.

        The rollout runs selection, expansion (if applicable), simulation
        and backpropagation, updating the internal statistics.
        """
        path = self._select(node)
        leaf = path[-1]

        # record that this leaf has been simulated
        self.leaf_cnt[leaf] += 1
        sim_reward = self._simulate(leaf)
        self._backpropagate(path, sim_reward)

    def _select(self, node: "Node") -> List["Node"]:
        """Selection + expansion phase: traverse from `node` to a leaf.

        Returns:
            path: list of nodes from root (`node`) to the selected leaf (inclusive).
        """
        path: List["Node"] = []
        while True:
            path.append(node)
            # increment visit count for selection statistics
            self.N[node] += 1
            if node.terminal:
                return path

            if len(self.children[node]) == 0:
                # populate children cache (Node implementations decide legality)
                children_found = node.find_children()
                self.children[node] = children_found

            # opponent move handling (turn == -1 indicates opponent)
            if node.turn == -1:
                if self.opp_policy == 'random':
                    node = node.find_random_child()
                if self.opp_policy == 'uct':
                    expandable = [n for n in self.children[node] if self.N[n] < 1]
                    if expandable:
                        node = expandable.pop()
                    else:
                        log_N_vertex = log(sum([self.N[c] for c in self.children[node]]))
                        node = min(
                            self.children[node],
                            key=lambda n: self.ave_Q[n]
                            - self.exploration_weight * sqrt(2 * log_N_vertex / self.N[n]),
                        )
                continue

            # normal agent's turn: choose a child to expand or select by policy
            expandable: List["Node"] = [n for n in self.children[node] if self.N[n] < self.n0]
            if expandable:
                a = self._expand(node)
                if len(self.children[a]) == 0:
                    self.children[a] = a.find_children()
                path.append(a)
                self.N[a] += 1
                return path
            else:
                if self.policy == 'uct':
                    a = self._uct_select(node)
                elif self.policy == 'AOAP':
                    a = self._AOAP_select(node)
                else:
                    a = self._ocba_select(node)
                node = a

    def _expand(self, node: "Node", path_reward: Optional[float] = None) -> "Node":
        explored_once: List["Node"] = [n for n in self.children[node] if self.N[n] < self.n0]
        return explored_once.pop()

    def _simulate(self, node):
        """Simulate a random playout from `node` until a terminal state.

        Returns:
            The simulation reward (float) obtained at the terminal node.
        """
        while True:
            if not node.is_terminal():
                node = node.find_random_child()
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
        for i in range(len(path) - 1, -1, -1):
            node = path[i]
            # accumulate total reward and record the sample
            self.Q[node] += r
            self.all_Q[node].append(r)

            old_ave_Q: float = self.ave_Q[node]
            # average reward for the node (Q / N)
            self.ave_Q[node] = self.Q[node] / self.N[node]
            # online update of sample std (Welford-like approximation)
            self.std[node] = sqrt(
                ((self.N[node] - 1) * self.std[node] ** 2 + (r - old_ave_Q) * (r - self.ave_Q[node])) / self.N[node]
            )

            if self.std[node] == 0:
                self.std[node] = self.sigma_0

            # Bayesian posterior variance/mean used by AOAP selection
            self.pv[node] = 1 / (1 / self.sigmaa_0 + self.N[node] / (self.std[node]) ** 2)
            self.pm[node] = self.pv[node] * (
                self.muu_0 / self.sigmaa_0 + self.N[node] * self.ave_Q[node] / (self.std[node]) ** 2
            )
            r *= self.GAMMA  # discounting for non-terminal rewards
            r += node.reward()

    def _uct_select(self, node: "Node") -> "Node":
        """Select a child using the UCT formula.

        Returns:
            The child Node maximizing UCT = ave_Q + c * sqrt(2 * ln(N_parent) / N_child)
        """
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = log(sum([self.N[c] for c in self.children[node]]))

        def uct(n: "Node") -> float:
            return self.ave_Q[n] + self.exploration_weight * sqrt(2 * log_N_vertex / self.N[n])

        return max(self.children[node], key=uct)
    
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