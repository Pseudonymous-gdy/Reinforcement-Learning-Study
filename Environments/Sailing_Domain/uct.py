from __future__ import annotations

import numpy
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, Dict, Set, List, Any, Optional
from numpy import log, sqrt
from Environments.Sailing_Domain.sailing_env import Node


class MCTS:
    '''
    Monte-Carlo Tree Search controller with several selection policies.

    This class maintains per-node statistics and provides methods to run
    rollouts and select children according to UCT, LUCB, and our algorithms.

    Attributes (high-level):
        Q: total accumulated reward per node (float)
        N: visit count per node (int)
        children: cached child sets per node (Set[Node])
        discarded: bool value to see whether it is considered suboptimal
    '''
    def __init__(self,
                 exploration_weight: float = 1,
                 policy: str = "uct",
                 budget: int = 1000,):
        self.Q: DefaultDict[Node, float] = defaultdict(float)
        self.N: DefaultDict[Node, int] = defaultdict(int)
        self.children: DefaultDict[Node, Set[Node]] = defaultdict(set)
        self.discarded: bool = False

    def choose(self, node: Node) -> Optional[Node]:
        '''
        Choose the best child of `node` according to the configured policy.

        Args:
            node: a non-terminal Node whose children have been populated.

        Returns:
            The selected child Node (deterministic based on stored stats).

        Raises:
            RuntimeError: if called on a terminal node.
        '''
        if node.is_terminal():
            raise RuntimeError("choose called on terminal node {node}")
        
        # no cached children, fall back to a random child
        if node not in self.children or len(self.children[node]) == 0:
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
