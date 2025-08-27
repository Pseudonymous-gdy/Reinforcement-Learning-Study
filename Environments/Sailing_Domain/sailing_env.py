'''
SailingEnv Introduction:

State: (x,y,wind_dir,tack), where:
- wind_dir: direction of wind
- tack: the relative direction of sailing compared
    to wind directions.

Actions (7): neighbor moves; action opposite to wind
    is forbidden. Cost is in [1,8.6], depends on
    action vs. wind direction.

API:
- reset(start=None, goal=None) -> state
- step(state, action) -> (next_state, cost, done,
    info)
- valid_cations(state) -> List[int]

Wind/Tack update: delegate to WindModel
'''
from __future__ import annotations

import cost_model
import wind_model
import numpy as np
from typing import List, Optional, Tuple, Any, Set, Dict
from abc import ABC, abstractmethod
from collections import defaultdict
from numpy import log, sqrt


class Node(ABC):
    '''
    A node in the future tree search.
    Basic implementation of a search tree node is like this:
    State: (x, y, wind_dir, tack)
    Reward: the reward obtained from reaching this node
    Sample Mean: the average reward obtained from paths including this node
    Visit Count: the number of times this node has been visited
    Action: the action taken to reach its children
    '''
    @abstractmethod
    def __init__(self, state: np.ndarray) -> None:
        self.state = state
        self.reward: float = 0.0
        self.visit_count: int = 0
        self.sample_mean: float = 0.0
        self.children: Dict[int, Node] = {}
        self.action: List[int] = []
        self.depth: Optional[int] = None

    # expected attributes on concrete Node implementations
    terminal: bool
    turn: int
    # common game node attributes used by callers
    state: Any
    winner: int
    space: int

    @abstractmethod
    def expand(self, action: int, next_state: np.ndarray) -> None:
        if action not in self.children:
            self.children[action] = Node(next_state)
        if action not in self.action:
            self.action.append(action)

    @abstractmethod
    def get_children(self) -> List["Node"]:
        '''Return a list of child nodes.'''
        return list(self.children.values())
    
    def find_children(self) -> Set["Node"]:
        '''Return a set of child nodes from this node.'''
        return set(self.children.values())

    @abstractmethod
    def is_terminal(self) -> bool:
        '''check if the node is terminated (leaf node).'''
        return len(self.children) == 0
    
    @abstractmethod
    def find_random_child(self) -> "Node":
        '''Return a randomly chosen child node (used for simulation).'''
        if not self.children:
            return None
        return self.children[np.random.choice(len(self.children))]

    @abstractmethod
    def get_reward(self) -> float:
        '''Return the reward of this node.'''
        return self.reward