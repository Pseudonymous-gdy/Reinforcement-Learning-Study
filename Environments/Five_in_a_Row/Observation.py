import numpy as np
import pandas as pd
from Environments.Five_in_a_Row import Judge
from typing import Tuple, List, Dict, Any, Optional


class Observation:
    """Clean observation implementation for Five-in-a-Row."""

    def __init__(self, N: int = 6, require_exact_five: bool = False):
        self.N = N
        self.require_exact_five = require_exact_five
        self.reset(N)

    def reset(self, N: int | None = None) -> np.ndarray:
        if N is not None:
            self.N = N
        self.board = np.zeros((self.N, self.N), dtype=int)
        self.steps = pd.DataFrame(columns=["player", "action"])
        self.done = False
        self.winner = 0
        self.winning_coords: Optional[List[Tuple[int, int]]] = None
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        return self.board.copy()

    def legal_actions(self) -> List[Tuple[int, int]]:
        inds = np.argwhere(self.board == 0)
        return [tuple(x) for x in inds]

    def record_step(self, player: int, action: Tuple[int, int]):
        self.steps = pd.concat([self.steps, pd.DataFrame([{"player": player, "action": action}])], ignore_index=True)

    def simulate(self) -> Tuple[bool, int]:
        if self.steps.empty:
            return self.done, self.winner
        last = self.steps.iloc[-1]
        action = last["action"]
        player = int(last["player"])
        if not (isinstance(action, (tuple, list)) and len(action) == 2):
            raise ValueError("action must be a (row,col) tuple")
        r, c = action
        if self.board[r, c] != 0:
            raise ValueError("cell occupied")
        self.board[r, c] = player
        done, winner, coords = Judge.judge(self.board, last_move=(r, c), require_exact_five=self.require_exact_five)
        self.done = done
        self.winner = winner
        self.winning_coords = coords
        return self.done, self.winner

    def step(self, player: int, action: Tuple[int, int]) -> Tuple[np.ndarray, int, bool, Dict[str, Any]]:
        self.record_step(player, action)
        done, winner = self.simulate()
        if done:
            if winner == 0:
                reward = 0
            elif winner == player:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0
        return self.get_observation(), reward, done, {"winner": winner, "winning_coords": self.winning_coords}

import numpy as np
import pandas as pd
from Environments.Five_in_a_Row import Judge
from typing import Tuple, List, Dict, Any


class Observation():
    """Observation / state container for Five-in-a-Row.

    Provides a small gym-like API: step, reset, get_observation, legal_actions.
    Action is expected to be a tuple (row, col).
    """

    def __init__(self, N: int = 6, require_exact_five: bool = False):
        self.steps = pd.DataFrame(columns=["player", "action"])
        self.N = N  # size of board
        self.board = np.zeros((N, N), dtype=int)  # 0: empty, 1: player 1, -1: player 2
        self.done = False
        self.winner = 0
        self.require_exact_five = require_exact_five

    def record_step(self, player: int, action: Tuple[int, int]):
        new_row = pd.DataFrame([{"player": player, "action": action}])
        self.steps = pd.concat([self.steps, new_row], ignore_index=True)

    def simulate(self) -> Tuple[bool, int]:
        """Apply the last recorded step to the board and run judge.

        Returns (done, winner).
        """
        if self.steps.empty:
            return self.done, self.winner

        last_step = self.steps.iloc[-1]
        action = last_step["action"]
        player = int(last_step["player"])

        # expect action to be a (r, c) tuple
        if not (isinstance(action, (tuple, list)) and len(action) == 2):
            raise ValueError("action must be a (row, col) tuple")

        r, c = action
        if self.board[r, c] != 0:
            raise ValueError(f"Cell {(r,c)} is already occupied")

        self.board[r, c] = player

        # call judge with last_move for efficient immediate judgement
        res = Judge.judge(self.board, last_move=(r, c), require_exact_five=self.require_exact_five)
        # judge now returns (done, winner, winning_coords)
        self.done, self.winner, self.winning_coords = res
        return self.done, self.winner

    def step(self, player: int, action: Tuple[int, int]) -> Tuple[np.ndarray, int, bool, Dict[str, Any]]:
        """Record an action for player, apply it, and return (obs, reward, done, info).

        Reward convention: +1 for immediate win for the player who moved, -1 for immediate loss,
        0 otherwise. Info contains 'winner'.
        """
        self.record_step(player, action)
        done, winner = self.simulate()

        if done:
            if winner == 0:
                reward = 0
            elif winner == player:
                reward = 1
            else:
                reward = -1
        else:
            reward = 0

        info = {"winner": winner, "winning_coords": getattr(self, 'winning_coords', None)}
        return self.get_observation(), reward, done, info

    def get_observation(self) -> np.ndarray:
        return self.board.copy()

    def legal_actions(self) -> List[Tuple[int, int]]:
        inds = np.argwhere(self.board == 0)
        return [tuple(x) for x in inds]

    def reset(self, N: int | None = None) -> np.ndarray:
        self.steps = pd.DataFrame(columns=["player", "action"])
        if N is not None:
            self.N = N
        self.board = np.zeros((self.N, self.N), dtype=int)
        self.done = False
        self.winner = 0
        return self.get_observation()