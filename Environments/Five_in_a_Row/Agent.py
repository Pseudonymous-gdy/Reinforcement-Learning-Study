"""Agent interfaces for Five-in-a-Row.

Provide a BaseAgent suitable for multimodal inputs and an example random agent.
"""
from typing import Optional, Tuple, List, Dict, Any, Protocol
import numpy as np
import random
from Environments.Five_in_a_Row import Observation


class BaseAgent(Protocol):
    """Minimal agent interface.

    - id: 1 or 2
    - observe(state) -> processedobs (allow multimodal preprocessing)
    - act(processedobs, legal_actions) -> action
    - update(...) optional learning hook
    """

    id: int

    def observe(self, state: np.ndarray) -> Any:
        ...

    def act(self, processed_state: Any, legal_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        ...

    def update(self, *args, **kwargs) -> None:
        ...


class FiveinaRow_Agent:
    """Simple random agent that conforms to BaseAgent-style interface.

    This agent is a good baseline and example for plugging into MCTS variants later.
    """

    def __init__(self, id: int, name: Optional[str] = None):
        if id not in (1, 2):
            raise ValueError("Agent id must be 1 or 2")
        self.id = id
        self.name = name or f"RandomAgent-{id}"

    def observe(self, state: np.ndarray) -> np.ndarray:
        # identity observation; multimodal agents can override this to produce features
        return state.copy()

    def act(self, processed_state: np.ndarray, legal_actions: List[Tuple[int, int]]) -> Tuple[int, int]:
        if not legal_actions:
            raise ValueError("No legal actions available")
        return random.choice(legal_actions)

    def update(self, *args, **kwargs) -> None:
        # placeholder for learning or tree updates (MCTS backprop etc.)
        return None