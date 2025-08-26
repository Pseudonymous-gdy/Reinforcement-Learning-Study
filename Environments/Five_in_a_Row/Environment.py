from typing import Tuple, Dict, Any
from Environments.Five_in_a_Row import Observation
from Environments.Five_in_a_Row.Agent import BaseAgent, FiveinaRow_Agent
import numpy as np


class FiveInARowEnv:
    """Environment to run a Five-in-a-Row match between two agents.

    Usage:
      env = FiveInARowEnv(N=6)
      env.reset()
      while not done:
          action = agent.act(agent.observe(env.observation.get_observation()), env.observation.legal_actions())
          obs, reward, done, info = env.step(agent.id, action)
    """

    def __init__(self, N: int = 10, require_exact_five: bool = True):
        # Environment enforces exact-five rule by default
        self.observation = Observation.Observation(N, require_exact_five=require_exact_five)
        self.N = N

    def reset(self) -> np.ndarray:
        return self.observation.reset(self.N)

    def step(self, player: int, action: Tuple[int, int]) -> Tuple[np.ndarray, int, bool, Dict[str, Any]]:
        return self.observation.step(player, action)

    def render(self) -> None:
        print(self.observation.get_observation())

    def legal_actions(self):
        return self.observation.legal_actions()
