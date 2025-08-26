import sys
if __name__ == '__main__':
    # when running as script, ensure repo root is on path
    sys.path.append('.')

from Environments.Five_in_a_Row.Environment import FiveInARowEnv
from Environments.Five_in_a_Row.Agent import FiveinaRow_Agent
from Environments.Five_in_a_Row.Render import render_board


def run_smoke(N=10, seed=None):
    import random
    if seed is not None:
        random.seed(seed)
    env = FiveInARowEnv(N=N)
    env.reset()

    agent1 = FiveinaRow_Agent(1)
    agent2 = FiveinaRow_Agent(2)
    agents = {1: agent1, 2: agent2}

    current = 1
    steps = 0
    while True:
        legal = env.legal_actions()
        proc = agents[current].observe(env.observation.get_observation())
        action = agents[current].act(proc, legal)
        obs, reward, done, info = env.step(current, action)
        steps += 1
        if done:
            print('Done after', steps, 'steps. Winner:', info['winner'])
            render_board(env.observation.get_observation())
            break
        current = 1 if current == 2 else 2


if __name__ == '__main__':
    run_smoke(N=10, seed=42)
