from __future__ import annotations

from collections import namedtuple, defaultdict
from random import choice
from typing import Set, Tuple, Optional, Iterable, Any, cast
from mcts import MCTS, Node
import os
import argparse
import dill

_Tree = namedtuple("Tree", "state terminal turn winner space")


class Tree(_Tree, Node):
    """Immutable game-state node for Tic-tac-toe that implements the Node API.

    Fields (from namedtuple):
        state: tuple of length 9 with values in {None, 1, -1}
        terminal: bool indicating whether the state is terminal
        turn: int (1 or -1) indicating which player's turn the node represents
        winner: int in {1, -1, 0} where 0 means no winner yet
        space: number of empty spaces remaining (int)
    """

    def find_children(self) -> Set[Node]:
        """Return all legal child nodes (resulting states) from this node.

        Returns an empty set for terminal nodes.
        """
        if self.terminal:
            return set()
        # generate immutable child Tree instances for each empty square
        # Node-typed return keeps signature compatible with Node ABC
        return {self.make_move(i) for i in range(9) if self.state[i] is None}

    def find_random_child(self) -> Node:
        """Return a random legal child node (used for simulation)."""
        possible_move = [i for i in range(9) if self.state[i] is None]
        return self.make_move(choice(possible_move))

    def reward(self, randomness: Optional[Any] = None) -> float:
        """Return scalar reward for terminal node.

        Reward convention: 1 for player 1 win, 0.5 for draw, 0 for loss.
        The `randomness` parameter is accepted for API compatibility.
        """
        if self.terminal:
            if self.winner == 1:
                return 1.0
            elif self.winner == 0:
                return 0.5
            else:
                return 0.0
        else:
            return 0.05

    def is_terminal(self) -> bool:
        """Return True when the node is terminal."""
        return self.terminal

    def make_move(self, k: int) -> Node:
        """Return a new Tree corresponding to playing `k` (0..8) on this node.

        The returned Tree is immutable (namedtuple) and contains updated
        fields: state, terminal, turn, winner, space.
        """
        state = self.state[:k] + (self.turn,) + self.state[k + 1:]

        turn = -self.turn
        winner = find_winner(state)
        space = self.space - 1
        is_terminal = (winner != 0) or (space == 0)
        # sanity-checks for unexpected logic errors (kept from original)
        if is_terminal and not (winner != 0 or all(s is not None for s in state)):
            print(
                "case 1",
                Tree(state=state, terminal=is_terminal, turn=turn, winner=winner, space=space),
            )
        if (winner != 0 or all(s is not None for s in state)) and not is_terminal:
            print(
                "case 2",
                Tree(state=state, terminal=is_terminal, turn=turn, winner=winner, space=space),
            )
        return Tree(state=state, terminal=is_terminal, turn=turn, winner=winner, space=space)


def find_winner(state: Tuple[Optional[int], ...]) -> int:
    """Return the winner of the given board state.

    Args:
        state: tuple of length 9 with values {None, 1, -1}.

    Returns:
        1 if player 1 has won, -1 if player -1 has won, 0 otherwise.
    """
    winning_combos = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  
        [0, 4, 8], [2, 4, 6]  
    ]
    for combo in winning_combos:
        s = 0
        for i in combo:
            val = state[i]
            if val is None:
                break
            # val is int here; cast to satisfy static checkers
            s += cast(int, val)
        if s == 3 or s == -3:
            return 1 if s == 3 else -1
    return 0


def play_game_uct(
    budget: int = 1000,
    exploration_weight: float = 1,
    optimum: int = 4,
    n0: int = 5,
    opp: str = 'random',
    muu_0: float = 1,
    sigmaa_0: float = 1,
    sigma_0: float = 1,
    opp_first_move: int = 0,
) -> Tuple[MCTS, Tree, Node]:
    """Run MCTS with UCT selection and return (mcts, root, chosen_child).

    The function performs `budget` rollouts then picks the best child.
    """
    mcts = MCTS(
        policy='uct',
        exploration_weight=exploration_weight,
        budget=budget,
        n0=n0,
        opp_policy=opp,
        sigma_0=sigma_0,
        muu_0=muu_0,
        sigmaa_0=sigmaa_0,
    )
    tree = new_tree(budget=budget, opp_first_move=opp_first_move)

    for _ in range(budget):
        mcts.do_rollout(tree)

    next_tree = mcts.choose(tree)
    return (mcts, tree, next_tree)


def play_game_ocba(
    budget: int = 1000,
    optimum: int = 0,
    n0: int = 5,
    opp: str = 'random',
    muu_0: float = 1,
    sigmaa_0: float = 1,
    sigma_0: float = 1,
    opp_first_move: int = 0,
) -> Tuple[MCTS, Tree, Node]:
    """Run MCTS with OCBA selection and return (mcts, root, chosen_child)."""
    mcts = MCTS(policy='ocba', budget=budget, n0=n0, opp_policy=opp, sigma_0=sigma_0, muu_0=muu_0, sigmaa_0=sigmaa_0)
    tree = new_tree(budget=budget, opp_first_move=opp_first_move)

    for _ in range(budget):
        mcts.do_rollout(tree)
    next_tree = mcts.choose(tree)
    return (mcts, tree, next_tree)

def play_game_AOAP(
    budget: int = 1000,
    optimum: int = 0,
    n0: int = 5,
    opp: str = 'random',
    muu_0: float = 1,
    sigmaa_0: float = 1,
    sigma_0: float = 1,
    opp_first_move: int = 0,
) -> Tuple[MCTS, Tree, Node]:
    """Run MCTS with AOAP selection and return (mcts, root, chosen_child)."""
    mcts = MCTS(policy='AOAP', budget=budget, n0=n0, opp_policy=opp, sigma_0=sigma_0, muu_0=muu_0, sigmaa_0=sigmaa_0)
    tree = new_tree(budget=budget, opp_first_move=opp_first_move)

    for _ in range(budget):
        mcts.do_rollout(tree)
    next_tree = mcts.choose(tree)
    return (mcts, tree, next_tree)

def new_tree(budget: int = 1000, opp_first_move: int = 0) -> Tree:
    """Create a new root Tree with the opponent's first move inserted.

    The root has `turn=1` (agent to move) and `space=8` empty squares.
    """
    root = (None,) * opp_first_move + (-1,) + (None,) * (8 - opp_first_move)
    return Tree(state=root, terminal=False, turn=1, winner=0, space=8)


if __name__ == "__main__":
    os.makedirs("results/tmp", exist_ok=True)
    os.makedirs("ckpt", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', type=int,
                        help='number of replications', default=100000)
    parser.add_argument('--budget_start', type=int,
                        help='budget (number of rollouts) starts from (inclusive)', default=400)
    parser.add_argument('--budget_end', type=int,
                        help='budget (number of rollouts) end at (inclusive)', default=400) #add
    parser.add_argument('--step', type=int,
                        help='stepsize in experiment', default=20)
    parser.add_argument(
        '--n0', type=int, help='initial samples to each action', default=10)
    parser.add_argument('--muu_0', type=int,
                        help='prior mean', default=0)
    parser.add_argument('--sigmaa_0', type=int,
                        help='prior variance', default=10)
    parser.add_argument('--sigma_0', type=int,
                        help='initial variance', default=0.01)
    parser.add_argument('--opp_policy', type=str,
                        help='opponent (Player 1) policy, must be either uct or random', default='random')
    parser.add_argument('--opp_first_move', type=int,
                        help='the first move of opponent (Player 1), must be either 2 or 4 (correspond to setup 1 and 2, resp.)', default=2)#add
    parser.add_argument(
        '--checkpoint', type=str, help='relative path to checkpoint', default='')

    parser.set_defaults(opp_policy='random')  
    parser.set_defaults(opp_first_move=4) 
    args = parser.parse_args()

    rep = args.rep
    budget_start = args.budget_start
    budget_end = args.budget_end
    step = args.step
    budget_range = range(budget_start, budget_end+1, step)
    n0 = args.n0
    muu_0 = args.muu_0
    sigmaa_0 = args.sigmaa_0
    sigma_0 = args.sigma_0
    opp_policy = args.opp_policy
    opp_first_move = args.opp_first_move
    results_uct = []
    results_ocba = []
    results_AOAP = []
    uct_selection = []
    ocba_selection = []
    AOAP_selection = []
    exploration_weight = 1
    uct_visit_ave_cnt_list, ocba_visit_ave_cnt_list, AOAP_visit_ave_cnt_list = [], [], []
   
    if opp_first_move == 2:
        optimal_set = {4}
        setup = 1
    elif opp_first_move == 4:
        optimal_set = {0, 2, 6, 8}
        setup = 2
    else:
        raise ValueError('opp_first_move should either be 2 or 4.')

    ckpt = args.checkpoint

    if ckpt != '':
        dill.load_session(ckpt)
        budget_range = range(budget_start, budget_end+1, step)

    for budget in budget_range:
        PCS_uct = 0
        PCS_ocba = 0
        PCS_AOAP = 0
        uct_selection.append([])
        ocba_selection.append([])
        AOAP_selection.append([])

        uct_visit_cnt, ocba_visit_cnt, AOAP_visit_cnt, TTTS_visit_cnt = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)
       
        for i in range(rep):
            uct_mcts, uct_root_node, uct_cur_node = play_game_uct(
                budget=budget,
                exploration_weight=exploration_weight,
                n0=n0,
                opp=opp_policy,
                muu_0 = muu_0, 
                sigmaa_0 = sigmaa_0,
                sigma_0=sigma_0,
                opp_first_move=opp_first_move
            )
            PCS_uct += any(uct_cur_node.state[i] for i in optimal_set)

            ocba_mcts, ocba_root_node, ocba_cur_node = play_game_ocba(
                budget=budget,
                n0=n0,
                opp=opp_policy,
                muu_0 = muu_0, 
                sigmaa_0 = sigmaa_0,
                sigma_0=sigma_0,
                opp_first_move=opp_first_move
            )
            PCS_ocba += any(ocba_cur_node.state[i] for i in optimal_set)

            AOAP_mcts, AOAP_root_node, AOAP_cur_node = play_game_AOAP(
                budget=budget,
                n0=n0,
                opp=opp_policy,
                muu_0 = muu_0, 
                sigmaa_0 = sigmaa_0,
                sigma_0=sigma_0,
                opp_first_move=opp_first_move
            )
            PCS_AOAP += any(AOAP_cur_node.state[i] for i in optimal_set)
            
            '''
            Update the ave dict
            '''
            uct_visit_cnt.update(dict(
                (c, uct_visit_cnt[c]+uct_mcts.N[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_visit_cnt.update(dict(
                (c, ocba_visit_cnt[c]+ocba_mcts.N[c]) for c in ocba_mcts.children[ocba_root_node]))
            AOAP_visit_cnt.update(dict(
                (c, AOAP_visit_cnt[c] + AOAP_mcts.N[c]) for c in AOAP_mcts.children[AOAP_root_node]))

            if (i+1) % 100 == 0:
                print('%0.2f%% finished for budget limit %d' %
                      (100*(i+1)/rep, budget))
                print('Current PCS: uct=%0.3f, ocba=%0.3f, AOAP=%0.3f' %
                      (PCS_uct/(i+1), (PCS_ocba/(i+1)), (PCS_AOAP/(i+1))))
                
        uct_visit_cnt.update(
            dict((c, uct_visit_cnt[c]/rep) for c in uct_mcts.children[uct_root_node]))
        ocba_visit_cnt.update(
            dict((c, ocba_visit_cnt[c]/rep) for c in ocba_mcts.children[ocba_root_node]))
        AOAP_visit_cnt.update(
            dict((c, AOAP_visit_cnt[c] / rep) for c in AOAP_mcts.children[AOAP_root_node]))

        uct_visit_ave_cnt_list.append(uct_visit_cnt)
        ocba_visit_ave_cnt_list.append(ocba_visit_cnt)
        AOAP_visit_ave_cnt_list.append(AOAP_visit_cnt)

        results_uct.append(PCS_uct/rep)
        results_ocba.append(PCS_ocba/rep)
        results_AOAP.append(PCS_AOAP/rep)

        print("Budget %d has finished" % (budget))
        print('PCS_uct = %0.3f, PCS_ocba = %0.3f, PCS_AOAP = %0.3f' %
              (PCS_uct/rep, PCS_ocba/rep, PCS_AOAP/rep))
        ckpt_output = 'ckpt/tic_tac_toe_{opp_policy}_opponent_setup{setup}.pkl'.format(
            opp_policy=opp_policy, setup=setup)
        results_output = 'results/tmp/tic_tac_toe_{opp_policy}_opponent_setup{setup}.pkl'.format(
            opp_policy=opp_policy, setup=setup)

        dill.dump_session(ckpt_output)