import numpy as np
from typing import Tuple, Optional


def judge(board: np.ndarray, last_move: Optional[Tuple[int, int]] = None, require_exact_five: bool = False) -> Tuple[bool, int]:
    """Check game result (game_over, winner).

    If `last_move` is provided, only sequences passing through that cell are checked
    (fast, suitable for immediate post-move judgment). If not provided, whole-board
    scan is performed (legacy behavior).

    If `require_exact_five` is True, a winning line must have exactly length 5
    (overlines >5 are not counted).
    """
    if not isinstance(board, np.ndarray):
        board = np.array(board)

    rows, cols = board.shape
    directions = ((0, 1), (1, 0), (1, 1), (-1, 1))

    def check_at(r0: int, c0: int) -> Optional[int]:
        val = board[r0, c0]
        if val == 0:
            return None
        for dr, dc in directions:
            count = 1
            # forward
            rr, cc = r0 + dr, c0 + dc
            while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == val:
                count += 1
                rr += dr
                cc += dc
            # backward
            rr, cc = r0 - dr, c0 - dc
            while 0 <= rr < rows and 0 <= cc < cols and board[rr, cc] == val:
                count += 1
                rr -= dr
                cc -= dc

            if require_exact_five:
                if count == 5:
                    return int(val)
            else:
                if count >= 5:
                    return int(val)
        return None

    # If last_move provided, only check lines through that cell
    if last_move is not None:
        r0, c0 = last_move
        if not (0 <= r0 < rows and 0 <= c0 < cols):
            return False, 0
        winner = check_at(r0, c0)
        if winner is not None:
            return True, winner

        # no win at last_move; check draw
        if not np.any(board == 0):
            return True, 0
        return False, 0

    # Legacy: scan whole board
    for r in range(rows):
        for c in range(cols):
            if board[r, c] == 0:
                continue
            winner = check_at(r, c)
            if winner is not None:
                return True, winner

    if not np.any(board == 0):
        return True, 0
    return False, 0
    