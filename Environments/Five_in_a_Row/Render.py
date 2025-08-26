from typing import Sequence, Optional, Iterable, Tuple


def render_board(board: Sequence[Sequence[int]], winning_coords: Optional[Iterable[Tuple[int, int]]] = None, show_empty_indices: bool = False) -> None:
    """ASCII renderer with optional winning highlight and empty-index display.

    - winning_coords: iterable of (r,c) to highlight (rendered in brackets [X]/[O]).
    - show_empty_indices: if True, empty cells render their linear index r*cols+c instead of '.'.
    """
    rows = len(board)
    cols = len(board[0]) if rows > 0 else 0
    winner_set = set(winning_coords) if winning_coords is not None else set()

    def symbol(r: int, c: int, v: int) -> str:
        if (r, c) in winner_set:
            if v == 1:
                return '[X]'
            if v == -1:
                return '[O]'
            return '[ ]'
        if v == 1:
            return ' X '
        if v == -1:
            return ' O '
        if show_empty_indices:
            idx = r * cols + c
            return f'{idx:3d}'
        return ' . '

    # column header
    header = '    ' + ' '.join(f'{c:3d}' for c in range(cols))
    print(header)
    for r in range(rows):
        rowstr = f'{r:3d} ' + ' '.join(symbol(r, c, int(board[r][c])) for c in range(cols))
        print(rowstr)
