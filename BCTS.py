"""
Evaluation function used in BCTS (Building controllers for Tetris)
by Thiery and Scherrer, combining their own features: Hole Depth and Rows with Holes
with features of Dellacherie's handmade controller.
"""

import tetris_env

def hole_depth(game):
    """
    The hole depth indicates how far holes are under the surface
    of the pile: it is the sum of the number of full cells above
    each hole
    """
    count=0
    h=tetris_env.column_height(game.field)
    for c in range(game.width):
        full_cells_above = 0
        for r in range(game.height-h[c], game.height):
            if game.field[r][c]==0:
                count += full_cells_above
            else:
                full_cells_above += 1
    return count

def rows_with_holes(game):
    """
    Counts the number of rows having at least one hole
    (two holes on the same row count for only one)
    """
    rows_containing_holes = set()
    h = tetris_env.column_height(game.field)
    for c in range(game.width):
        for r in range(game.height-h[c], game.height):
            if game.field[r][c]==0 and r not in rows_containing_holes:
                rows_containing_holes.add(r)
    return len(rows_containing_holes)

def landing_height(game):
    """
    Calculates landing height which is the row where the Tetromino's lowest
    filled cell will land after a hard drop.
    """
    fig = game.figure
    row = fig.y
    row_offset = 0
    for cell_number in fig.image():
        row_offset = max(row_offset, cell_number // 4)
    return row + row_offset

def eroded_piece_cells(game):
    """
    Returns the number of cells eroded by the
    Tetromino placement leading to this configuration.
    """
    return game.broken_lines * game.width

def row_transitions(game):
    """
    Number of horizontal full to empty or empty to full
    transitions between the cells on the board
    """
    count = 0
    for r in range(game.height):
        for c in range(1, game.width):
            if game.field[r][c] != game.field[r][c - 1]:
                count += 1
    return count

def col_transitions(game):
    """
    Number of vertical full to empty or empty to full
    transitions between the cells on the board
    """
    count = 0
    for c in range(game.width):
        for r in range(1, game.height):
            if game.field[r][c] != game.field[r - 1][c]:
                count += 1
    return count

def board_wells(game):
    """
    A well is a succession of unoccupied cells in a column
    such that their left cells and right cells are both occupied.
    For each well sums arithmetic series of well depth:
    Σw∈wells(1 + 2 +· · · + depth(w))
    """
    count = 0
    for c in range(game.width):
        r = 0
        while r < game.height and game.field[r][c] == 0:
            r += 1
        r -= 1 # r is now at the last empty cell descending from the top
        well_depth = 0
        while r > 0:
            left_occupied = (c == 0 or game.field[r][c - 1] > 0)
            right_occupied = (c == game.width - 1 or game.field[r][c + 1] > 0)
            if not (left_occupied and right_occupied):
                break
            well_depth += 1
            r -= 1
        count += (well_depth * (well_depth + 1)) // 2
    return count

def evaluate_BCTS(W, game):
    """Evaluate the Tetris grid using Thiery and Scherrers' feature set."""

    S0 = W[0] * landing_height(game)
    S1 = W[1] * eroded_piece_cells(game)
    S2 = W[2] * row_transitions(game)
    S3 = W[3] * col_transitions(game)
    S4 = W[4] * tetris_env.holes(game.field)
    S5 = W[5] * board_wells(game)
    S6 = W[6] * hole_depth(game)
    S7 = W[7] * rows_with_holes(game)

    return S0 + S1 + S2 + S3 + S4 + S5 + S6 + S7
