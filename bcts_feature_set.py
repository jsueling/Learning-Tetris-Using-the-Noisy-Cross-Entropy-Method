"""
Evaluation function used in BCTS (Building controllers for Tetris)
by Thiery and Scherrer, combining their own features: Hole Depth and Rows with Holes
with features of Dellacherie's handmade controller.
"""

import numpy as np

import tetris_env

def hole_depth(game, column_heights):
    """
    The hole depth indicates how far holes are under the surface
    of the pile: it is the sum of the number of full cells above
    each hole
    """
    count = 0
    for c in range(game.width):
        full_cells_above = 0
        for r in range(game.height - column_heights[c], game.height):
            if game.grid[r][c] == 0:
                count += full_cells_above
            else:
                full_cells_above += 1
    return count

def rows_with_holes(game, column_heights):
    """
    Counts the number of rows having at least one hole
    (two holes on the same row count for only one)
    """
    rows_containing_holes = set()
    for c in range(game.width):
        for r in range(game.height-column_heights[c], game.height):
            if game.grid[r][c] == 0 and r not in rows_containing_holes:
                rows_containing_holes.add(r)
    return len(rows_containing_holes)

def landing_height(game):
    """
    Calculated as the height of the lowest cell of the last
    placed Tetromino leading to this grid configuration (ranges from 1 to grid height)
    """
    current_tetromino = game.current_tetromino
    row = current_tetromino.y
    row_offset = 0
    for cell_number in current_tetromino.image():
        row_offset = max(row_offset, cell_number // 4)
    return game.height - (row + row_offset)

def eroded_piece_cells(game):
    """
    (Number of rows eliminated in the last
    move) * (Number of bricks eliminated
    from the last piece added)
    """
    bricks_eliminated_from_last_piece_added = 0
    for cell_index in game.current_tetromino.image(): # Tetromino just placed
        row_offset = cell_index // 4
        row_index = game.current_tetromino.y + row_offset
        if row_index in game.broken_line_indices: # lines broken by the last piece
            bricks_eliminated_from_last_piece_added += 1
    return len(game.broken_line_indices) * bricks_eliminated_from_last_piece_added

def row_transitions(grid_filled):
    """
    Number of horizontal full to empty or empty to full
    transitions between the cells on the board
    """
    return np.sum(grid_filled[:, 1:] != grid_filled[:, :-1])

def col_transitions(grid_filled):
    """
    Number of vertical full to empty or empty to full
    transitions between the cells on the board
    """
    return np.sum(grid_filled[1:, :] != grid_filled[:-1, :])

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
        while r < game.height and game.grid[r][c] == 0:
            r += 1
        r -= 1 # r is now at the last empty cell descending from the top
        well_depth = 0
        while r > 0:
            left_occupied = (c == 0 or game.grid[r][c - 1] > 0)
            right_occupied = (c == game.width - 1 or game.grid[r][c + 1] > 0)
            if not (left_occupied and right_occupied):
                break
            well_depth += 1
            r -= 1
        count += (well_depth * (well_depth + 1)) // 2
    return count

def evaluate_bcts(weight_vector, game):
    """Evaluate the Tetris grid using Thiery and Scherrers' BCTS feature set."""

    grid_filled = (game.grid > 0) # Filled cells are represented by any non-zero value
    column_heights = tetris_env.get_column_heights(grid_filled)

    return sum([
        weight_vector[0] * landing_height(game),
        weight_vector[1] * eroded_piece_cells(game),
        weight_vector[2] * row_transitions(grid_filled),
        weight_vector[3] * col_transitions(grid_filled),
        weight_vector[4] * tetris_env.count_holes(grid_filled, column_heights),
        weight_vector[5] * board_wells(game),
        weight_vector[6] * hole_depth(game, column_heights),
        weight_vector[7] * rows_with_holes(game, column_heights)
    ])
