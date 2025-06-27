"""
Evaluation function used in BCTS (Building controllers for Tetris)
by Thiery and Scherrer, combining their own features: Hole Depth and Rows with Holes
with features of Dellacherie's handmade controller.
"""

import numpy as np

import tetris_env

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
    broken_line_indices = game.broken_line_indices
    count_broken_lines = len(broken_line_indices)
    if count_broken_lines == 0:
        return 0
    bricks_eliminated_from_last_piece_added = 0
    for cell_index in game.current_tetromino.image(): # Tetromino just placed
        row_offset = cell_index // 4
        row_index = game.current_tetromino.y + row_offset
        if row_index in broken_line_indices: # lines broken by the last piece
            bricks_eliminated_from_last_piece_added += 1
    return count_broken_lines * bricks_eliminated_from_last_piece_added

def row_transitions(filled_cell_mask):
    """
    Number of horizontal full to empty or empty to full
    transitions between the cells on the board
    """
    return np.sum(filled_cell_mask[:, 1:] != filled_cell_mask[:, :-1])

def col_transitions(filled_cell_mask):
    """
    Number of vertical full to empty or empty to full
    transitions between the cells on the board
    """
    return np.sum(filled_cell_mask[1:, :] != filled_cell_mask[:-1, :])

def get_wells_score(filled_cell_mask, above_surface_mask):
    """
    Calculates the wells score of a grid using vectorised operations.
    This implementation's definition of a well is the first contiguous run of empty cells
    surrounded by walls moving down the column ending at the first filled
    cell or a cell which doesn't have walls on both sides.
    Note: this counts only exposed wells (exposed to the surface
    as opposed to buried wells)
    """

    # Padding of none (0, 0) along row axis (above, below) and (1, 1)
    # along column axis (left, right) with constant values True.
    # Creates padded grid where leftmost and rightmost boundaries
    # of each row are treated as walls (filled cells)
    padded_grid = np.pad(
        filled_cell_mask,
        pad_width=((0, 0), (1, 1)),
        mode='constant',
        constant_values=True
    )

    # Boolean masks for left and right walls immediately adjacent to each cell.
    # Values are True if cells are filled on their left or right sides respectively
    # with the padded grid allowing for checking against row boundaries (which are
    # considered filled)
    has_left_wall_mask = padded_grid[:, :-2]
    has_right_wall_mask = padded_grid[:, 2:]

    # A cell is part of a well if it has walls on both sides and is exposed to the surface.
    # above_surface_mask is True for any cell above the highest filled cell in each column
    # which logically implies they are also not filled.
    well_cells = has_left_wall_mask & has_right_wall_mask & above_surface_mask

    well_score = 0

    for col_index in range(well_cells.shape[1]):

        column = well_cells[:, col_index]

        # Skip columns that have no well cells
        if not column.any():
            continue

        # Pad with False values to catch contiguous wells starting or ending at the boundaries
        padded_col = np.concatenate(([False], column, [False]))
        # np.diff finds the start and end of any contiguous wells (after int conversion)
        # start of a well: F->T converted to +1 (a[i+1] - a[i] == (True - False))
        # end of a well: T->F converted to -1 (a[i+1] - a[i] == (False - True))
        diffs = np.diff(padded_col.astype(np.int8))

        # Extract the first contiguous well's start and end indices (at dimension 0)
        first_well_start_index = np.where(diffs == 1)[0][0]
        first_well_end_index = np.where(diffs == -1)[0][0]

        # The depth of the first well is the difference between its end and start index
        well_depth = first_well_end_index - first_well_start_index

        # The score for a well of depth d is the sum of the arithmetic series:
        # d * (d + 1) / 2
        well_score += np.sum(well_depth * (well_depth + 1) // 2)

    # Σ_w_∈_wells (1 + 2 + ... + depth(w))
    return well_score

def get_hole_features(cell_filled_mask, at_or_below_surface_mask):
    """
    Calculates count_holes, rows_with_holes, and hole_depth in a single
    pass using vectorised operations for efficiency
    """

    # A hole is not filled and below column height (surface mask)
    # hole_mask is a boolean array with shape (height, width) where True indicates a hole
    hole_mask = ~cell_filled_mask & at_or_below_surface_mask

    # True evaluates to 1, False to 0
    count_holes = np.sum(hole_mask)

    # +1 per row if any value in the hole_mask is True
    count_rows_with_holes = np.sum(np.any(hole_mask, axis=1))

    # Sum of filled cells above each hole, maintains shape (height, width)
    # The cumulative sum of filled cells is calculated from the top down for each column
    # The value at a hole's position represents its depth below the surface of the pile
    cumulative_filled = np.cumsum(cell_filled_mask, axis=0)
    # Boolean indexing to get count of filled cells above each hole, then sum over all holes
    total_hole_depth = np.sum(cumulative_filled[hole_mask])

    return count_holes, count_rows_with_holes, total_hole_depth

def evaluate_bcts(weight_vector, game):
    """Evaluate the Tetris grid using Thiery and Scherrers' BCTS feature set."""

    # Precompute shared inputs needed for feature calculations

    # Filled cells are represented by any non-zero value
    filled_cell_mask = game.grid > 0 # shape (height, width)
    column_heights = tetris_env.get_column_heights(filled_cell_mask)

    height = filled_cell_mask.shape[0]

    # Create broadcastable column vector of row indices
    row_indices = np.arange(height)[:, np.newaxis]
    # Compare each row index with indices of the column heights
    # broadcasts shape to (height, width)
    # at_or_below_surface_mask is True for any cell at or below the column height
    # where column heights are the highest filled cell in each column
    at_or_below_surface_mask = row_indices >= (height - column_heights)

    # This mask is True for cells that are above the highest filled cell in each column
    above_surface_mask = ~at_or_below_surface_mask

    # Calculate all hole-related features in single pass
    count_holes, count_rows_with_holes, sum_hole_depths = \
        get_hole_features(filled_cell_mask, at_or_below_surface_mask)

    features = np.array([
        landing_height(game),
        eroded_piece_cells(game),
        row_transitions(filled_cell_mask),
        col_transitions(filled_cell_mask),
        count_holes,
        get_wells_score(filled_cell_mask, above_surface_mask),
        sum_hole_depths,
        count_rows_with_holes
    ])

    return np.dot(weight_vector, features)
