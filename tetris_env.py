"""Tetris environment for AI agents"""

import random

import numpy as np
import matplotlib.pyplot as plt
import imageio

from bcts_feature_set import evaluate_bcts

class Tetromino:
    """Represents a Tetromino with its position, type, and rotation."""

    figures = [
        # Each figure is represented by a flattened 4x4 dimension
        # with a list of each of the rotations it can take
        [[0, 4, 8, 12], [0, 1, 2, 3]], # I
        [[0, 1, 5, 6], [1, 4, 5, 8]], # Z
        [[4, 5, 1, 2], [0, 4, 5, 9]], # S
        [[1, 0, 4, 8], [0, 4, 5, 6], [1, 5, 9, 8], [0, 1, 2, 6]], # J
        [[0, 1, 5, 9], [4, 0, 1, 2], [0, 4, 8, 9], [4, 5, 6, 2]], # L
        [[1, 4, 5, 6], [1, 4, 5, 9], [0, 1, 2, 5], [0, 4, 8, 5]], # T
        [[0, 1, 4, 5]], # O
    ]

    default_spawns = [
        # Default spawns are horizontally centered along the top row
        # format: column, rotation
        [3, 1],  # I
        [3, 0],  # Z
        [3, 0],  # S
        [3, 3],  # J
        [3, 1],  # L
        [3, 2],  # T
        [4, 0],  # O
    ]

    # Each Tetromino's position is represented by
    # The following 4x4 grid:

    #  0  1  2  3
    #  4  5  6  7
    #  8  9 10 11
    # 12 13 14 15

    # The Tetromino's (x, y) coordinates reference the position
    # of where cell 0 in the 4x4 grid is located in the 10x20 grid.

    def __init__(self, x, y, fig_type, rotation):
        self.x = x #position de la pièce sur la largeur du jeu
        self.y = y #position de la pièce sur la longueur du jeu
        self.type = fig_type #type de la pièce entre 0 et 6
        self.rotation = rotation #rotation de la pièce

    def image(self):
        """Returns the current 4x4 image of the Tetromino based on its type and rotation."""
        return self.figures[self.type][self.rotation]

class Tetris:
    """
    Represents the Tetris game state,
    including the grid, current tetromino, score, and game state.
    """
    def __init__(self, height, width, tetromino_randomisation_scheme="uniform"):
        self.current_tetromino = None
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=int)
        self.score = 0
        self.state = "start"
        # indices of rows broken by the last Tetromino placement
        self.broken_line_indices = set()

        # scheme can be type "uniform" or "bag"
        self.tetromino_randomisation_scheme = tetromino_randomisation_scheme

        if self.tetromino_randomisation_scheme == "bag":
            # Initialise a bag of random Tetrominoes
            self.bag = list(range(len(Tetromino.figures)))
            random.shuffle(self.bag)

    def new_tetromino(self, fig_type, x, y, rotation):
        """Creates a new Tetromino at the specified (x, y) and rotation."""
        self.current_tetromino = Tetromino(x, y, fig_type, rotation)

    def get_next_piece(self):
        """
        Returns the next piece type based on the randomisation scheme.
        For "uniform", it returns a random piece type.
        For "bag", it returns a piece type from a bag containing each
        Tetromino in a random order, refilling and reshuffling it when empty.
        """

        if self.tetromino_randomisation_scheme == "uniform":
            # Randomly select a piece type uniformly
            return random.randint(0, 6)

        if self.tetromino_randomisation_scheme == "bag":
            if not self.bag:
                self.bag = list(range(len(Tetromino.figures)))
                random.shuffle(self.bag)
            return self.bag.pop()

        raise ValueError(
            f"Invalid tetromino randomisation scheme: {self.tetromino_randomisation_scheme}"
        )

    def intersects(self):
        """
        Returns True if the current Tetromino placement is invalid or False otherwise
        (OOB and collision checks).
        """
        x, y = self.current_tetromino.x, self.current_tetromino.y

        for cell_index in self.current_tetromino.image():
            tetromino_row = y + (cell_index // 4)
            tetromino_col = x + (cell_index % 4)
            if ( # OOB checks and collision check
                tetromino_row < 0 or
                tetromino_row >= self.height or
                tetromino_col >= self.width or
                tetromino_col < 0 or
                self.grid[tetromino_row][tetromino_col] > 0
            ):
                return True
        return False

    def break_lines(self):
        """Break lines that are completely filled by Tetrominoes."""

        filled_lines = np.all(self.grid > 0, axis=1)
        # All filled lines now become broken lines
        broken_lines = np.count_nonzero(filled_lines)
        if broken_lines > 0:
            self.grid = np.vstack((
                # Add empty rows at the top to replace the broken lines
                np.zeros((broken_lines, self.width), dtype=int),
                # Keep only rows that were not filled (maintains ordering)
                self.grid[~filled_lines]
            ))
            # Store the indices of the most recently broken lines (used in eroded_piece_cells)
            self.broken_line_indices = set(np.flatnonzero(filled_lines))
            self.score += broken_lines

    def hard_drop(self, colour=1):
        """Move the current tetromino directly down to the bottom of the grid."""
        while not self.intersects():
            self.current_tetromino.y += 1
        self.current_tetromino.y -= 1
        self.freeze(colour)

    def freeze(self, colour):
        """Freeze the current tetromino, it now becomes part of the grid."""
        x, y = self.current_tetromino.x, self.current_tetromino.y
        for cell_index in self.current_tetromino.image():
            tetromino_row = y + (cell_index // 4)
            tetromino_col = x + (cell_index % 4)
            self.grid[tetromino_row][tetromino_col] = colour
        self.break_lines()

    def path_exists_to_col(self, target_column):
        """
        Check if there is a valid path across the top row to the desired column
        for the Tetromino hard drop placement. This assumes that
        all shifts and rotations are possible during lock delay.
        """
        if target_column < self.current_tetromino.x:
            for col in range(self.current_tetromino.x, target_column - 1, -1):
                if self.grid[0][col] > 0 or self.grid[1][col] > 0:
                    return False
        elif target_column > self.current_tetromino.x:
            for col in range(self.current_tetromino.x, target_column + 1):
                if self.grid[0][col] > 0 or self.grid[1][col] > 0:
                    return False
        return True

def get_column_heights(grid_filled):
    """Returns the height of each column in the grid in order as a list."""

    heights = np.zeros((grid_filled.shape[1],), dtype=int)
    for col in range(grid_filled.shape[1]):
        if grid_filled[:, col].any():
            # If the column has filled cells, calculate its height.
            # argmax returns the index of the first True value traversed
            # from top to bottom i.e. the first filled cell for this column
            heights[col] = grid_filled.shape[0] - grid_filled[:, col].argmax()
        # Otherwise the column height is 0 and no action is needed
    return heights

def get_adj_col_height_diffs(grid):
    """Returns the absolute difference between all adjacent columns."""
    adj_col_height_diffs = []
    column_heights = get_column_heights(grid)

    for j in range(9):
        adj_col_height_diffs.append(abs(column_heights[j+1]-column_heights[j]))

    return adj_col_height_diffs

def count_holes(grid_filled, column_heights):
    """Count the number of inaccessible holes in the Tetris grid."""
    hole_count = 0
    for col in range(grid_filled.shape[1]):
        # For each column, accumulate and count holes below the
        # row of the highest filled cell in the column
        start_row = grid_filled.shape[0] - column_heights[col] + 1
        hole_count += np.count_nonzero(grid_filled[start_row:, col] == 0)
    return hole_count

# Evalue la configuration de la grille en pondérant les features par le vecteur W de taille 21
def evaluate_bertsekas(weight_vector, game):
    """Evaluate the Tetris grid using Bertsekas and Tsitsiklis' feature set."""
    # weight_vector = [w1, ..., w21] vector of parameters to tune

    grid = game.grid
  # Convert to boolean grid for filled cells
    grid_filled = (grid > 0)

    col_heights = get_column_heights(grid_filled)
    adj_col_height_diffs = get_adj_col_height_diffs(grid)
    holes = count_holes(grid_filled, col_heights)
    max_col_height = max(col_heights)

    score = 0

    for col_index, height in enumerate(col_heights):
        score += height * weight_vector[col_index]

    for col_index, diff in enumerate(adj_col_height_diffs):
        score += diff * weight_vector[10 + col_index]

    score += weight_vector[19] * holes

    score += weight_vector[20] * max_col_height

    return score

def evaluate_best_move(weight_vector, grid, fig_type, colour):
    """
    Evaluates all valid placements and returns the best column and rotation.
    """

    # If no valid moves are found, return invalid move since the game is over
    best_move = (100, 0)
    best_score = float('inf')

    # Iterate through all possible rotations and columns for
    # the current Tetromino (ignoring symmetrical rotations)
    for rotation in range(len(Tetromino.figures[fig_type])):
        for col in range(10):

            game_copy = Tetris(20, 10)
            # Copy the current grid to the game copy
            np.copyto(game_copy.grid, grid)

            game_copy.new_tetromino(fig_type, col, 0, rotation)

            # Checks if target rotation is valid at the target column
            if game_copy.intersects():
                continue

            game_copy.hard_drop(colour)

            score = evaluate_bcts(weight_vector, game_copy)
            if score < best_score:
                best_score = score
                best_move = (col, rotation)

    return best_move

#simule une partie
def simulation(weight_vector, seed=None, tetromino_randomisation_scheme=None):
    """
    Simulates a Tetris game with the given weight vector W for its evaluation function.
    returns the final score of the game.
    """

    if tetromino_randomisation_scheme not in ["uniform", "bag"]:
        raise ValueError(
            "tetromino_randomisation_scheme must be set to either 'uniform' or 'bag'."
        )
    if seed is None:
        raise ValueError("Seed must be provided for reproducibility.")

    random.seed(seed)
    np.random.seed(seed)

    game = Tetris(20, 10, tetromino_randomisation_scheme=tetromino_randomisation_scheme)
    while game.state != "gameover":

        fig_type = game.get_next_piece()

        colour = 1

        # Evaluates all possible columns and rotations for the current Tetromino
        col, rotation = evaluate_best_move(weight_vector, game.grid, fig_type, colour)

        # Attempt to place the Tetromino in the best column and rotation
        game.new_tetromino(fig_type, col, 0, rotation)
        # evaluate_best_move may return invalid moves
        if game.intersects():
            game.state = "gameover"
        else:
            game.hard_drop(colour)

    return game.score

def simulation_data_collection(weight_vector, max_samples=1000, sample_freq=10):
    """
    Simulates a Tetris game with the given weight vector W for its evaluation function
    and returns sample grids (concatenation of flattened grid and one-hot encoded piece).
    """

    game = Tetris(20, 10)
    samples = []

    move_counter = 0
    while len(samples) < max_samples:

        fig_type = game.get_next_piece()

        colour = 1

        col, rotation = evaluate_best_move(weight_vector, game.grid, fig_type, colour)

        game.new_tetromino(fig_type, col, 0, rotation)

        if game.intersects():
            break

        game.hard_drop(colour)

        if move_counter > 0 and move_counter % sample_freq == 0:
            # 200 binary features for the grid (20x10 flattened)
            samples.append(game.grid.flatten())

        move_counter += 1

    return samples

def simulation_gif(weight_vector, num_moves=100): #Pas encore optimisé pour les pièces qui arrivent en haut
    """
    Simulates a Tetris game with the given weight vector W for its evaluation function
    and saves the frames as a GIF
    """

    with imageio.get_writer('tetris.gif', mode='I', fps=50) as writer:

        game = Tetris(20, 10)

        for _ in range(num_moves):

            fig_type = game.get_next_piece()
            colour = random.randint(1, 4)

            col, rotation = evaluate_best_move(weight_vector, game.grid, fig_type, colour)

            game.new_tetromino(fig_type, col, 0, rotation)

            if game.intersects():
                break

            game.hard_drop(colour)

            fig, ax = plt.subplots()
            ax.set_title(str(game.score))
            ax.matshow(game.grid, cmap='Reds')
            fig.canvas.draw()
            image = imageio.core.asarray(fig.canvas.renderer.buffer_rgba())
            writer.append_data(image)
            plt.close(fig)
