import random
import copy

import numpy as np
import matplotlib.pyplot as plt
import imageio

from bcts_feature_set import evaluate_bcts

class Figure:
    """Represents a Tetromino with its position, type, and rotation."""
    x = 0
    y = 0
    # liste des 6 différentes figures et leur rotation
    figures = [
        # Each figure is represented by a flattened 4x4 dimension
        # with a list of each 4 rotations it can take
        [[0, 4, 8, 12], [0, 1, 2, 3], [0, 4, 8, 12], [0, 1, 2, 3]], # I
        [[0, 1, 5, 6], [1, 4, 5, 8], [0, 1, 5, 6], [1, 4, 5, 8]], # Z
        [[4, 5, 1, 2], [0, 4, 5, 9], [4, 5, 1, 2], [0, 4, 5, 9]], # S
        [[1, 0, 4, 8], [0, 4, 5, 6], [1, 5, 9, 8], [0, 1, 2, 6]], # J
        [[0, 1, 5, 9], [4, 0, 1, 2], [0, 4, 8, 9], [4, 5, 6, 2]], # L
        [[1, 4, 5, 6], [1, 4, 5, 9], [0, 1, 2, 5], [0, 4, 8, 5]], # T
        [[0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5]], # O
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

    # Each Figure's position is represented by
    # The following 4x4 grid:

    #  0  1  2  3
    #  4  5  6  7
    #  8  9 10 11
    # 12 13 14 15

    # The Figure's (x, y) coordinates reference the position
    # of where cell 0 in the 4x4 grid is located in the 10x20 grid.

    def __init__(self, x, y, fig_type, rotation): 
        self.x = x #position de la pièce sur la largeur du jeu 
        self.y = y #position de la pièce sur la longueur du jeu
        self.type = fig_type #type de la pièce entre 0 et 6
        self.rotation = rotation #rotation de la pièce

    #séléction de la pièce (type et rotation) dans la liste figures
    def image(self):
        return self.figures[self.type][self.rotation]

class Tetris:
    """
    Represents the Tetris game state,
    including the field, current figure, score, and game state.
    """
    def __init__(self, height, width): #initialisation du jeu 

        self.figure = None
        self.height = height
        self.width = width
        self.field = np.zeros((height, width), dtype=int)
        self.score = 0
        self.state = "start"
        # indices of rows broken by the last Tetromino placement
        self.broken_line_indices = []

    def new_figure(self,fig_type,x,y,rotation):
        self.figure = Figure(x, y,fig_type,rotation) #introduction d'une nouvelle figure type en (x,y) 

    def intersects(self): #check if the currently flying figure intersecting with something fixed on the field. 
        intersection = False
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    if i + self.figure.y > self.height - 1 or \
                            j + self.figure.x > self.width - 1 or \
                            j + self.figure.x < 0 or \
                            self.field[i + self.figure.y][j + self.figure.x] > 0:
                        intersection = True
        return intersection



    def break_lines(self):
        """Break lines that are completely filled by Tetrominoes."""

        # credits to https://github.com/nuno-faria/tetris-ai/blob/master/tetris.py#L161
        lines_to_clear = [index for index, row in enumerate(self.field) if all(row > 0)]
        broken_lines = len(lines_to_clear)
        if broken_lines > 0:
            self.field = np.array([
                self.field[row_index] for row_index in range(self.height)
                if row_index not in lines_to_clear
            ])
            # Add new lines at the top
            for _ in lines_to_clear:
                self.field = np.insert(self.field, 0, [0 for _ in range(self.width)], axis=0)
            self.broken_line_indices = lines_to_clear
            self.score += broken_lines #** 2 -- remove Tetris line-clear bonus for now

    def hard_drop(self,color):
        """Move the current figure directly down to the bottom of the field."""
        while not self.intersects():
            self.figure.y += 1
        self.figure.y -= 1
        self.freeze(color)

    def freeze(self,color):
        """Freeze the current figure, it now becomes part of the field."""
        for i in range(4):
            for j in range(4):
                if i * 4 + j in self.figure.image():
                    self.field[i + self.figure.y][j + self.figure.x] = color
        self.break_lines()

    def path_exists_to_col(self, target_column):
        """
        Check if there is a valid path across the top row to the desired column
        for the Tetromino hard drop placement. This assumes that
        all shifts and rotations are possible during lock delay.
        """
        if target_column < self.figure.x:
            for col in range(self.figure.x, target_column - 1, -1):
                if self.field[0][col] > 0 or self.field[1][col] > 0:
                    return False
        elif target_column > self.figure.x:
            for col in range(self.figure.x, target_column + 1):
                if self.field[0][col] > 0 or self.field[1][col] > 0:
                    return False
        return True

def get_column_heights(field):
    """Returns the height of each column in the grid in order as a list."""
    column_heights = []
    for col in range(10):
        row_pointer = 0 # pointer to contiguous empty cells in the column
        while row_pointer < 20 and field[row_pointer][col] == 0:
            row_pointer += 1
        col_height = 20 - row_pointer
        column_heights.append(col_height)
    return column_heights

def get_adj_col_height_diffs(field):
    """Returns the absolute difference between all adjacent columns."""
    adj_col_height_diffs = []
    column_heights = get_column_heights(field)

    for j in range(9):
        adj_col_height_diffs.append(abs(column_heights[j+1]-column_heights[j]))

    return adj_col_height_diffs

def count_holes(field):
    """Count the number of inaccessible holes in the Tetris grid."""
    hole_count = 0
    column_heights = get_column_heights(field)

    for col_index in range(10):
        for row_index in range(20-column_heights[col_index], 20):
            if field[row_index][col_index] == 0:
                hole_count += 1

    return hole_count

# Evalue la configuration de la grille en pondérant les features par le vecteur W de taille 21
def evaluate_bertsekas(weight_vector, game):
    """Evaluate the Tetris grid using Bertsekas and Tsitsiklis' feature set."""
    # weight_vector = [w1, ..., w21] vector of parameters to tune

    field = game.field

    col_heights = get_column_heights(field)
    adj_col_height_diffs = get_adj_col_height_diffs(field)
    holes = count_holes(field)
    max_col_height = max(col_heights)

    score = 0

    for col_index, height in enumerate(col_heights):
        score += height * weight_vector[col_index]

    for col_index, diff in enumerate(adj_col_height_diffs):
        score += diff * weight_vector[10 + col_index]

    score += weight_vector[19] * holes

    score += weight_vector[20] * max_col_height

    return score

def evaluate_best_move(weight_vector, field, fig_type, color):
    """
    Evaluates all valid placements and returns the best column and rotation.
    """

    candidate_moves = []
    score = []
    for rotation in range(4):
        for col in range(10):

            game_copy = Tetris(20, 10)

            game_copy.field = copy.deepcopy(field)

            game_copy.new_figure(fig_type, col, 0, rotation)

            # Checks if target rotation is valid at the target column
            if game_copy.intersects():
                continue

            game_copy.hard_drop(color)

            score.append(evaluate_bcts(weight_vector, game_copy))
            candidate_moves.append([col, rotation])

    if len(candidate_moves) > 0:
        best_move = score.index(min(score))
        return candidate_moves[best_move]

    # If no valid moves are found, return invalid move since the game is over
    return [100, 0]

#simule une partie
def simulation(weight_vector):
    """
    Simulates a Tetris game with the given weight vector W for its evaluation function.
    returns the final score of the game.
    """

    game = Tetris(20, 10)
    while game.state != "gameover":

        fig_type = random.randint(0, 6)
        color = 1

        # Evaluates all possible columns and rotations for the current Tetromino
        col, rotation = evaluate_best_move(weight_vector, game.field, fig_type, color)

        # Attempt to place the Tetromino in the best column and rotation
        game.new_figure(fig_type, col, 0, rotation)
        # evaluate_best_move may return invalid moves
        if game.intersects():
            game.state = "gameover"
        else:
            game.hard_drop(color)

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

        fig_type = random.randint(0, 6)

        color = 1

        col, rotation = evaluate_best_move(weight_vector, game.field, fig_type, color)

        game.new_figure(fig_type, col, 0, rotation)

        if game.intersects():
            break

        game.hard_drop(color)

        if move_counter > 0 and move_counter % sample_freq == 0:
            # 200 binary features for the grid (20x10 flattened)
            samples.append(game.field.flatten())

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

            fig_type = random.randint(0, 6)
            color = random.randint(1, 4)

            col, rotation = evaluate_best_move(weight_vector, game.field, fig_type, color)

            game.new_figure(fig_type, col, 0, rotation)

            if game.intersects():
                break

            game.hard_drop(color)

            fig, ax = plt.subplots()
            ax.set_title(str(game.score))
            ax.matshow(game.field, cmap='Reds')
            fig.canvas.draw()
            image = imageio.core.asarray(fig.canvas.renderer.buffer_rgba())
            writer.append_data(image)
            plt.close(fig)
