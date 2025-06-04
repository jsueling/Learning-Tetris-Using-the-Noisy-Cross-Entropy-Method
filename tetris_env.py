import random
import copy

import numpy as np
import matplotlib.pyplot as plt
import imageio

from BCTS import evaluate_BCTS

class Figure:
    x = 0
    y = 0
#liste des 6 différentes figures et leur rotation
    figures = [ 
        [[0, 4, 8, 12], [0, 1, 2, 3], [0, 4, 8, 12], [0, 1, 2, 3]], # I
        [[0, 1, 5, 6], [1, 4, 5, 8], [0, 1, 5, 6], [1, 4, 5, 8]], # Z
        [[4, 5, 1, 2], [0, 4, 5, 9], [4, 5, 1, 2], [0, 4, 5, 9]], # S
        [[1, 0, 4, 8], [0, 4, 5, 6], [1, 5, 9, 8], [0, 1, 2, 6]], # J
        [[0, 1, 5, 9], [4, 0, 1, 2], [0, 4, 8, 9], [4, 5, 6, 2]], # L
        [[1, 4, 5, 6], [1, 4, 5, 9], [0, 1, 2, 5], [0, 4, 8, 5]], # T
        [[0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5]], # O
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
    def __init__(self, height, width): #initialisation du jeu 

        self.figure = None
        self.height = height
        self.width = width
        self.field = np.zeros((height, width), dtype=int)
        self.score = 0
        self.state = "start"
        # number of lines broken by the last Tetromino placement
        self.broken_lines = 0

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
        """Check for completed rows, remove them by shifting down all rows above."""
        broken_lines = 0
        for i in range(1, self.height):
            zeros = 0
            for j in range(self.width):
                if self.field[i][j] == 0:
                    zeros += 1
            if zeros == 0:
                broken_lines += 1
                for i1 in range(i, 1, -1):
                    for j in range(self.width):
                        self.field[i1][j] = self.field[i1 - 1][j]
        self.score += broken_lines #** 2 -- remove Tetris line-clear bonus for now
        self.broken_lines = broken_lines

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

#Features du jeu 
#retourne la taille des 10 colonnes du jeu 
def column_height(field): #from top to bottom
    """Returns the height of each column in the grid as a list."""
    h = []
    for j in range(10):
        column = [field[i][j] for i in range(20)]
        height = 0 # height is a pointer counting contiguous empty cells

        while height < 20 and column[height] == 0:
            height += 1

        h.append(20 - height)

    return h

#retourne la taille maximale des colonnes du jeu
def maximum_height(field):
    return(max(column_height(field)))

#retourne la différence en valeur absolue de la taille d'une colonne avec celle de sa voisine 
def column_difference(field):# absolute difference between adjacent columns
    df=[]
    h=column_height(field)

    for j in range(9):
        df.append(abs(h[j+1]-h[j]))
    
    return(df)
#compte le nombre de troux inaccessibles du jeu 
def holes(field):
    L=0
    h=column_height(field)

    for j in range(10):
        for i in range(20-h[j],20):
            if field[i][j]==0:
                L+=1
    
    return(L)


#Evalue la configuration de la grille en pondérant les features par le vecteur W de taille 21
def evaluate_Bertsekas(W, field):
    """Evaluate the Tetris grid using Bertsekas and Tsitsiklis' feature set."""
    #W=[w1, ..., w21] vector of parameters to tune 

    h=column_height(field)
    dh=column_difference(field)
    L=holes(field)
    H=maximum_height(field)
    
    S1,S2,S3,S4=0,0,0,0

    for k in range (len(h)):
        S1+=h[k]*W[k]
    
    for k in range (len(dh)):
        S2+=dh[k]*W[10+k]

    S3=W[19]*L

    S4=W[20]*H

    return(S1+S2+S3+S4)

def evaluate_best_move(W, field, fig_type, color):
    """Evaluates the best Tetromino placement"""
    L = []
    score = []
    for k in range(4):
        for col in range(10):

            game_copy = Tetris(20, 10)

            game_copy.field = copy.deepcopy(field)

            # Try to place Tetromino at (col, 0) with rotation k
            game_copy.new_figure(fig_type, col, 0, k)

            if game_copy.intersects():
                continue

            game_copy.hard_drop(color)

            score.append(evaluate_BCTS(W, game_copy))
            L.append([col, k])

    if len(L) > 0:
        best_move = score.index(min(score))
        return L[best_move]

    # If no valid moves are found, return a default move
    return [0, 0]

#simule une partie
def simulation(W):
    """
    Simulates a Tetris game with the given weight vector W for its evaluation function.
    """

    game = Tetris(20, 10)
    while game.state != "gameover":

        fig_type = random.randint(0, 6)
        color = 1

        # Evaluates all possible columns and rotations for the current Tetromino
        col, rot = evaluate_best_move(W, game.field, fig_type, color)

        # Attempt to place the Tetromino in the best column and rotation
        game.new_figure(fig_type, col, 0, rot)

        # Check since evaluate_best_move may return (col, rot) that is already occupied
        if game.intersects():
            game.state = "gameover"
        else:
            game.hard_drop(color)

    return game.score

def simulation_gif(W, num_moves=100): #Pas encore optimisé pour les pièces qui arrivent en haut
    """
    Simulates a Tetris game with the given weight vector W for its evaluation function
    and saves the frames as a GIF
    """

    with imageio.get_writer('tetris.gif', mode='I') as writer:

        game = Tetris(20, 10)

        for _ in range(num_moves):

            fig_type = random.randint(0, 6)
            color = random.randint(1, 4)

            col, rot = evaluate_best_move(W, game.field, fig_type, color)

            game.new_figure(fig_type, col, 0, rot)

            if game.intersects():
                break

            game.hard_drop(color)

            fig, ax = plt.subplots()
            ax.set_title(str(game.score))
            ax.matshow(game.field, cmap='Blues')
            fig.canvas.draw()
            image = imageio.core.asarray(fig.canvas.renderer.buffer_rgba())
            writer.append_data(image)
            plt.close(fig)
