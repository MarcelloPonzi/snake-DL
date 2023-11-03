
import pygame
import numpy as np
from food import Food
from game import Game

class Player(object):
    def __init__(self, size: list[int]):
        x = 0.45 * size[0]
        y = 0.5 * size[1]
        self.x = x - x % 20
        self.y = y - y % 20
        self.tail = [self.position]
        self.eaten = False
        self.image = pygame.image.load('img/snakeBody.png')
        self.x_change = 20
        self.y_change = 0
    
    @property
    def position(self):
        return [self.x, self.y]
    
    @property
    def food(self):
        return len(self.tail)

    def update_position(self):
        if self.tail[-1] == self.position:
            return
        self.tail = self.tail[1:] + [self.position]

    def do_move(self, move, game: Game, food: Food):
        move_array = [self.x_change, self.y_change]

        if self.eaten:
            self.tail.append(self.position)
            self.eaten = False
        if np.array_equal(move, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(move, [0, 1, 0]) and self.y_change == 0:  # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(move, [0, 1, 0]) and self.x_change == 0:  # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(move, [0, 0, 1]) and self.y_change == 0:  # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(move, [0, 0, 1]) and self.x_change == 0:  # left - going vertical
            move_array = [self.y_change, 0]
        self.x_change, self.y_change = move_array
        self.x, self.y = self.x + self.x_change, self.y + self.y_change

        game.crash = self.isGameOver(game)
        self.eat(food, game)
        self.update_position()
    
    def isGameOver(self, game: Game):
        def isOut(i: int): return self.position[i] < 20 or self.position[i] > game.size[i] - 40
        return any((isOut(i) for i in range(2))) or self.position in self.tail[:-1]

    def eat(self, food: Food, game: Game):
        if self.position != food.position: return
        food.randomize(game.size, self.tail)
        self.eaten = True
        game.score += 1

    def display_player(self, game: Game):
        for i in range(self.food):
            game.gameDisplay.blit(self.image, self.tail[-(i + 1)])