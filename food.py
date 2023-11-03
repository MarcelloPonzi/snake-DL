
from random import randint
import pygame


class Food:
    def __init__(self):
        self.x = 240
        self.y = 200
        self.image = pygame.image.load('img/food2.png')

    @property
    def position(self):
        return [self.x, self.y]
    
    @position.setter
    def position(self, position: list[int, int]):
        self.x, self.y = position

    @staticmethod
    def random(size: float):
        x_rand = randint(20, size - 40)
        return x_rand - x_rand % 20
    
    def randomize(self, size: tuple[float, float], occupiedPoints: list[list]):
        while True:
            self.position = [Food.random(size) for size in size]
            if self.position not in occupiedPoints:
                break