
import pygame

class Game:    
    def __init__(self, game_width: int, game_height: int):
        pygame.display.set_caption('SnakeGen')
        self.game_width = game_width
        self.game_height = game_height
        self.gameDisplay = pygame.display.set_mode((game_width, game_height + 60))
        self.bg = pygame.image.load("img/background.png")
        self.crash = False
        self.score = 0

    @property
    def size(self): 
        return [self.game_width, self.game_height]
