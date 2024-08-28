import pygame
from config import *
from timer import Timer

class VisualSystem:
    def __init__(self, queues, running):
        pygame.init()
        self._rendering_queue = queues['rendering_queue']
        self.screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
        self.running = running
        self.timer = Timer("Render ")

    def draw(self, positions):
        self.screen.fill(BACKGROUND_COLOR)
        for pos in positions:
            pygame.draw.circle(self.screen, AGENT_COLOR, (int(pos[0]), int(pos[1])), AGENT_RADIUS, 1)

        pygame.display.flip()

    def render(self):
        if not self._rendering_queue.empty():
            positions = self._rendering_queue.get()
            self.draw(positions)
            
    def run(self):
        pygame.init()
        clock = pygame.time.Clock()

        while self.running.value:
            self.timer.start()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running.value = False
                    return
                
            self.render()

            self.timer.print_fps(1)        
            clock.tick(RENDER_FPS)
        pygame.quit()