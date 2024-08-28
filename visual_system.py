import pygame
import numpy as np
from config import *
from timer import Timer
from creature import Creature
from typing import Dict

class VisualSystem:
    def __init__(self, queues, running):
        pygame.init()
        self._rendering_queue = queues['rendering_queue']
        self._eco_to_visual_creatures = queues['eco_to_visual_creatures']
        self.screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
        self.running = running
        self.timer = Timer("Render ")
        self.creatures: Dict[int, Creature] = {}
        self.rotation_angle = 0
        self.rotation_speed = 0.1  # 1フレームあたりの回転角度（度）
        self.world_surface = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))

    def create_creature(self, agent_id: int, agent_species: int, x: float, y: float) -> Creature:
        creature = Creature(agent_species, pygame.Vector2(x, y))
        self.creatures[agent_id] = creature
        return creature

    def update_creature(self, agent_id: int, x: float, y: float):
        if agent_id in self.creatures:
            self.creatures[agent_id].update(pygame.Vector2(x, y))

    def draw(self):
        self.world_surface.fill(BACKGROUND_COLOR)
        
        for creature in self.creatures.values():
            creature.draw(self.world_surface)

        # 回転した世界を画面に描画
        rotated_surface = pygame.transform.rotate(self.world_surface, self.rotation_angle)
        rotated_rect = rotated_surface.get_rect(center=(WORLD_WIDTH//2, WORLD_HEIGHT//2))
        self.screen.fill(BACKGROUND_COLOR)
        self.screen.blit(rotated_surface, rotated_rect)
        
        pygame.display.flip()

        # 回転角度を更新
        self.rotation_angle += self.rotation_speed
        if self.rotation_angle >= 360:
            self.rotation_angle -= 360

    def process_creature_queue(self):
        while not self._eco_to_visual_creatures.empty():
            creature_info = self._eco_to_visual_creatures.get()
            if creature_info['action'] == 'create':
                self.create_creature(
                    creature_info['agent_id'],
                    creature_info['agent_species'],
                    creature_info['x'],
                    creature_info['y']
                )

    def render(self):
        self.process_creature_queue()
        if not self._rendering_queue.empty():
            positions = self._rendering_queue.get()
            for agent_id, pos in enumerate(positions):
                self.update_creature(agent_id, pos[0], pos[1])
            self.draw()

    def run(self):
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