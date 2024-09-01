import pygame
import numpy as np
from config import *
from timer import Timer
from creature import Creature
from typing import Dict
import time

class VisualSystem:
    def __init__(self, queues, running):
        pygame.init()
        self._rendering_queue = queues['rendering_queue']
        self._eco_to_visual_creatures = queues['eco_to_visual_creatures']
        self.screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
        self.running = running
        self.timer = Timer("Render ")
        self.creatures: Dict[int, Creature] = {}
        self.all_sprites = pygame.sprite.Group()
        
        # Display settings
        self.rotation_angle = 0
        self.rotation_speed = 0.0
        self.world_surface = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))
        self.zoom_factor = 1.0
        self.rotation_enabled = False
        self.zoom_enabled = False
        
        self.clock = pygame.time.Clock()


    def create_creature(self, agent_id: int, agent_species: int, x: float, y: float) -> Creature:
        creature = Creature(agent_species, pygame.Vector2(x, y))
        self.creatures[agent_id] = creature
        self.all_sprites.add(creature)
        return creature

    def update_creature(self, agent_id: int, x: float, y: float):
        if agent_id in self.creatures:
            self.creatures[agent_id].update(pygame.Vector2(x, y))

    def draw(self):
        self.world_surface.fill(BACKGROUND_COLOR)
        self.all_sprites.draw(self.world_surface)

        surface_to_draw = self.world_surface
        
        if self.rotation_enabled:
            surface_to_draw = pygame.transform.rotate(surface_to_draw, self.rotation_angle)

        if self.zoom_enabled:
            surface_to_draw = pygame.transform.smoothscale(surface_to_draw, 
                                (int(surface_to_draw.get_width() * self.zoom_factor),
                                int(surface_to_draw.get_height() * self.zoom_factor)))
            
        rect = surface_to_draw.get_rect(center=(WORLD_WIDTH//2, WORLD_HEIGHT//2))
        self.screen.fill(BACKGROUND_COLOR)
        self.screen.blit(surface_to_draw, rect)

        pygame.display.flip()

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
            time.sleep(0.001)
            self.draw()
            
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running.value = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.zoom_factor *= 1.1  # 拡大
                elif event.key == pygame.K_MINUS:
                    self.zoom_factor /= 1.1  # 縮小
        return True
    
    def update(self):
        self.timer.start()
        if not self.handle_events():
            return False
        self.render()
        self.timer.print_fps(5)        
        self.clock.tick(RENDER_FPS)
        return True

    def cleanup(self):
        pygame.quit()
        