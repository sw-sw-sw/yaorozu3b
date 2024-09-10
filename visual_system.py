import pygame
import numpy as np
from config_manager import ConfigManager
from timer import Timer
from creature import Creature
from typing import Dict
import time

class VisualSystem:
    def __init__(self, queues):
        pygame.init()
        self.config_manager = ConfigManager()
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')
        self.background_color = self.config_manager.get_trait_value_as_tuple('BACKGROUND_COLOR')
        self.render_fps = self.config_manager.get_trait_value('RENDER_FPS')
        
        self._box2d_to_visual_render = queues['box2d_to_visual_render']
        self._eco_to_visual_creatures = queues['eco_to_visual_creatures']
        
        self.screen = pygame.display.set_mode((self.world_width, self.world_height))
        self.timer = Timer("Render ")
        self.creatures: Dict[int, Creature] = {}
        self.all_sprites = pygame.sprite.Group()
        
        # Display settings
        self.world_surface = pygame.Surface((self.world_width, self.world_height))
        self.rotation_enabled = False
        
        self.clock = pygame.time.Clock()

    def initialize(self):
        # Wait for initialization data from Ecosystem
        while self._eco_to_visual_creatures.empty():
            time.sleep(0.1)
        
        init_data = self._eco_to_visual_creatures.get()
        
        # Create initial creatures
        positions = init_data['positions']
        agent_ids = init_data['agent_ids']
        agent_species = init_data['species']
        
        for i, agent_id in enumerate(agent_ids):
            x, y = positions[i]
            species = agent_species[i]
            creature = self.create_creature(agent_id, species, x, y)
            self.creatures[agent_id] = creature  # Ad
        
        print(f"VisualSystem initialized with {len(self.creatures)} creatures")

    def create_creature(self, agent_id: int, agent_species: int, x: float, y: float) -> Creature:
        creature = Creature(agent_species, pygame.Vector2(x, y))
        self.creatures[agent_id] = creature
        self.all_sprites.add(creature)
        return creature

    # ------------------ Main routine ---------------------

    def run(self):
        self.timer.start()
        self.update()
        self.timer.print_fps(5)        
        self.clock.tick(self.render_fps)
        return True

    # ------------------ update ---------------------

    def update(self):
        while not self._box2d_to_visual_render.empty():
            render_data = self._box2d_to_visual_render.get()
            positions = render_data['positions']
            agent_ids = render_data['agent_ids']
            self.update_creatures(agent_ids, positions)
        self.draw()        

    def update_creatures(self, agent_ids, positions):
        for agent_id, pos in zip(agent_ids, positions):    
            self.creatures[agent_id].update(pygame.Vector2(pos[0], pos[1]))

    def draw(self):
        self.world_surface.fill(self.background_color)
        self.all_sprites.draw(self.world_surface)
        surface_to_draw = self.world_surface
        rect = surface_to_draw.get_rect(center=(self.world_width//2, self.world_height//2))
        self.screen.fill(self.background_color)
        self.screen.blit(surface_to_draw, rect)
        pygame.display.flip()


    def cleanup(self):
        pygame.quit()
        