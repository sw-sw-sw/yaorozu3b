import pygame
import numpy as np
from config_manager import ConfigManager
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
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')

        
        self._box2d_to_visual_render = queues['box2d_to_visual_render']
        self._eco_to_visual_creatures = queues['eco_to_visual_creatures']
        
        self.screen = pygame.display.set_mode((self.world_width, self.world_height))
        self.creatures: Dict[int, Creature] = {}
        self.all_sprites = pygame.sprite.Group()
        
        # Display settings
        self.world_surface = pygame.Surface((self.world_width, self.world_height))
        self.rotation_enabled = False
        
        self.clock = pygame.time.Clock()
        self.current_agent_count = 0


    def initialize(self):
        # Wait for initialization data from Ecosystem
        while self._eco_to_visual_creatures.empty():
            time.sleep(0.1)
        
        init_data = self._eco_to_visual_creatures.get()
        
        # Create initial creatures
        positions = init_data['positions']
        agent_ids = init_data['agent_ids']
        agent_species = init_data['agent_species']
        self.current_agent_count = init_data['current_agent_count']
        
        for i in range(self.current_agent_count):
            x, y = positions[i]
            species = agent_species[i]
            agent_id = agent_ids[i]
            self.create_creature(agent_id, species, x, y)


        
        print(f"VisualSystem initialized with {self.current_agent_count} creatures")

    def create_creature(self, agent_id: int, agent_species: int, x: float, y: float):
        creature = Creature(agent_species, pygame.Vector2(x, y))
        self.creatures[agent_id] = creature
        self.all_sprites.add(creature)
        
    def remove_creature(self, agent_id):
        if agent_id in self.creatures:
            self.all_sprites.remove(self.creatures[agent_id])
            del self.creatures[agent_id]

    # ------------------ Main update ---------------------

    def update(self):
        self.process_creature_queue()
        if not self._box2d_to_visual_render.empty():
            render_data = self._box2d_to_visual_render.get()
            self.update_creatures(render_data)
        self.draw() 
        
    def process_creature_queue(self):
        pass        
        
    def update_creatures(self, render_data):
        positions = render_data['positions']
        for i, agent_id in enumerate(render_data['agent_ids']):
            if agent_id in self.creatures:
                self.creatures[agent_id].update(pygame.Vector2(positions[i][0], positions[i][1]))
                
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
        