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
        
        self._eco_to_visual_render = queues['eco_to_visual_render']
        self._eco_to_visual_init = queues['eco_to_visual_init']
        self._eco_to_visual = queues['eco_to_visual']
        
        self.screen = pygame.display.set_mode((self.world_width, self.world_height))
        self.creatures: Dict[int, Creature] = {}
        self.all_sprites = pygame.sprite.Group()
        
        self.world_surface = pygame.Surface((self.world_width, self.world_height))
        self.rotation_enabled = False
        
        self.clock = pygame.time.Clock()
        self.current_agent_count = 0


    def initialize(self):
        # Wait for initialization data from Ecosystem
        while self._eco_to_visual_init.empty():
            time.sleep(0.1)
        
        init_data = self._eco_to_visual_init.get()
        
        # Create initial creatures
        _positions = init_data['positions']
        _agent_ids = init_data['agent_ids']
        _species = init_data['species']
        self.current_agent_count = init_data['current_agent_count']
        
        for i in range(self.current_agent_count):
            x, y = _positions[i]
            species = _species[i]
            agent_id = _agent_ids[i]
            self.create_creature(agent_id, species, x, y)

        print(f"VisualSystem initialized with {self.current_agent_count} creatures")

    def create_creature(self, agent_id: int, species: int, x: float, y: float):
        creature = Creature(species, pygame.Vector2(x, y))
        self.creatures[agent_id] = creature
        self.all_sprites.add(creature)
        
    def remove_creature(self, agent_id):
        if agent_id in self.creatures:
            self.all_sprites.remove(self.creatures[agent_id])
            del self.creatures[agent_id]

    # ------------------ Main update ---------------------

    def update(self):
        self.process_queue()
        if not self._eco_to_visual_render.empty():
            render_data = self._eco_to_visual_render.get()
            self.update_creatures(render_data)
        self.draw() 
        
    def process_queue(self):
        while not self._eco_to_visual.empty():
            update_data = self._eco_to_visual.get()
            action = update_data.get('action')

            if action == 'add':
                self._handle_agent_added(update_data)
            elif action == 'remove':
                self._handle_agent_removed(update_data)  
        
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

    def _handle_agent_added(self, data):
        agent_id = data['agent_id']
        species = data['species']
        position = pygame.Vector2(data['position'])
        new_creature = Creature(species, position)
        self.creatures[agent_id] = new_creature
        self.all_sprites.add(new_creature)

        print(f"VisualSystem: Agent {agent_id} added")

    def _handle_agent_removed(self, data):
        agent_id = data['agent_id']
        if agent_id in self.creatures:
            self.all_sprites.remove(self.creatures[agent_id])
            del self.creatures[agent_id]
            
            print(f"VisualSystem: Agent {agent_id} removed")

    def cleanup(self):
        pygame.quit()
        