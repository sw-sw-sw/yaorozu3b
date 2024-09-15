import pygame
import numpy as np
from config_manager import ConfigManager
from creature import Creature
from typing import Dict
import time
from queue import Empty

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
        print("VisualSystem: Waiting for initialization data...")
        while True:
            try:
                init_data = self._eco_to_visual_init.get(timeout=0.1)
                break
            except Empty:
                continue
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
        self.update_creatures()
        self.draw()
        
        
    def process_queue(self):
        while True:
            try:
                update_data = self._eco_to_visual.get_nowait()
                action = update_data.get('action')

                if action == 'add':
                    self._handle_agent_added(update_data)
                elif action == 'remove':
                    self._handle_agent_removed(update_data)
            except Empty:
                break
    def update_creatures(self):
        try:
            render_data = self._eco_to_visual_render.get_nowait()
            positions = render_data['positions']
            agent_ids = render_data['agent_ids']
            self.current_agent_count = render_data['current_agent_count']

            for i in range(self.current_agent_count):
                agent_id = agent_ids[i]
                if agent_id in self.creatures:
                    self.creatures[agent_id].update(pygame.Vector2(positions[i][0], positions[i][1]))
        except Empty:
            pass
                    
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
        self.create_creature(agent_id, species, position.x, position.y)
        self.current_agent_count += 1
        print(f"VisualSystem: Agent {agent_id} added. Total agents: {self.current_agent_count}")

    def _handle_agent_removed(self, data):
        agent_id = data['agent_id']
        if agent_id in self.creatures:
            self.remove_creature(agent_id)
            self.current_agent_count -= 1
            print(f"VisualSystem: Agent {agent_id} removed. Total agents: {self.current_agent_count}")
        else:
            print(f"VisualSystem: Attempted to remove non-existent agent {agent_id}")

    def cleanup(self):
        pygame.quit()
        