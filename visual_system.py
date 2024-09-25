import pygame
from pygame import Vector2
import numpy as np
from config_manager import ConfigManager
from creature import Creature
from typing import Dict, List
import time
from timer import Timer
from queue import Empty
from log import get_logger


class VisualSystem:
    def __init__(self, queues):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing VisualSystem")
        self.config_manager = ConfigManager()

        # for screen
        pygame.init()
        # self.clock = pygame.time.Clock()
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')
        self.background_color = self.config_manager.get_trait_value_as_tuple('BACKGROUND_COLOR')
        self.screen = pygame.display.set_mode((self.world_width, self.world_height))
        self.target_fps = self.config_manager.get_trait_value('RENDER_FPS')
        self.world_surface = pygame.Surface((self.world_width, self.world_height))
        self.all_sprites = pygame.sprite.Group()
        
        # main property
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.current_agent_count = 0
        self.positions = np.zeros((self.max_agents_num, 2), dtype=np.float32)
        self.agent_ids = np.full(self.max_agents_num, -1, dtype=np.int32)
        self.species = np.zeros(self.max_agents_num, dtype=np.int32)
        self.creatures: Dict[int, Creature] = {}
        # queue
        self._box2d_to_visual_render = queues['box2d_to_visual_render']
        self._eco_to_visual_init = queues['eco_to_visual_init']
        self._eco_to_visual = queues['eco_to_visual']
        
        self.timer = Timer('Visual System ')
        self.stats1 = Timer('Stats1 ')
        
    def initialize(self):
        self.logger.info("VisualSystem: Waiting for initialization data...")
        while True:
            try:
                init_data = self._eco_to_visual_init.get(timeout=0.1)
                break
            except Empty:
                self.logger.warning("VisualSystem: No initialization data received, retrying...")
                continue
        self.current_agent_count = init_data['current_agent_count']
        self.positions = init_data['positions']
        self.agent_ids = init_data['agent_ids']
        self.species = init_data['species']
        self.current_agent_count = init_data['current_agent_count']

        self.logger.debug(f"Received positions: {self.positions[:5]}...")
        self.logger.debug(f"Received agent_ids: {self.agent_ids[:5]}...")
        self.logger.debug(f"Received species: {self.species[:5]}...")
        
        # Create creatures and initialize their positions
        for i in range(self.current_agent_count):
            try:
                x, y = self.positions[i]
                species = self.species[i]
                agent_id = self.agent_ids[i]
                self.create_creature(agent_id, species, x, y)
            except Exception as e:
                self.logger.error(f"Error creating creature {i}: {e}")

        self.logger.info(f"VisualSystem initialized with {self.current_agent_count} creatures")
        self.initialized = True
        
    def create_creature(self, agent_id: int, species: int, x: float, y: float):
        creature = Creature(species, Vector2(x, y))
        self.creatures[agent_id] = creature
        self.all_sprites.add(creature)
        self.logger.debug(f"Created creature: agent_id={agent_id}, species={species}, position=({x}, {y})")
        
    def remove_creature(self, agent_id):
        if agent_id in self.creatures:
            creature = self.creatures[agent_id]
            self.all_sprites.remove(creature)
            del self.creatures[agent_id]
            self.logger.debug(f"Removed creature: agent_id={agent_id}")
        else:
            self.logger.warning(f"Attempted to remove non-existent creature: agent_id={agent_id}")

    def update(self):
        self.process_queue()
        self.update_property()
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

    def update_property(self):
        try:
            render_data = self._box2d_to_visual_render.get_nowait()
            self.positions = render_data['positions']
            self.agent_ids = render_data['agent_ids']
        except Empty:
            pass

    def update_creatures(self):
        for agent_id, position in zip(self.agent_ids, self.positions):
            if agent_id in self.creatures:
                self.creatures[agent_id].update(position)     
           
    def draw(self):
        self.world_surface.fill(self.background_color)
        self.all_sprites.draw(self.world_surface)
        rect = self.world_surface.get_rect(center=(self.world_width//2, self.world_height//2))
        self.screen.fill(self.background_color)
        self.screen.blit(self.world_surface, rect)
        
        pygame.display.flip()
        self.logger.debug("Frame rendered")
            
    def _handle_agent_added(self, data):
        agent_id = data['agent_id']
        species = data['species']
        position = data['position']
        self.current_agent_count = data['current_agent_count']

        self.create_creature(agent_id, species, position[0], position[1])
        self.logger.debug(f"Agent {agent_id} added. Total agents: {self.current_agent_count}")

    def _handle_agent_removed(self, data):
        agent_id = data['agent_id']
        self.current_agent_count = data['current_agent_count']
        if agent_id in self.creatures:
            self.remove_creature(agent_id)
            self.logger.debug(f"Agent {agent_id} removed . Total agents: {self.current_agent_count}")

    def cleanup(self):        
        pygame.quit()
        self.logger.info("VisualSystem cleaned up")
        