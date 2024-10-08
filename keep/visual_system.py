import pygame
import numpy as np
from config_manager import ConfigManager
from creature import Creature
from typing import Dict
import time
from timer import Timer
from queue import Empty
from log import get_logger
from collections import deque
from numpy_position_buffer import NumpyPositionBuffer 

logger = get_logger()

# class NumpyPositionBuffer:
#     def __init__(self, target_size=10, min_size=5, max_size=20):
#         self.buffer = deque(maxlen=max_size)
#         self.target_size = target_size
#         self.min_size = min_size
#         self.max_size = max_size
#         self.current_positions = {}
#         self.next_positions = {}
#         self.overflow_count = 0

#     def initialize(self, initial_positions):
#         self.current_positions = initial_positions
#         self.next_positions = initial_positions.copy()
#         self.buffer.clear()
#         self.buffer.append(self.current_positions)

#     def update(self, new_positions, interpolation_steps):
#         self.current_positions = self.next_positions
#         self.next_positions = new_positions
#         interpolated_frames = self._interpolate_vectorized(interpolation_steps)
#         self.add_frames(interpolated_frames)
#         self._adjust_buffer_size()
#         return self.get_stats()

#     def _interpolate_vectorized(self, steps):
#         agent_ids = list(self.current_positions.keys())
#         current_array = np.array([self.current_positions[aid] for aid in agent_ids])
#         next_array = np.array([self.next_positions[aid] for aid in agent_ids])
        
#         t_values = np.linspace(0, 1, steps+1)[1:]
#         interpolated_arrays = current_array[np.newaxis, :, :] + \
#                               (next_array - current_array)[np.newaxis, :, :] * t_values[:, np.newaxis, np.newaxis]
        
#         return [{aid: interpolated_arrays[i, j] for j, aid in enumerate(agent_ids)} 
#                 for i in range(steps)]

#     def add_frames(self, frames):
#         for frame in frames:
#             if len(self.buffer) >= self.max_size:
#                 self.overflow_count += 1
#                 self.buffer.popleft()
#             self.buffer.append(frame)

#     def get_next_position(self):
#         return self.buffer.popleft() if self.buffer else self.current_positions

#     def _adjust_buffer_size(self):
#         current_size = len(self.buffer)
#         if current_size < self.target_size - 2:
#             new_size = min(current_size + 2, self.max_size)
#         elif current_size > self.target_size + 2:
#             new_size = max(current_size - 2, self.min_size)
#         else:
#             return

#         self.buffer = deque(list(self.buffer)[-new_size:], maxlen=new_size)

#     def get_stats(self):
#         return {
#             "current_size": len(self.buffer),
#             "target_size": self.target_size,
#             "max_size": self.max_size,
#             "overflow_count": self.overflow_count
#         }

class VisualSystem:
    def __init__(self, queues):
        logger.info("Initializing VisualSystem")
        self.config_manager = ConfigManager()

        # for screen
        pygame.init()
        self.clock = pygame.time.Clock()
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
        self.creatures: Dict[int, Creature] = {}
        
        # queue
        self._eco_to_visual_render = queues['eco_to_visual_render']
        self._eco_to_visual_init = queues['eco_to_visual_init']
        self._eco_to_visual = queues['eco_to_visual']
        
        # フレーム補間
        self.position_buffer = NumpyPositionBuffer(target_size=30, min_size=5, max_size=40)
        self.frame_times = deque(maxlen=60)  # Store last 60 frame times
        self.last_physics_update_time = time.time()
        self.physics_update_interval = 0.1  # Initial estimate, will be updated dynamically
        self.physics_update_count = 0
        self.last_physics_update_count_time = time.time()# Assume 30 FPS for physics updates, adjust as needed

        self.timer = Timer('Visual System ')
        self.last_buffer_print_time = time.time()

    def initialize(self):
        logger.info("VisualSystem: Waiting for initialization data...")
        while True:
            try:
                init_data = self._eco_to_visual_init.get(timeout=0.1)
                break
            except Empty:
                logger.warning("VisualSystem: No initialization data received, retrying...")
                continue
        _positions = init_data['positions']
        _agent_ids = init_data['agent_ids']
        _species = init_data['species']
        self.current_agent_count = init_data['current_agent_count']

        logger.debug(f"Received positions: {_positions[:5]}...")
        logger.debug(f"Received agent_ids: {_agent_ids[:5]}...")
        logger.debug(f"Received species: {_species[:5]}...")

       # Initialize the PositionBuffer with the initial positions
        initial_positions = {agent_id: pygame.Vector2(pos[0], pos[1]) 
                             for agent_id, pos in zip(_agent_ids, _positions)}
        self.position_buffer.initialize(initial_positions)
        
        # Create creatures and initialize their positions
        for i in range(self.current_agent_count):
            try:
                x, y = _positions[i]
                species = _species[i]
                agent_id = _agent_ids[i]
                self.create_creature(agent_id, species, x, y)
            except Exception as e:
                logger.error(f"Error creating creature {i}: {e}")

        # Add the initial frame to the buffer
        self.position_buffer.add_frames([initial_positions])

        logger.info(f"VisualSystem initialized with {self.current_agent_count} creatures")
        logger.debug(f"Initial buffer size: {len(self.position_buffer.buffer)}")
        self.initialized = True
        
    def create_creature(self, agent_id: int, species: int, x: float, y: float):
        creature = Creature(species, pygame.Vector2(x, y))
        self.creatures[agent_id] = creature
        self.all_sprites.add(creature)
        logger.debug(f"Created creature: agent_id={agent_id}, species={species}, position=({x}, {y})")
        
    def remove_creature(self, agent_id):
        if agent_id in self.creatures:
            self.all_sprites.remove(self.creatures[agent_id])
            del self.creatures[agent_id]
            logger.debug(f"Removed creature: agent_id={agent_id}")
        else:
            logger.warning(f"Attempted to remove non-existent creature: agent_id={agent_id}")

    def update(self):
        self.timer.start()
        self.process_queue()
        self.update_buffer()
        self.update_creatures()
        self.draw()
        self.print_buffer_size()
        
        # Measure and update frame time
        frame_time = self.timer.calculate_time()
        self.frame_times.append(frame_time)
        # 目標フレーム時間を計算
        target_frame_time = 1.0 / self.target_fps
        
        # 残り時間を計算し、必要に応じてスリープ
        remaining_time = max(0, target_frame_time - frame_time)
        if remaining_time > 0:
            time.sleep(remaining_time)
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

    def update_buffer(self):
        try:
            render_data = self._eco_to_visual_render.get_nowait()
            positions = render_data['positions']
            agent_ids = render_data['agent_ids']
            self.current_agent_count = render_data['current_agent_count']

            new_positions = {agent_ids[i]: positions[i] for i in range(self.current_agent_count)}
            current_time = time.time()
            frame_time = self.timer.calculate_time()
            self.position_buffer.add_frame_time(frame_time)

            interpolated_frames, buffer_stats, interpolation_steps = self.position_buffer.update(new_positions, current_time, self.target_fps)
            
            # 補間されたフレームをバッファに追加
            for frame in interpolated_frames:
                self.position_buffer.buffer.append(frame)

            logger.info(f"Buffer stats: {buffer_stats}, Interpolation steps: {interpolation_steps}")

        except Empty:
            pass    

    def update_creatures(self):
        if self.position_buffer.buffer:
            next_position = self.position_buffer.buffer.popleft()
            for agent_id, position in next_position.items():
                if agent_id in self.creatures:
                    pygame_pos = pygame.Vector2(position[0], position[1])
                    self.creatures[agent_id].update(pygame_pos)

    def draw(self):
        self.world_surface.fill(self.background_color)
        self.all_sprites.draw(self.world_surface)
        rect = self.world_surface.get_rect(center=(self.world_width//2, self.world_height//2))
        self.screen.fill(self.background_color)
        self.screen.blit(self.world_surface, rect)
        
        pygame.display.flip()
        logger.debug("Frame rendered")

    def _handle_agent_added(self, data):
        agent_id = data['agent_id']
        species = data['species']
        position = pygame.Vector2(data['position'])
        self.create_creature(agent_id, species, position.x, position.y)
        self.current_agent_count += 1
        logger.debug(f"Agent {agent_id} added. Total agents: {self.current_agent_count}")

    def _handle_agent_removed(self, data):
        agent_id = data['agent_id']
        if agent_id in self.creatures:
            self.remove_creature(agent_id)
            self.current_agent_count -= 1
            logger.debug(f"Agent {agent_id} removed. Total agents: {self.current_agent_count}")
        else:
            logger.warning(f"Attempted to remove non-existent agent {agent_id}")

    def print_buffer_size(self):
        current_time = time.time()
        if current_time - self.last_buffer_print_time >= 1.0:
            buffer_stats = self.position_buffer.get_stats()
            logger.info(f"Buffer stats: {buffer_stats}")
            logger.info(f"Visual FPS: {self.clock.get_fps():.2f}")
            logger.info(f"Physics update rate: {1.0/self.physics_update_interval:.2f} FPS")
            self.last_buffer_print_time = current_time

    def cleanup(self):
        pygame.quit()
        logger.info("VisualSystem cleaned up")