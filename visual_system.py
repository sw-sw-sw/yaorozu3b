import pygame
from pygame import Vector2
import numpy as np
from config_manager import ConfigManager
from creature import Creature
from typing import Dict
import time
from timer import Timer
from queue import Empty
from log import get_logger
from collections import deque


        
class NumpyPositionBuffer:
    def __init__(self, max_agents, target_size=10, min_size=5, max_size=20):
        self.logger = get_logger(self.__class__.__name__)
        self.buffer = deque(maxlen=max_size)
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
        self.current_positions = np.zeros((max_agents, 2), dtype=np.float32)
        self.next_positions = np.zeros((max_agents, 2), dtype=np.float32)
        self.agent_ids = np.full(max_agents, -1, dtype=np.int32)
        self.overflow_count = 0
        self.last_agent_count = 0

    def initialize(self, initial_positions, initial_agent_ids):
        self.current_positions[:len(initial_positions)] = initial_positions
        self.next_positions[:len(initial_positions)] = initial_positions
        self.agent_ids[:len(initial_agent_ids)] = initial_agent_ids
        self.buffer.clear()
        self.buffer.append(self.current_positions[:len(initial_positions)])
        self.last_agent_count = len(initial_positions)

    def update(self, new_positions, new_agent_ids, interpolation_steps, current_agent_count):
        if current_agent_count != self.last_agent_count:
            self.logger.warning(f"Agent count mismatch. Expected: {self.last_agent_count}, Got: {current_agent_count}. Resetting buffer.")
            self.current_positions[:current_agent_count] = new_positions
            self.next_positions[:current_agent_count] = new_positions
            self.agent_ids[:current_agent_count] = new_agent_ids
            # self.buffer.clear()
            self.buffer.append(self.current_positions[:current_agent_count])
            self.last_agent_count = current_agent_count
            self.get_stats()

        self.current_positions[:current_agent_count] = self.next_positions[:current_agent_count]
        self.next_positions[:current_agent_count] = new_positions
        self.agent_ids[:current_agent_count] = new_agent_ids

        interpolated_frames = self._interpolate_vectorized(interpolation_steps, current_agent_count)
        self.add_frames(interpolated_frames)
        self._adjust_buffer_size()
        
        self.last_agent_count = current_agent_count
        return self.get_stats()

    def update_with_physics_data(self, positions, agent_ids, current_agent_count, physics_update_interval, avg_frame_time):
        self.current_positions[:current_agent_count] = self.next_positions[:current_agent_count]
        self.next_positions[:current_agent_count] = positions
        self.agent_ids[:current_agent_count] = agent_ids

        base_interpolation_steps = max(1, int(physics_update_interval / avg_frame_time))
        correction_value = 2  # This can be adjusted based on your needs
        interpolation_steps = base_interpolation_steps + correction_value

        current_buffer_size = len(self.buffer)
        if current_buffer_size < self.target_size:
            interpolation_steps += 1
        elif current_buffer_size > self.target_size:
            interpolation_steps = max(1, interpolation_steps - 1)

        interpolated_frames = self._interpolate_vectorized(interpolation_steps, current_agent_count)
        self.add_frames(interpolated_frames)
        
        self.last_agent_count = current_agent_count
        
        self.logger.info(f"Buffer stats: {self.get_stats()}, Interpolation steps: {interpolation_steps}")
        
        return 
    
    def _interpolate_vectorized(self, steps, current_agent_count):
        current_array = self.current_positions[:current_agent_count]
        next_array = self.next_positions[:current_agent_count]
        t_values = np.linspace(0, 1, steps+1)[1:]
        interpolated_arrays = current_array[np.newaxis, :, :] + \
                              (next_array - current_array)[np.newaxis, :, :] * t_values[:, np.newaxis, np.newaxis]
        
        return [interpolated_arrays[i] for i in range(steps)]

    def add_frames(self, frames):
        for frame in frames:
            if len(self.buffer) >= self.max_size:
                self.overflow_count += 1
                self.buffer.popleft()
            self.buffer.append(frame)

    def get_next_position(self):
        return self.buffer.popleft() if self.buffer else self.current_positions[:self.last_agent_count]

    def get_stats(self):
        return {
            "current_size": len(self.buffer),
            "target_size": self.target_size,
            "max_size": self.max_size,
            "overflow_count": self.overflow_count
        }
        
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
        self._eco_to_visual_render = queues['eco_to_visual_render']
        self._eco_to_visual_init = queues['eco_to_visual_init']
        self._eco_to_visual = queues['eco_to_visual']
        
        # フレーム補間
        self.position_buffer = NumpyPositionBuffer(self.max_agents_num, target_size=30, min_size=5, max_size=40)        
        self.frame_times = deque(maxlen=60)  # Store last 60 frame times
        self.last_physics_update_time = time.time()
        self.physics_update_interval = 0.1  # Initial estimate, will be updated dynamically
        self.physics_update_count = 0
        self.last_physics_update_count_time = time.time()# Assume 30 FPS for physics updates, adjust as needed

        self.timer = Timer('Visual System ')
        self.stats1 = Timer('Stats1 ')
        self.last_buffer_print_time = time.time()

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
        self.positions[:self.current_agent_count] = init_data['positions']
        self.agent_ids[:self.current_agent_count] = init_data['agent_ids']
        self.species[:self.current_agent_count] = init_data['species']
        self.current_agent_count = init_data['current_agent_count']

        self.logger.debug(f"Received positions: {self.positions[:5]}...")
        self.logger.debug(f"Received agent_ids: {self.agent_ids[:5]}...")
        self.logger.debug(f"Received species: {self.species[:5]}...")

   
        self.position_buffer.initialize(self.positions, self.agent_ids)
        
        # Create creatures and initialize their positions
        for i in range(self.current_agent_count):
            try:
                x, y = self.positions[i]
                species = self.species[i]
                agent_id = self.agent_ids[i]
                self.create_creature(agent_id, species, x, y)
            except Exception as e:
                self.logger.error(f"Error creating creature {i}: {e}")

        # Add the initial frame to the buffer
        self.position_buffer.add_frames([self.positions])
        self.last_agent_count = self.current_agent_count
        
        self.logger.info(f"VisualSystem initialized with {self.current_agent_count} creatures")
        self.logger.debug(f"Initial buffer size: {len(self.position_buffer.buffer)}")
        self.initialized = True
        
    def create_creature(self, agent_id: int, species: int, x: float, y: float):
        creature = Creature(species, Vector2(x, y))
        self.creatures[agent_id] = creature
        self.all_sprites.add(creature)
        self.logger.debug(f"Created creature: agent_id={agent_id}, species={species}, position=({x}, {y})")
        
    def remove_creature(self, agent_id):
        if agent_id in self.creatures:
            self.all_sprites.remove(self.creatures[agent_id])
            del self.creatures[agent_id]
            self.logger.debug(f"Removed creature: agent_id={agent_id}")
        else:
            self.logger.warning(f"Attempted to remove non-existent creature: agent_id={agent_id}")

    def update(self):
        self.timer.start()
        self.process_queue()
        
        
        self.update_buffer()
        
        # self.stats1.start()
        self.update_creatures()
        # self.stats1.print_lap_time(1)
        self.draw()
        # self.print_buffer_size()
        
        # Measure and update frame time
        frame_time = self.timer.calculate_time()
        self.frame_times.append(frame_time)
        # Adaptive frame rate control
        # self.clock.tick(self.target_fps)

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
            render_data = self._eco_to_visual_render.get()
            positions = render_data['positions']
            agent_ids = render_data['agent_ids']
            self.current_agent_count = render_data['current_agent_count']
            self.positions[:self.current_agent_count] = positions
            self.agent_ids[:self.current_agent_count] = agent_ids
            
            #演算フレームレート(physics_update_interval)のを計算
            current_time = time.time()
            self.physics_update_count += 1
            if current_time - self.last_physics_update_count_time >= 1.0:
                self.physics_update_interval = 1.0 / self.physics_update_count
                self.physics_update_count = 0
                self.last_physics_update_count_time = current_time

            #描画フレームレート(avg_frame_time)の計算
            avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else (1.0 / self.target_fps)
            
            self.position_buffer.update_with_physics_data(
                positions, 
                agent_ids, 
                self.current_agent_count, 
                self.physics_update_interval, 
                avg_frame_time
                )
            
        except Empty:
            pass

    def update_creatures(self):
        next_position = self.position_buffer.get_next_position()
        for agent_id, position in zip(self.agent_ids, next_position):
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
        self.create_creature(agent_id, species, position[0], position[1])
        
        index = self.current_agent_count
        self.positions[index] = np.array(position, dtype=np.float32)
        self.agent_ids[index] = agent_id
        self.species[index] = species
        self.current_agent_count += 1

        self.logger.debug(f"Agent {agent_id} added. Total agents: {self.current_agent_count}")

    def _handle_agent_removed(self, data):
        agent_id = data['agent_id']
        if agent_id in self.creatures:
            self.remove_creature(agent_id)
                        # Update numpy arrays
            index = np.where(self.agent_ids == agent_id)[0][0]
            self.agent_ids[index:-1] = self.agent_ids[index+1:]
            self.species[index:-1] = self.species[index+1:]
            self.agent_ids[self.current_agent_count-1] = -1
            self.species[self.current_agent_count-1] = 0
            self.current_agent_count -= 1
            
            self.logger.debug(f"Agent {agent_id} removed. Total agents: {self.current_agent_count}")
        else:
            self.logger.warning(f"Attempted to remove non-existent agent {agent_id}")
        

    def print_buffer_size(self):
        current_time = time.time()
        if current_time - self.last_buffer_print_time >= 1.0:
            buffer_stats = self.position_buffer.get_stats()
            self.logger.info(f"Buffer stats: {buffer_stats}")
            self.logger.info(f"Visual FPS: {self.clock.get_fps():.2f}")
            self.logger.info(f"Physics update rate: {1.0/self.physics_update_interval:.2f} FPS")
            self.last_buffer_print_time = current_time

    def cleanup(self):        
        pygame.quit()
        self.logger.info("VisualSystem cleaned up")
        