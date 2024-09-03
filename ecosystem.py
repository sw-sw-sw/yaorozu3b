# ecosystem.py
import numpy as np
from creature import Creature
import pygame
from config_manager import ConfigManager
import time

class Ecosystem:
    def __init__(self, queues):
        self.config_manager = ConfigManager()
        self.eco_to_visual_creatures = queues['eco_to_visual_creatures']
        self.eco_to_box2d_creatures = queues['eco_to_box2d_creatures']

        # ConfigManagerから値を取得してプロパティとして設定
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')
        self.initial_velocity_min = self.config_manager.get_trait_value('INITIAL_VELOCITY_MIN')
        self.initial_velocity_max = self.config_manager.get_trait_value('INITIAL_VELOCITY_MAX')

    def initialize_agents(self, shared_memory):
        with shared_memory['lock']:
            np_positions = np.frombuffer(shared_memory['positions'].get_obj(), dtype=np.float32).reshape((self.max_agents_num, 2))
            np_velocities = np.frombuffer(shared_memory['velocities'].get_obj(), dtype=np.float32).reshape((self.max_agents_num, 2))
            np_forces = np.frombuffer(shared_memory['forces'].get_obj(), dtype=np.float32).reshape((self.max_agents_num, 2))
            np_agent_ids = np.frombuffer(shared_memory['agent_ids'].get_obj(), dtype=np.int32)
            np_agent_species = np.frombuffer(shared_memory['agent_species'].get_obj(), dtype=np.int32)

            # 円形領域の中心と半径を定義
            center_x, center_y = self.world_width / 2, self.world_height / 2
            radius = min(self.world_width, self.world_height) / 2

            # 円形領域内にランダムな位置を生成
            angles = np.random.uniform(0, 2 * np.pi, self.max_agents_num)
            radii = np.sqrt(np.random.uniform(0, 1, self.max_agents_num)) * radius  # 均一な分布のために平方根を使用
            
            np_positions[:, 0] = center_x + radii * np.cos(angles)
            np_positions[:, 1] = center_y + radii * np.sin(angles)

            np_velocities[:] = np.random.uniform(self.initial_velocity_min, self.initial_velocity_max, (self.max_agents_num, 2))
            np_forces[:] = np.zeros((self.max_agents_num, 2))
            np_agent_ids[:] = np.arange(self.max_agents_num)
            np_agent_species[:] = np.random.randint(1, 9, self.max_agents_num)

            shared_memory['current_agent_count'].value = self.max_agents_num

            for i in range(self.max_agents_num):
                temp_creature = Creature(np_agent_species[i], pygame.Vector2(np_positions[i][0], np_positions[i][1]))
                radius = temp_creature.get_radius()

                self.eco_to_visual_creatures.put({
                    'action': 'create',
                    'agent_id': int(np_agent_ids[i]),
                    'agent_species': int(np_agent_species[i]),
                    'x': float(np_positions[i][0]),
                    'y': float(np_positions[i][1])
                })
                self.eco_to_box2d_creatures.put({
                    'action': 'create',
                    'agent_id': int(np_agent_ids[i]),
                    'agent_species': int(np_agent_species[i]),
                    'x': float(np_positions[i][0]),
                    'y': float(np_positions[i][1]),
                    'radius': float(radius)
                })

    def run(self, shared_memory, running):
        while running.value:
            time.sleep(1)
            pass
            