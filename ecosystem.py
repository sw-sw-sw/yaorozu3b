# ecosystem.py
import numpy as np
from config import *
from creature import Creature
import pygame

class Ecosystem:
    def __init__(self, queues):
        self.eco_to_visual_creatures = queues['eco_to_visual_creatures']
        self.eco_to_box2d_creatures = queues['eco_to_box2d_creatures']

    def initialize_agents(self, shared_memory):
        with shared_memory['lock']:
            np_positions = np.frombuffer(shared_memory['positions'].get_obj(), dtype=np.float32).reshape((MAX_AGENTS_NUM, 2))
            np_velocities = np.frombuffer(shared_memory['velocities'].get_obj(), dtype=np.float32).reshape((MAX_AGENTS_NUM, 2))
            np_forces = np.frombuffer(shared_memory['forces'].get_obj(), dtype=np.float32).reshape((MAX_AGENTS_NUM, 2))
            np_agent_ids = np.frombuffer(shared_memory['agent_ids'].get_obj(), dtype=np.int32)
            np_agent_species = np.frombuffer(shared_memory['agent_species'].get_obj(), dtype=np.int32)

            # 円形領域の中心と半径を定義
            center_x, center_y = WORLD_WIDTH / 2, WORLD_HEIGHT / 2
            radius = min(WORLD_WIDTH, WORLD_HEIGHT) / 2

            # 円形領域内にランダムな位置を生成
            angles = np.random.uniform(0, 2 * np.pi, MAX_AGENTS_NUM)
            radii = np.sqrt(np.random.uniform(0, 1, MAX_AGENTS_NUM)) * radius  # 均一な分布のために平方根を使用
            
            np_positions[:, 0] = center_x + radii * np.cos(angles)
            np_positions[:, 1] = center_y + radii * np.sin(angles)

            np_velocities[:] = np.random.uniform(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX, (MAX_AGENTS_NUM, 2))
            np_forces[:] = np.zeros((MAX_AGENTS_NUM, 2))
            np_agent_ids[:] = np.arange(MAX_AGENTS_NUM)
            np_agent_species[:] = np.random.randint(1, 9, MAX_AGENTS_NUM)

            shared_memory['current_agent_count'].value = MAX_AGENTS_NUM

            for i in range(MAX_AGENTS_NUM):
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
            with shared_memory['lock']:
                # Ecosystem management logic here
                pass