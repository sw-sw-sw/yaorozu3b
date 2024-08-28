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
            np_positions = np.frombuffer(shared_memory['positions'].get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))
            np_velocities = np.frombuffer(shared_memory['velocities'].get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))
            np_forces = np.frombuffer(shared_memory['forces'].get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))
            np_agent_ids = np.frombuffer(shared_memory['agent_ids'].get_obj(), dtype=np.int32)
            np_agent_species = np.frombuffer(shared_memory['agent_species'].get_obj(), dtype=np.int32)

            np_positions[:] = np.random.uniform(0, 1, (NUM_AGENTS, 2))
            np_positions[:, 0] *= WORLD_WIDTH
            np_positions[:, 1] *= WORLD_HEIGHT
            np_velocities[:] = np.random.uniform(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX, (NUM_AGENTS, 2))
            np_forces[:] = np.zeros((NUM_AGENTS, 2))
            np_agent_ids[:] = np.arange(NUM_AGENTS)
            np_agent_species[:] = np.random.randint(1, 9, NUM_AGENTS)

            shared_memory['current_agent_count'].value = NUM_AGENTS

            for i in range(NUM_AGENTS):
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