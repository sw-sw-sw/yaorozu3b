# ecosystem.py
import numpy as np
from config import *

class Ecosystem:
    def __init__(self):
        self.max_agents = NUM_AGENTS
        self.current_agent_count = NUM_AGENTS

    def initialize_agents(self, shared_memory):
        with shared_memory['lock']:
            np_positions = np.frombuffer(shared_memory['positions'].get_obj(), dtype=np.float32).reshape((self.max_agents, 2))
            np_velocities = np.frombuffer(shared_memory['velocities'].get_obj(), dtype=np.float32).reshape((self.max_agents, 2))
            np_forces = np.frombuffer(shared_memory['forces'].get_obj(), dtype=np.float32).reshape((self.max_agents, 2))
            np_agent_ids = np.frombuffer(shared_memory['agent_ids'].get_obj(), dtype=np.int32)
            np_agent_species = np.frombuffer(shared_memory['agent_species'].get_obj(), dtype=np.int32)

            # Initialize agents
            np_positions[:self.current_agent_count] = np.random.uniform(0, 1, (self.current_agent_count, 2))
            np_positions[:self.current_agent_count, 0] *= WORLD_WIDTH
            np_positions[:self.current_agent_count, 1] *= WORLD_HEIGHT
            np_velocities[:self.current_agent_count] = np.random.uniform(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX, (self.current_agent_count, 2))
            np_forces[:self.current_agent_count] = np.zeros((self.current_agent_count, 2))
            np_agent_ids[:self.current_agent_count] = np.arange(self.current_agent_count)
            np_agent_species[:self.current_agent_count] = np.zeros(self.current_agent_count, dtype=np.int32)

            shared_memory['current_agent_count'].value = self.current_agent_count

    def run(self, shared_memory, queues, running):
        while running.value:
            with shared_memory['lock']:
                # Ecosystem management logic here
                # Access shared memory directly each time it's needed
                # ... work with np_positions ...
                pass
