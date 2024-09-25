# ecosystem.py
import numpy as np
from config_manager import ConfigManager
from agents_data import AgentsData
import random
from queue import Empty
import time
from log import get_logger
from timer import Timer

logger = get_logger(__name__)

class Ecosystem:
    def __init__(self, queues):
        logger.info("Initializing Ecosystem")
        self.config_manager = ConfigManager()
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')
        self.ad = AgentsData(queues)
        self.ad.initialize()
        self.add_timer = Timer("add random agent")
        self.remove_timer = Timer("remove random agent") # Timer
        self._box2d_to_eco_collisions = queues['box2d_to_eco_collisions'] # Timer
        self.env_energy = self.config_manager.get_trait_value('INITIAL_ENV_ENERGY')
        self.producer_threshold = self.config_manager.get_trait_value('PRODUCER_THRESHOLD')

        logger.info(f"Ecosystem initialized with max_agents_num: {self.max_agents_num}, world_size: {self.world_width}x{self.world_height}")

        
    def initialize(self):
        self.ad.initialize()

    def update(self):
        self.ad.update()
        self.process_collisions()
        self.env_energy += self.ad.update_life_energy()
        self.env_energy += self.ad.check_deaths()
        self.ad.check_reproductions()
        
        if self.ad.current_agent_count < self.max_agents_num:
            self._add_producer()

        # self.random_add_agent(0) 
        # self.random_remove_agents(0)         

    def process_collisions(self):
        try:
            collision_data = self._box2d_to_eco_collisions.get_nowait()
            collisions = collision_data['collisions']
            for agent_id1, agent_id2 in collisions:
                self._handle_collision(agent_id1, agent_id2)
        except Empty:
            pass

    def _handle_collision(self, agent_id1, agent_id2):
        index1 = np.where(self.ad.agent_ids == agent_id1)[0][0]
        index2 = np.where(self.ad.agent_ids == agent_id2)[0][0]
        species1 = self.ad.species[index1]
        species2 = self.ad.species[index2]

        if species1 == species2:
            # Same species interaction (cooperation)
            if self.ad.life_energy[index1] > self.ad.life_energy[index2]:
                energy_transfer = min(self.ad.life_gain[index1], self.ad.life_energy[index1] - self.ad.life_energy[index2])
                self.ad.life_energy[index1] -= energy_transfer
                self.ad.life_energy[index2] += energy_transfer
                logger.debug(f'life energy transferred {agent_id1} to {agent_id2}. ')
            else:
                energy_transfer = min(self.ad.life_gain[index2], self.ad.life_energy[index2] - self.ad.life_energy[index1])
                self.ad.life_energy[index2] -= energy_transfer
                self.ad.life_energy[index1] += energy_transfer
                logger.debug(f'life energy transferred {agent_id2} to {agent_id1}. ')

        else:
            # Different species interaction (predation)
            predator_species = self.config_manager.get_species_trait_value('PREDATOR_SPECIES', species1)
            if species2 == predator_species:
                if random.random() < self.ad.predator_rate[index2]:
                    self.ad.life_energy[index2] += self.ad.life_energy[index1]
                    self.ad.life_energy[index1] = 0
            else:
                predator_species = self.config_manager.get_species_trait_value('PREDATOR_SPECIES', species2)
                if species1 == predator_species:
                    if random.random() < self.ad.predator_rate[index1]:
                        self.ad.life_energy[index1] += self.ad.life_energy[index2]
                        self.ad.life_energy[index2] = 0

    def _add_producer(self):
        if self.env_energy > self.producer_threshold:  # Threshold for adding a new producer
            position = (random.uniform(0, self.world_width), random.uniform(0, self.world_height))
            new_agent_id = self.ad.add_agent(8, position)  # Species 8 is the producer
            if new_agent_id is not None:
                new_index = np.where(self.ad.agent_ids == new_agent_id)[0][0]
                self.ad.life_energy[new_index] = 1000  # Initial energy for the new producer
                self.env_energy -= self.producer_threshold
                logger.info(f"Added new producer agent with ID {new_agent_id} at position {position}")

            
    def random_add_agent(self,num = 5, interval_time = 1):
        if num == 0:
            return
        
        def rnd_pos(radius):
            rnd = np.array([random.randint(-radius, radius), random.randint(-radius, radius)])
            return rnd
        
        if self.add_timer.interval_timer(interval_time):
            try:           
                agent_ids = self.ad.available_agent_ids()
                if len(agent_ids) > 10:
                    agent_id = random.choice(agent_ids)
                    species = self.ad.species[agent_id]
                    position = self.ad.positions[agent_id]
                    if species == 0:
                        logger.warning("No agents available for reproduction")
                    else:
                        for i in range(num):
                            velocity = rnd_pos(3)
                            self.ad.add_agent(species, position + velocity, velocity)                
                else:
                    logger.warning("No agents available for reproduction")
            except Exception as e:
                logger.exception(f"Failed to add agent: {e}")
        

    
    def random_remove_agents(self, num = 10, interval_time = 2):
        if num == 0:
            return
        
        def remove_random_agent():
            agent_ids = self.ad.available_agent_ids()
            if len(agent_ids) > 0:
                agent_id = random.choice(agent_ids)
                self.ad.remove_agent(agent_id)
                logger.info(f"Removed agent with ID: {agent_id}")
            else:
                logger.warning("No agents available for removal")

        if self.remove_timer.interval_timer(interval_time):
            for _ in range(num):
                remove_random_agent()
