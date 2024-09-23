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
        self.ad = AgentsData(self.max_agents_num, queues)        # タイマー関連の設定
        self.add_timer = Timer("add random agent")
        self.remove_timer = Timer("remove random agent")
        self._box2d_to_eco_collisions = queues['box2d_to_eco_collisions']
        logger.info(f"Ecosystem initialized with max_agents_num: {self.max_agents_num}, world_size: {self.world_width}x{self.world_height}")

        self.random_add_agents_num = 50
        self.random_remove_agents_num = 0
    
    # ----------------- Initialize ----------------------
    
    def initialize(self):
        logger.info("Initializing Ecosystem agents")
        octagon_radius = self.world_width / 4
        octagon_centers = []
        
        for i in range(8):
            angle = i * np.pi / 4
            x = self.world_width / 2 + octagon_radius * np.cos(angle)
            y = self.world_height / 2 + octagon_radius * np.sin(angle)
            octagon_centers.append((x, y))

        # 各種ごとにエージェントの位置とspeciesを初期化
        for species in range(1, 9):
            initial_agent_num = self.config_manager.get_species_trait_value('INITIAL_AGENT_NUM', species)
            center_x, center_y = octagon_centers[species - 1]
            circle_radius = self.world_width / 5

            # 正規分布を使用して円内にランダムな位置を生成
            r = np.random.randint(0, circle_radius / 2, initial_agent_num)
            theta = np.random.uniform(0, 2 * np.pi, initial_agent_num)
            x = center_x + r * np.cos(theta)
            y = center_y + r * np.sin(theta)
            for i in range(initial_agent_num):
                self.ad.add_agent_no_notify(species, (float(x[i]), float(y[i])))
            logger.info(f"Initialized {initial_agent_num} agents for species {species}")
                
        self.ad.send_data_to_box2d_initialize()
        self.ad.send_data_to_tf_initialize()
        self.ad.send_data_to_visual_initialize()
        logger.info("Ecosystem initialization completed")

    # ------------------ update ---------------------

    def update(self):
        # transfer box2data and update from box2d data
        self.ad.update()
        self.process_collisions() 
        
        # if self.add_timer.interval_timer(2):
        #     self.add_random_agent(30) 

        # if self.remove_timer.interval_timer(2):
        #     self.remove_random_agents(1)         
    
    # ----------------- collision ----------------------
    
    def process_collisions(self):
        try:
            collision_data = self._box2d_to_eco_collisions.get_nowait()
            collisions = collision_data['collisions']
            # for agent_id1, agent_id2 in collision_data['collisions']:
            print('collisions data num = ', len(collisions))
        except Empty:
            pass
    
    # ----------------- Random test ----------------------
            
    # add agent 
            
    def add_random_agent(self,num = 10):
        if num == 0:
            return
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
                        velocity = self.rnd_pos(3)
                        self.ad.add_agent(species, position + velocity, velocity)
                
                    
            else:
                logger.warning("No agents available for reproduction")
        except Exception as e:
            logger.exception(f"Failed to add agent: {e}")
    
    # remove agents
    
    def remove_random_agents(self, num = 10):
        if num == 0:
            return
        for _ in range(num):
            self.remove_random_agent()
    
    def remove_random_agent(self):
        agent_ids = self.ad.available_agent_ids()
        if len(agent_ids) > 0:
            agent_id = random.choice(agent_ids)
            self.ad.remove_agent(agent_id)
            logger.info(f"Removed agent with ID: {agent_id}")
        else:
            logger.warning("No agents available for removal")
            
    # ----------------- sub method ----------------------
    
    def rnd_pos(self, radius):
        rnd = np.array([random.randint(-radius, radius), random.randint(-radius, radius)])
        return rnd