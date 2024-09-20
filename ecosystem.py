# ecosystem.py
import numpy as np
from config_manager import ConfigManager
from agents_data import AgentsData
import random
import time
from log import get_logger

logger = get_logger(__name__)
class Ecosystem:
    def __init__(self, queues):
        logger.info("Initializing Ecosystem")
        self.config_manager = ConfigManager()
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')
        self.ad = AgentsData(self.max_agents_num, queues)        # タイマー関連の設定
    
        self.count = 0
        logger.info(f"Ecosystem initialized with max_agents_num: {self.max_agents_num}, world_size: {self.world_width}x{self.world_height}")

        
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
            circle_radius = self.world_width / 6

            # 正規分布を使用して円内にランダムな位置を生成
            r = np.random.normal(0, circle_radius / 2, initial_agent_num)
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

    def update(self):
        # transfer box2data and update from box2d data
        self.ad.update()

        # self.count += 1
        # if self.count == 20:
        #     self.count = 0
        #     self.remove_random_agents(1)
            
        
        # if self.count == 5:
        #     self.count = 0
        #     self.add_random_agent(1)
        # if self.count == 10:
        #     self.count = 0
        #     self.add_random_agent(2)
            
            
    def add_random_agent(self,num = 1):
        try:
            agent_ids = self.ad.available_agent_ids()
            if len(agent_ids) > 10:
                agent_id = random.choice(agent_ids)
                species = self.ad.species[agent_id]
                position = self.ad.positions[agent_id]
                
                for _ in range(num):
                    velocity = self.rnd_pos(5)
                    new_agent_id = self.ad.add_agent(species, position + velocity, velocity)
                    # time.sleep(0.001)
                    if new_agent_id is None:
                        break
                   
                
                if new_agent_id is not None:
                    logger.info(f"Added new agent with ID: {new_agent_id}")
                else:
                    logger.warning("Failed to add new agent: maximum capacity reached")
            else:
                logger.warning("No agents available for reproduction")
        except Exception as e:
            logger.exception(f"Failed to add agent: {e}")
    
    def remove_random_agents(self, num = 10):
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
            
    def rnd_pos(self, radius):
        rnd = np.array([random.randint(-radius, radius), random.randint(-radius, radius)])
        return rnd