# ecosystem.py
import numpy as np
from config_manager import ConfigManager
from agents_data import AgentsData
import random, time
from log import *

class Ecosystem:
    def __init__(self, queues):
        self.config_manager = ConfigManager()
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')
        self.ad = AgentsData(self.max_agents_num, queues)        # タイマー関連の設定
    
        self.count = 0
        
    def initialize(self):
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
                
        self.ad.send_data_to_box2d_initialize()
        self.ad.send_data_to_tf_initialize()
        self.ad.send_data_to_visual_initialize()

    def update(self):
        # transfer box2data and update from box2d data
        self.ad.update()

        # self.count += 1
        # if self.count == 300:
        #     self.count = 0
        #     self.add_random_agent()
            
    def add_random_agent(self):
        #既存のエージェントが増殖する。
        try:
            agent_ids = self.ad.available_agent_ids()
            if len(agent_ids) > 0:
                agent_id = random.choice(agent_ids)
                species = self.ad.species[agent_id]
                position = self.ad.positions[agent_id] + np.array([1,1])
                new_agent_id = self.ad.add_agent(species, position)
                new_agent_id = self.ad.add_agent(species, position)
                new_agent_id = self.ad.add_agent(species, position)
                new_agent_id = self.ad.add_agent(species, position)
                new_agent_id = self.ad.add_agent(species, position)
                new_agent_id = self.ad.add_agent(species, position)

                if new_agent_id is not None:
                    print(f"Added new agent with ID: {new_agent_id}")
                else:
                    print("Failed to add new agent: maximum capacity reached")
            else:
                print("No agents available for reproduction")
        except Exception as e:
            print(f"Failed to add agent: {e}")
    
    def remove_random_agent(self):
        agent_ids = self.ad.available_agent_ids()
        if len(agent_ids) > 0:
            agent_id = random.choice(agent_ids)
            self.ad.remove_agent(agent_id)
            print(f"Removed agent with ID: {agent_id}")
        else:
            print("No agents available for removal")