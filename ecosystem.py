# ecosystem.py
import numpy as np
from config_manager import ConfigManager
from agents_data import AgentsData
import random, time

class Ecosystem:
    def __init__(self, queues):
        self.config_manager = ConfigManager()
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')
        self.ad = AgentsData(self.max_agents_num, queues)

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
                self.ad.add_agent(species, (float(x[i]), float(y[i])))
                
        self.ad.send_data_to_box2d_initialize()
        self.ad.send_data_to_tf_initialize()
        self.ad.send_data_to_visual_initialize()

    def update(self):
        self.ad.update_from_box2d()
        self.ad.send_data_to_visual()
        
        # if random.random() < 0.01:  # 5%の確率で追加
        #     self.add_random_agent()
        # if random.random() < 0.05:  # 3%の確率で削除
        #     self.remove_random_agent()
        # time.sleep(0.1)
        
    # for test of add_agent()
    
    def add_random_agent(self):
        #既存のエージェントが増殖する。
        try:
            agent_id = random.choice(self.ad.available_agent_ids())
            species = self.ad.species[agent_id]
            position =self.ad.positions[agent_id]
            self.ad.add_agent(species, position)
        except ValueError as e:
            print(f"Failed to add agent: {e}")
    
    # for test of remove_agent() 
    
    def remove_random_agent(self):
        if self.ad.current_agent_count > 0:
            # ランダムなエージェントIDを選択
            agent_id = random.choice(self.ad.available_agent_ids())
            self.ad.remove_agent(agent_id)
