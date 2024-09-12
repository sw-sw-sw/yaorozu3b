# ecosystem.py
import numpy as np
from creature import Creature
import pygame
from config_manager import ConfigManager
import time

class Ecosystem:
    def __init__(self, queues):
        self.queues = queues
        self.eco_to_visual_creatures = queues['eco_to_visual_creatures']
        self.eco_to_box2d_creatures = queues['eco_to_box2d_creatures']
        self.eco_to_tf = queues['eco_to_tf']

        self.config_manager = ConfigManager()
        
        # ConfigManagerから値を取得してプロパティとして設定
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')

        self.positions = np.zeros((self.max_agents_num, 2), dtype=np.float32)
        self.species = np.zeros(self.max_agents_num, dtype=np.int32)
        self.current_agent_count = 0
        self.agent_ids = np.arange(self.max_agents_num, dtype=np.int32)
    

    def initialize2(self):


        # 正八角形の頂点を計算
        octagon_radius = self.world_width / 4
        octagon_centers = []
        for i in range(8):
            angle = i * np.pi / 4
            x = self.world_width / 2 + octagon_radius * np.cos(angle)
            y = self.world_height / 2 + octagon_radius * np.sin(angle)
            octagon_centers.append((x, y))

        # 各種ごとにエージェントの位置とspeciesを初期化
        current_index = 0
        for species in range(1, 9):
            initial_agent_num = self.config_manager.get_species_trait_value('INITIAL_AGENT_NUM', species)
            center_x, center_y = octagon_centers[species - 1]
            circle_radius = self.world_width / 6

            # 正規分布を使用して円内にランダムな位置を生成
            r = np.random.normal(0, circle_radius / 2, initial_agent_num)
            theta = np.random.uniform(0, 2 * np.pi, initial_agent_num)
            x = center_x + r * np.cos(theta)
            y = center_y + r * np.sin(theta)

            end_index = current_index + initial_agent_num
            self.positions[current_index:end_index, 0] = x
            self.positions[current_index:end_index, 1] = y
            self.species[current_index:end_index] = species
            current_index = end_index

        self.current_agent_count = current_index

        # 初期化データを他のコンポーネントに送信
        init_data1 = {
            'positions': self.positions,
            'agent_species': self.species,
            'current_agent_count': self.current_agent_count
        }
        
        init_data2 = {
            'positions': self.positions[:self.current_agent_count],
            'agent_species': self.species[:self.current_agent_count],
            'agent_ids': self.agent_ids[:self.current_agent_count],
            'current_agent_count': self.current_agent_count
        }

        # 必要に応じて他のコンポーネントにデータを送信
        self.eco_to_box2d_creatures.put(init_data2)
        self.eco_to_tf.put(init_data1)
        self.eco_to_visual_creatures.put(init_data2)

    def run(self, shared_memory, running):
        while running.value:
            time.sleep(1)
            # 今後、エージェント同士の反応処理などをここに追加
            pass