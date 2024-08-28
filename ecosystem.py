import numpy as np
from config import *
import time

class Ecosystem:
    def __init__(self, queues):
        self.shared_memory_manager = shared_memory_manager
        self.queue_eco_to_box2d = queues['eco_to_box2d']
        self.queue_box2d_to_eco = queues['box2d_to_eco']
        self.queue_eco_to_visual = queues['eco_to_visual']
        self.queue_visual_to_eco = queues['visual_to_eco']
        self.queue_eco_to_tensorflow = queues['eco_to_tensorflow']
        self.queue_tensorflow_to_eco = queues['tensorflow_to_eco']
        self.num_agents = shared_memory_manager.max_agents
        self.initialize_system()

    def initialize_system(self):
        self.initialize_agents()
        self.initialize_subsystems()

    def initialize_agents(self):
        initial_positions = np.random.uniform(0, 1, (self.num_agents, 2)).astype(np.float32)
        initial_positions[:, 0] *= WORLD_WIDTH
        initial_positions[:, 1] *= WORLD_HEIGHT
        initial_velocities = np.random.uniform(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX, (self.num_agents, 2)).astype(np.float32)
        print(initial_positions)
        # まとめてエージェントをSharedMemoryManagerに追加
        for i in range(self.num_agents):
            self.shared_memory_manager.add_agent(initial_positions[i], initial_velocities[i], 0)  # 0 is a placeholder for species

    def initialize_subsystems(self):
        # 各サブシステムに初期化コマンドを送信
        self.queue_eco_to_box2d.put(('initialize',))
        self.queue_eco_to_visual.put(('initialize',))
        self.queue_eco_to_tensorflow.put(('initialize',))

        # 各サブシステムからの初期化完了通知を待つ
        self.wait_for_initialization(self.queue_box2d_to_eco)
        self.wait_for_initialization(self.queue_visual_to_eco)
        self.wait_for_initialization(self.queue_tensorflow_to_eco)
        
    def wait_for_initialization(self, queue):
        while True:
            message = queue.get()
            if message == 'initialization_complete':
                break

    def run(self):
        while True:
            self.step()
            time.sleep(0.01)
            print('ecosystem loop!')

    def step(self):
        # 各プロセスからの更新を非同期的に処理
        self.process_box2d_updates()
        self.process_tensorflow_updates()
        self.process_visual_system_updates()

        # 必要に応じて新しい計算をトリガー
        self.queue_eco_to_box2d.put(('step',))
        self.queue_eco_to_tensorflow.put(('step',))
        self.queue_eco_to_visual.put(('step',))

        # エコシステムの状態を更新（例：エージェントの追加/削除）
        # self.update_ecosystem_state()

    def process_box2d_updates(self):
        while not self.queue_box2d_to_eco.empty():
            update = self.queue_box2d_to_eco.get()
            # Box2Dからの更新を処理

    def process_tensorflow_updates(self):
        while not self.queue_tensorflow_to_eco.empty():
            update = self.queue_tensorflow_to_eco.get()
            # TensorFlowからの更新を処理

    def process_visual_system_updates(self):
        while not self.queue_visual_to_eco.empty():
            update = self.queue_visual_to_eco.get()
            # Visual Systemからの更新を処理

    # def update_ecosystem_state(self):
    #     # エコシステムの状態を更新（エージェントの追加/削除など）
    #     pass