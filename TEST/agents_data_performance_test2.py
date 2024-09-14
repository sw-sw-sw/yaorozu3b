import time
import random
import numpy as np
from queue import Queue
from agents_data import AgentsData

def performance_test():
    max_agents = 5000  # 最大エージェント数を5000に設定
    queue_dict = {
        'eco_to_box2d': Queue(),
        'eco_to_visual': Queue(),
        'eco_to_box2d_init': Queue(),
        'eco_to_visual_init': Queue(),
        'eco_to_tf_init': Queue(),
        'box2d_to_eco': Queue(),
        'eco_to_visual_render': Queue()
    }
    
    agents_data = AgentsData(max_agents, queue_dict)

    # 3000エージェントを追加
    print("Adding 3000 agents...")
    start_time = time.time()
    for i in range(3000):
        agents_data.add_agent(species=i % 8, position=(random.uniform(0, 1000), random.uniform(0, 1000)))
    end_time = time.time()
    print(f"Time taken to add 3000 agents: {end_time - start_time:.4f} seconds")

    # ランダムに1000エージェントを追加
    print("\nAdding 1000 random agents...")
    start_time = time.time()
    for _ in range(1000):
        agents_data.add_agent(species=random.randint(0, 7), position=(random.uniform(0, 1000), random.uniform(0, 1000)))
    end_time = time.time()
    print(f"Time taken to add 1000 random agents: {end_time - start_time:.4f} seconds")

    # 現在のエージェント数を確認
    print(f"\nCurrent agent count: {agents_data.current_agent_count}")

    # ランダムに1000エージェントを削除
    print("\nRemoving 1000 random agents...")
    start_time = time.time()
    available_ids = agents_data.available_agent_ids()
    for _ in range(1000):
        if available_ids.size > 0:
            agent_id = np.random.choice(available_ids)
            agents_data.remove_agent(agent_id)
            available_ids = agents_data.available_agent_ids()
    end_time = time.time()
    print(f"Time taken to remove 1000 random agents: {end_time - start_time:.4f} seconds")

    # 最終的なエージェント数を確認
    print(f"\nFinal agent count: {agents_data.current_agent_count}")

    # メモリ使用量を確認
    print(f"\nMemory usage:")
    print(f"Positions array shape: {agents_data.positions.shape}")
    print(f"Species array shape: {agents_data.species.shape}")
    print(f"Agent IDs array shape: {agents_data.agent_ids.shape}")

    # パフォーマンステスト: 100,000回のランダムな追加と削除
    print("\nPerforming 100,000 random add/remove operations...")
    start_time = time.time()
    for _ in range(100000):
        if random.random() < 0.5 and agents_data.current_agent_count < max_agents:
            agents_data.add_agent(species=random.randint(0, 7), position=(random.uniform(0, 1000), random.uniform(0, 1000)))
        elif agents_data.current_agent_count > 0:
            available_ids = agents_data.available_agent_ids()
            if available_ids.size > 0:
                agent_id = np.random.choice(available_ids)
                agents_data.remove_agent(agent_id)
    end_time = time.time()
    print(f"Time taken for 100,000 random operations: {end_time - start_time:.4f} seconds")
    print(f"Final agent count after random operations: {agents_data.current_agent_count}")

if __name__ == "__main__":
    performance_test()