import time
import random
import numpy as np
from queue import Queue
from agents_data import AgentsData

def simulate_ecosystem(duration_seconds=300):  # 5分間のシミュレーション
    max_agents = 5000
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

    # 初期エージェントを追加（例：1000エージェント）
    print("Initializing with 1000 agents...")
    for _ in range(1000):
        agents_data.add_agent(
            species=random.randint(0, 7),
            position=(random.uniform(0, 1000), random.uniform(0, 1000))
        )

    print(f"Initial agent count: {agents_data.current_agent_count}")

    start_time = time.time()
    operation_count = 0
    add_count = 0
    remove_count = 0

    print("Starting ecosystem simulation...")
    while time.time() - start_time < duration_seconds:
        # ランダムな間隔で操作を実行
        time.sleep(random.uniform(0.001, 0.01))  # 1〜10ミリ秒の間隔

        if random.random() < 0.55:  # 55%の確率で追加
            if agents_data.current_agent_count < max_agents:
                agents_data.add_agent(
                    species=random.randint(0, 7),
                    position=(random.uniform(0, 1000), random.uniform(0, 1000))
                )
                add_count += 1
        else:  # 45%の確率で削除
            if agents_data.current_agent_count > 0:
                available_ids = agents_data.available_agent_ids()
                if available_ids.size > 0:
                    agent_id = np.random.choice(available_ids)
                    agents_data.remove_agent(agent_id)
                    remove_count += 1

        operation_count += 1

        # 進捗状況を1分ごとに表示
        if operation_count % 1000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Time: {elapsed_time:.2f}s, Agents: {agents_data.current_agent_count}, "
                  f"Ops: {operation_count}, Adds: {add_count}, Removes: {remove_count}")

    end_time = time.time()
    total_time = end_time - start_time

    print("\nSimulation complete.")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total operations: {operation_count}")
    print(f"Total adds: {add_count}")
    print(f"Total removes: {remove_count}")
    print(f"Final agent count: {agents_data.current_agent_count}")
    print(f"Operations per second: {operation_count / total_time:.2f}")

    # メモリ使用量を確認
    print("\nMemory usage:")
    print(f"Positions array shape: {agents_data.positions.shape}")
    print(f"Species array shape: {agents_data.species.shape}")
    print(f"Agent IDs array shape: {agents_data.agent_ids.shape}")

if __name__ == "__main__":
    simulate_ecosystem()