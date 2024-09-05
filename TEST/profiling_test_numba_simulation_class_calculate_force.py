import numpy as np
from numba_simulation import NumbaSimulation
from config_manager import ConfigManager
import time

def run_profiling_test(num_agents=3000, num_iterations=100):
    # ConfigManagerのインスタンス化
    config_manager = ConfigManager()

    # NumbaSimulationのインスタンス化（ダミーのqueuesを渡す）
    queues = {'ui_to_numba': None}  # 実際のキューは使用しないのでNoneを渡す
    nmb = NumbaSimulation(queues)

    # パラメータの設定（ConfigManagerから取得）
    world_width = config_manager.get_trait_value('WORLD_WIDTH')
    world_height = config_manager.get_trait_value('WORLD_HEIGHT')

    # テストデータの生成
    positions = np.random.uniform(0, min(world_width, world_height), (num_agents, 2)).astype(np.float32)
    species = np.random.randint(1, 9, num_agents, dtype=np.int32)

    print(f"Starting profiling test with {num_agents} agents and {num_iterations} iterations...")
    
    # プロファイリングを有効化
    nmb.enable_profiling()

    start_time = time.time()

    # 最初の呼び出しでJITコンパイル
    nmb.calculate_forces(positions, species)

    for _ in range(num_iterations):
        forces = nmb.calculate_forces(positions, species)

    end_time = time.time()
    total_time = end_time - start_time

    print("="*40)
    print(f"Test completed in {total_time:.2f} seconds")
    print(f"Average time per iteration: {1000*total_time/num_iterations:.1f} ms")
    print("="*40)

    # プロファイリング結果の表示
    print(nmb.get_profiling_results())

    # プロファイリングを無効化
    nmb.disable_profiling()

if __name__ == "__main__":
    run_profiling_test(num_agents=3000, num_iterations=100)