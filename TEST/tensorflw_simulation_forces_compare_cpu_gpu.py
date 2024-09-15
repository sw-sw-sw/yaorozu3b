import tensorflow as tf
import numpy as np
from tensorflow_simulation import TensorFlowSimulation
import time
from queue import Queue
import matplotlib.pyplot as plt


    
def run_performance_test(num_agents=1000, num_iterations=100):
    # 必要なキューを作成
    queues = {
        'ui_to_tensorflow': Queue(),
        'box2d_to_tf': Queue(),
        'eco_to_tf_init': Queue(),
        'tf_to_box2d': Queue()
    }

    # TensorFlowSimulationインスタンスの作成
    tf_sim = TensorFlowSimulation(queues, max_agents=num_agents)

    # 初期化データの生成とキューへの追加
    init_data = {
        'positions': np.random.uniform(0, tf_sim.world_width, (num_agents, 2)).astype(np.float32),
        'species': np.random.randint(1, 9, num_agents, dtype=np.int32),
        'current_agent_count': num_agents
    }
    queues['eco_to_tf_init'].put(init_data)

    # initialize()メソッドの呼び出し
    tf_sim.initialize()

    # パフォーマンステストの実行
    execution_times = []
    for _ in range(num_iterations):
        # Box2Dからのデータ更新をシミュレート
        update_data = {
            'positions': np.random.uniform(0, tf_sim.world_width, (num_agents, 2)).astype(np.float32),
            'species': np.random.randint(1, 9, num_agents, dtype=np.int32),
            'current_agent_count': num_agents
        }
        queues['box2d_to_tf'].put(update_data)

        # プロパティの更新
        tf_sim.update_property()

        # calculate_forcesの実行時間を計測
        start_time = time.time()
        forces = tf_sim.calculate_forces()
        end_time = time.time()

        execution_times.append(end_time - start_time)

    # 統計情報の計算
    avg_time = np.mean(execution_times)
    min_time = np.min(execution_times)
    max_time = np.max(execution_times)

    print(f"Performance Test Results:")
    print(f"Number of agents: {num_agents}")
    print(f"Number of iterations: {num_iterations}")
    print(f"Average execution time of calculate_forces: {avg_time:.6f} seconds")
    print(f"Minimum execution time: {min_time:.6f} seconds")
    print(f"Maximum execution time: {max_time:.6f} seconds")

    # TensorFlowの統計情報を表示
    print("\nTensorFlow Statistics:")
    print(tf.config.experimental.get_memory_info('GPU:0'))
    print(tf.config.list_physical_devices())
    
    analyze_performance(execution_times, num_agents, num_iterations)

def analyze_performance(execution_times, num_agents, num_iterations):
    percentiles = np.percentile(execution_times, [25, 50, 75, 90, 95, 99])
    
    print(f"25th percentile: {percentiles[0]:.6f} seconds")
    print(f"Median (50th percentile): {percentiles[1]:.6f} seconds")
    print(f"75th percentile: {percentiles[2]:.6f} seconds")
    print(f"90th percentile: {percentiles[3]:.6f} seconds")
    print(f"95th percentile: {percentiles[4]:.6f} seconds")
    print(f"99th percentile: {percentiles[5]:.6f} seconds")
    
    plt.figure(figsize=(10, 6))
    plt.hist(execution_times, bins=50, edgecolor='black')
    plt.title(f'Distribution of Execution Times (Agents: {num_agents}, Iterations: {num_iterations})')
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Frequency')
    plt.savefig(f'execution_time_distribution_{num_agents}_{num_iterations}.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(execution_times)
    plt.title(f'Execution Times over Iterations (Agents: {num_agents}, Iterations: {num_iterations})')
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (seconds)')

    # x軸のラベルを適切に設定
    if num_iterations > 100:
        step = num_iterations // 10  # 10個程度のラベルを表示
        plt.xticks(range(0, num_iterations, step), range(0, num_iterations, step))
    
    plt.savefig(f'execution_time_trend_{num_agents}_{num_iterations}.png')
    plt.close()

    # 移動平均を計算してプロット
    window_size = min(50, num_iterations // 10)  # 移動平均の窓サイズ
    moving_average = np.convolve(execution_times, np.ones(window_size)/window_size, mode='valid')
    
    plt.figure(figsize=(12, 6))
    plt.plot(moving_average)
    plt.title(f'Moving Average of Execution Times (Window: {window_size}, Agents: {num_agents})')
    plt.xlabel('Iteration')
    plt.ylabel('Average Execution Time (seconds)')
    
    # x軸のラベルを適切に設定
    if len(moving_average) > 100:
        step = len(moving_average) // 10
        plt.xticks(range(0, len(moving_average), step), range(window_size-1, num_iterations, step))
    
    plt.savefig(f'TEST/execution_time_moving_average_{num_agents}_{num_iterations}.png')
    plt.close()
    
if __name__ == "__main__":
    run_performance_test(num_agents=2500, num_iterations=500)