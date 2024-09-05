import tensorflow as tf
import numpy as np
from tensorflow_simulation import TensorFlowSimulation
import time
from datetime import datetime

def run_profiling_test(num_agents=1000, num_iterations=100):
    # モックのキューを作成（実際の使用では適切なキューを渡す）
    mock_queues = {'ui_to_tensorflow': None}

    # TensorFlowSimulationインスタンスの作成
    tf_sim = TensorFlowSimulation(mock_queues)

    # テストデータの生成
    num_agents = 3000
    positions = tf.random.uniform((num_agents, 2), minval=0, maxval=tf_sim.world_width)
    species = tf.random.uniform((num_agents,), minval=1, maxval=9, dtype=tf.int32)

    # プロファイリングの有効化
    tf_sim.enable_profiling()

    print(f"Starting profiling test with {num_agents} agents and {num_iterations} iterations...")
    start_time = time.time()

    for _ in range(num_iterations):
        # 主要なメソッドのテスト
        forces = tf_sim.calculate_forces(positions, species)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Test completed in {total_time:.2f} seconds")

    # プロファイリング結果の取得
    profiling_results = tf_sim.get_profiling_results()

    # 結果の詳細な分析
    detailed_results = analyze_results(tf_sim.profiling_results, num_iterations)

    # ファイルに結果を書き込む
    write_results_to_file(num_agents, num_iterations, total_time, profiling_results, detailed_results)

    # プロファイリングの無効化とクリア
    tf_sim.disable_profiling()
    tf_sim.clear_profiling_results()

def analyze_results(profiling_results, num_iterations):
    detailed_results = []
    total_execution_time = sum([sum(times) for times in profiling_results.values()])

    for func_name, times in profiling_results.items():
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        total_time = sum(times)
        percentage = (total_time / total_execution_time) * 100  # 修正された行

        result = {
            "function": func_name,
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "total_time": total_time,
            "percentage": percentage
        }
        detailed_results.append(result)

    return detailed_results

def write_results_to_file(num_agents, num_iterations, total_time, profiling_results, detailed_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"profiling_results/profiling_results_{timestamp}.txt"

    with open(filename, "a") as f:
        f.write(f"Profiling Test Results - {timestamp}\n")
        f.write(f"Number of agents: {num_agents}\n")
        f.write(f"Number of iterations: {num_iterations}\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        
        f.write("Profiling Results:\n")
        f.write(profiling_results)
        f.write("\n")

        f.write("Detailed Analysis:\n")
        for result in detailed_results:
            f.write(f"\n{result['function']}:\n")
            f.write(f"  Average time: {result['average_time']:.6f} seconds\n")
            f.write(f"  Minimum time: {result['min_time']:.6f} seconds\n")
            f.write(f"  Maximum time: {result['max_time']:.6f} seconds\n")
            f.write(f"  Total time: {result['total_time']:.6f} seconds\n")
            f.write(f"  Percentage of total execution time: {result['percentage']:.2f}%\n")

        f.write("\n" + "="*50 + "\n\n")

    print(f"Results have been written to {filename}")

if __name__ == "__main__":
    # run_profiling_test()

    # 異なるパラメータでテストを実行する例
    # run_profiling_test(num_agents=5000, num_iterations=50)
    run_profiling_test(num_agents=3000, num_iterations=200)