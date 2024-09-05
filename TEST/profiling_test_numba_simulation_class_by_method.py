import numpy as np
import time
from datetime import datetime
from numba_simulation import NumbaSimulation, separation, cohesion, predator_prey_forces, environment_forces

def run_profiling_test(num_agents=3000, num_iterations=200):
    mock_queues = {'ui_to_numba': None}
    numba_sim = NumbaSimulation(mock_queues)

    positions = np.random.uniform(0, numba_sim.world_width, (num_agents, 2)).astype(np.float32)
    species = np.random.randint(1, 9, num_agents, dtype=np.int32)
    distances = np.random.uniform(0, numba_sim.world_width, (num_agents, num_agents)).astype(np.float32)

    print(f"Starting profiling test with {num_agents} agents and {num_iterations} iterations...")
    start_time = time.time()

    profiling_results = {}

    for method_name, method in [
        ("separation", separation),
        ("cohesion", cohesion),
        ("predator_prey_forces", predator_prey_forces),
        ("environment_forces", environment_forces)
    ]:
        method_times = []
        for i in range(num_iterations):
            if method_name == "separation":
                start = time.time()
                method(positions, distances, numba_sim.separation_distance, numba_sim.separation_weight)
                end = time.time()
            elif method_name == "cohesion":
                start = time.time()
                method(positions, distances, numba_sim.cohesion_distance, numba_sim.cohesion_weight)
                end = time.time()
            elif method_name == "predator_prey_forces":
                start = time.time()
                method(positions, distances, species, numba_sim.predator_species, numba_sim.prey_species,
                       numba_sim.escape_distance, numba_sim.chase_distance, numba_sim.escape_weight, numba_sim.chase_weight)
                end = time.time()
            elif method_name == "environment_forces":
                start = time.time()
                method(positions, numba_sim.world_center, numba_sim.world_radius,
                       numba_sim.center_attraction_weight, numba_sim.confinement_weight,
                       numba_sim.rotation_strength, numba_sim.max_force)
                end = time.time()
            
            if i > 0:  # 最初の実行を除外
                method_times.append(end - start)
        
        profiling_results[method_name] = method_times

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Test completed in {total_time:.2f} seconds")

    detailed_results = analyze_results(profiling_results, num_iterations)
    write_results_to_file(num_agents, num_iterations, total_time, profiling_results, detailed_results)

def remove_outliers(data, threshold=3):
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)
    return data[abs(data - mean) < threshold * std]

def analyze_results(profiling_results, num_iterations):
    detailed_results = []
    total_execution_time = 0

    for func_name, times in profiling_results.items():
        filtered_times = remove_outliers(times)
        avg_time = np.mean(filtered_times)
        min_time = np.min(filtered_times)
        max_time = np.max(filtered_times)
        total_time = np.sum(filtered_times)
        total_execution_time += total_time

        result = {
            "function": func_name,
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "total_time": total_time,
            "filtered_count": len(filtered_times),
            "original_count": len(times)
        }
        detailed_results.append(result)

    for result in detailed_results:
        result["percentage"] = (result["total_time"] / total_execution_time) * 100

    return detailed_results

def write_results_to_file(num_agents, num_iterations, total_time, profiling_results, detailed_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"profiling_results/numba_profiling_results_{timestamp}.txt"

    with open(filename, "a") as f:
        f.write(f"Numba Simulation Profiling Test Results - {timestamp}\n")
        f.write(f"Number of agents: {num_agents}\n")
        f.write(f"Number of iterations: {num_iterations}\n")
        f.write(f"Total execution time: {total_time:.2f} seconds\n\n")
        
        f.write("Profiling Results (after outlier removal):\n")
        for result in detailed_results:
            f.write(f"{result['function']}: Average execution time: {result['average_time']:.6f} seconds\n")
        f.write("\n")

        f.write("Detailed Analysis:\n")
        for result in detailed_results:
            f.write(f"\n{result['function']}:\n")
            f.write(f"  Average time: {result['average_time']:.6f} seconds\n")
            f.write(f"  Minimum time: {result['min_time']:.6f} seconds\n")
            f.write(f"  Maximum time: {result['max_time']:.6f} seconds\n")
            f.write(f"  Total time: {result['total_time']:.6f} seconds\n")
            f.write(f"  Percentage of total execution time: {result['percentage']:.2f}%\n")
            f.write(f"  Measurements used: {result['filtered_count']} (filtered) / {result['original_count']} (original)\n")

        f.write("\n" + "="*50 + "\n\n")

    print(f"Results have been written to {filename}")

if __name__ == "__main__":
    run_profiling_test()