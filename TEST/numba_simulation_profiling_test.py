import numpy as np
from predator_prey_force_numba import calculate_forces_numba
import time

def run_profiling_test(num_agents=1000, num_iterations=100):
    # パラメータの設定
    escape_distance = 20.0
    chase_distance = 50.0
    escape_weight = 50.0
    chase_weight = 50.0

    # テストデータの生成
    positions = np.random.uniform(0, 3000, (num_agents, 2)).astype(np.float64)
    species = np.random.randint(1, 9, num_agents, dtype=np.int32)
    predator_species = np.array([2, 3, 4, 5, 6, 7, 8, 1], dtype=np.int32)
    prey_species = np.array([8, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)

    print(f"Starting profiling test with {num_agents} agents and {num_iterations} iterations...")
    start_time = time.time()

    # Compile the function
    calculate_forces_numba(positions, species, predator_species, prey_species,
                     escape_distance, chase_distance, escape_weight, chase_weight)

    for _ in range(num_iterations):
        forces = calculate_forces_numba(positions, species, predator_species, prey_species,
                                  escape_distance, chase_distance, escape_weight, chase_weight)

    end_time = time.time()
    total_time = end_time - start_time
    print("="*40)
    print(f"Test completed in {total_time:.2f} seconds")
    print(f"Average time per iteration: {1000*total_time/num_iterations:.1f} ms")
    print("="*40)

if __name__ == "__main__":
    # run_profiling_test()
    run_profiling_test(num_agents=3000, num_iterations=100)