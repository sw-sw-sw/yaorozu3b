import numpy as np
from numba import njit, prange, vectorize
from numba import set_num_threads
from config_manager import ConfigManager
import time

# スレッド数を設定（CPUコア数に応じて調整してください）
set_num_threads(4)

@vectorize(['float32(float32, float32, float32, float32)'])
def vec_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@njit
def calculate_distances(positions):
    n = len(positions)
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        distances[i] = vec_distance(positions[i, 0], positions[i, 1], positions[:, 0], positions[:, 1])
    return distances

@njit
def limit_magnitude(vectors, max_magnitude):
    magnitudes = np.sqrt(np.sum(vectors**2, axis=1))
    scale = np.minimum(max_magnitude / (magnitudes + 1e-5), 1.0)
    return vectors * scale[:, np.newaxis]

@njit
def separation(positions, distances, separation_distance, separation_weight):
    forces = np.zeros_like(positions)
    for i in prange(len(positions)):
        force = np.zeros(2)
        count = 0
        for j, dist in enumerate(distances[i]):
            if 0 < dist < separation_distance:
                diff = positions[i] - positions[j]
                force += diff / (dist + 1e-5)
                count += 1
        forces[i] = force * separation_weight if count > 0 else force
    return forces

@njit
def cohesion(positions, distances, cohesion_distance, cohesion_weight):
    forces = np.zeros_like(positions)
    for i in prange(len(positions)):
        center = np.zeros(2)
        count = 0
        for j, dist in enumerate(distances[i]):
            if 0 < dist < cohesion_distance:
                center += positions[j]
                count += 1
        if count > 0:
            center /= count
            forces[i] = (center - positions[i]) * cohesion_weight
    return forces

@njit
def environment_forces(positions, world_center, world_radius, center_attraction_weight, confinement_weight, rotation_strength):
    to_center = world_center - positions
    distances = np.sqrt(np.sum(to_center**2, axis=1))
    normalized_to_center = to_center / (distances[:, np.newaxis] + 1e-5)

    center_force = center_attraction_weight * normalized_to_center
    outside_circle = (distances > world_radius)[:, np.newaxis]
    confinement_force = outside_circle * confinement_weight * (distances[:, np.newaxis] - world_radius) * normalized_to_center

    rotation_force = np.column_stack((-to_center[:, 1], to_center[:, 0]))
    rotation_magnitudes = np.sqrt(np.sum(rotation_force**2, axis=1))
    rotation_force = rotation_force / (rotation_magnitudes[:, np.newaxis] + 1e-5) * rotation_strength

    return center_force + confinement_force + rotation_force

@njit
def predator_prey_forces(positions, distances, species, all_species, predator_species, prey_species,
                         escape_distance, chase_distance, escape_weight, chase_weight):
    forces = np.zeros_like(positions)
    for i in prange(len(positions)):
        predator = predator_species[species[i] - 1]
        prey = prey_species[species[i] - 1]

        nearest_predator_dist = np.inf
        nearest_prey_dist = np.inf
        nearest_predator = -1
        nearest_prey = -1

        for j, dist in enumerate(distances[i]):
            if all_species[j] == predator and dist < escape_distance:
                if dist < nearest_predator_dist:
                    nearest_predator_dist = dist
                    nearest_predator = j
            elif all_species[j] == prey and dist < chase_distance:
                if dist < nearest_prey_dist:
                    nearest_prey_dist = dist
                    nearest_prey = j

        if nearest_predator != -1:
            escape_dir = positions[i] - positions[nearest_predator]
            escape_norm = np.sqrt(np.sum(escape_dir**2))
            if escape_norm > 0:
                forces[i] += escape_dir / escape_norm * escape_weight

        if nearest_prey != -1:
            chase_dir = positions[nearest_prey] - positions[i]
            chase_norm = np.sqrt(np.sum(chase_dir**2))
            if chase_norm > 0:
                forces[i] += chase_dir / chase_norm * chase_weight

    return forces

class NumbaSimulation:
    def __init__(self, queues):
        self.queues = queues
        self.ui_to_numba_queue = queues['ui_to_numba']
        self.config_manager = ConfigManager()
        self.load_config()
        
        # Profiling properties
        self.profiling_enabled = False
        self.profiling_results = {}
        
    def load_config(self):    
        # ConfigManagerから値を取得してプロパティとして設定
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')

        self.world_size = np.array([self.world_width, self.world_height], dtype=np.float32)
        self.world_center = self.world_size / 2
        self.world_radius = np.min(self.world_size) / 2 + 50
        self.max_force = np.float32(self.config_manager.get_trait_value('MAX_FORCE'))
        self.separation_distance = np.float32(self.config_manager.get_trait_value('SEPARATION_DISTANCE'))
        self.cohesion_distance = np.float32(self.config_manager.get_trait_value('COHESION_DISTANCE'))
        self.separation_weight = np.float32(self.config_manager.get_trait_value('SEPARATION_WEIGHT'))
        self.cohesion_weight = np.float32(self.config_manager.get_trait_value('COHESION_WEIGHT'))
        self.center_attraction_weight = np.float32(self.config_manager.get_trait_value('CENTER_ATTRACTION_WEIGHT'))
        self.rotation_strength = np.float32(self.config_manager.get_trait_value('ROTATION_STRENGTH'))
        self.confinement_weight = np.float32(self.config_manager.get_trait_value('CONFINEMENT_WEIGHT'))

        self.escape_distance = np.float32(self.config_manager.get_trait_value('ESCAPE_DISTANCE'))
        self.escape_weight = np.float32(self.config_manager.get_trait_value('ESCAPE_WEIGHT'))
        self.chase_distance = np.float32(self.config_manager.get_trait_value('CHASE_DISTANCE'))
        self.chase_weight = np.float32(self.config_manager.get_trait_value('CHASE_WEIGHT'))

        # 種別情報の初期化
        self.species = np.zeros(self.max_agents_num, dtype=np.int32)
        self.positions = np.zeros((self.max_agents_num, 2), dtype=np.float32)

        self.predator_species = np.array([self.config_manager.get_species_trait_value('PREDATOR_SPECIES', i) for i in range(1, 9)], dtype=np.int32)
        self.prey_species = np.array([self.config_manager.get_species_trait_value('PREY_SPECIES', i) for i in range(1, 9)], dtype=np.int32)

    def enable_profiling(self):
        self.profiling_enabled = True
        self.profiling_results = {}

    def disable_profiling(self):
        self.profiling_enabled = False

    def profile(func):
        def wrapper(self, *args, **kwargs):
            if not self.profiling_enabled:
                return func(self, *args, **kwargs)

            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()

            func_name = func.__name__
            execution_time = end_time - start_time

            if func_name not in self.profiling_results:
                self.profiling_results[func_name] = []
            self.profiling_results[func_name].append(execution_time)

            return result
        return wrapper

    def get_profiling_results(self):
        if not self.profiling_enabled:
            return "Profiling is not enabled."

        results = "Profiling Results:\n"
        for func_name, times in self.profiling_results.items():
            avg_time = sum(times) / len(times)
            results += f"{func_name}: Average execution time: {avg_time:.6f} seconds\n"
        return results

    def clear_profiling_results(self):
        self.profiling_results.clear()

#----------------- main ----------------------

    def update(self, positions, species):
        self.positions = positions
        self.species = species

    def calculate_forces(self, positions, species):
        if self.profiling_enabled:
            start_time = time.time()
        
        result = self._calculate_forces_numba(
            positions, species,
            self.world_center, self.world_radius,
            self.separation_distance, self.cohesion_distance,
            self.separation_weight, self.cohesion_weight,
            self.center_attraction_weight, self.confinement_weight,
            self.rotation_strength, self.max_force,
            self.predator_species, self.prey_species,
            self.escape_distance, self.chase_distance,
            self.escape_weight, self.chase_weight
        )
        
        if self.profiling_enabled:
            end_time = time.time()
            execution_time = end_time - start_time
            if 'calculate_forces' not in self.profiling_results:
                self.profiling_results['calculate_forces'] = []
            self.profiling_results['calculate_forces'].append(execution_time)
        
        return result

    @staticmethod
    @njit(parallel=True)
    def _calculate_forces_numba(positions, species, 
                                world_center, world_radius, 
                                separation_distance, cohesion_distance, 
                                separation_weight, cohesion_weight,
                                center_attraction_weight, confinement_weight, 
                                rotation_strength, max_force,
                                predator_species, prey_species,
                                escape_distance, chase_distance, 
                                escape_weight, chase_weight):
        distances = calculate_distances(positions)
        
        sep_forces = separation(positions, distances, separation_distance, separation_weight)
        coh_forces = cohesion(positions, distances, cohesion_distance, cohesion_weight)
        env_forces = environment_forces(positions, world_center, world_radius,
                                        center_attraction_weight, confinement_weight,
                                        rotation_strength)
        pred_prey_forces = predator_prey_forces(positions, distances, species, species,
                                                predator_species, prey_species,
                                                escape_distance, chase_distance,
                                                escape_weight, chase_weight)

        total_forces = sep_forces + coh_forces + env_forces + pred_prey_forces
        return limit_magnitude(total_forces, max_force)

    def update_parameters(self):
        while not self.ui_to_numba_queue.empty():
            param_name, value = self.ui_to_numba_queue.get()
            if hasattr(self, param_name):
                setattr(self, param_name, np.float32(value))