import tensorflow as tf
from config_manager import ConfigManager
import time

class TensorFlowSimulation:
    def __init__(self, queues):
        self.queues = queues
        self.ui_to_tensorflow_queue = queues['ui_to_tensorflow']
        self.config_manager = ConfigManager()
        
        # ConfigManagerから値を取得してプロパティとして設定
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')

        self.world_size = tf.constant([self.world_width, self.world_height], dtype=tf.float32)
        self.world_center = self.world_size / 2
        self.world_radius = tf.reduce_min(self.world_size) / 2 + 50
        self.max_force = tf.Variable(self.config_manager.get_trait_value('MAX_FORCE'), dtype=tf.float32)
        self.separation_distance = tf.Variable(self.config_manager.get_trait_value('SEPARATION_DISTANCE'), dtype=tf.float32)
        self.cohesion_distance = tf.Variable(self.config_manager.get_trait_value('COHESION_DISTANCE'), dtype=tf.float32)
        self.separation_weight = tf.Variable(self.config_manager.get_trait_value('SEPARATION_WEIGHT'), dtype=tf.float32)
        self.cohesion_weight = tf.Variable(self.config_manager.get_trait_value('COHESION_WEIGHT'), dtype=tf.float32)
        self.center_attraction_weight = tf.Variable(self.config_manager.get_trait_value('CENTER_ATTRACTION_WEIGHT'), dtype=tf.float32)
        self.rotation_strength = tf.Variable(self.config_manager.get_trait_value('ROTATION_STRENGTH'), dtype=tf.float32)
        self.confinement_weight = tf.Variable(self.config_manager.get_trait_value('CONFINEMENT_WEIGHT'), dtype=tf.float32)

        self.escape_distance = tf.Variable(self.config_manager.get_trait_value('ESCAPE_DISTANCE'), dtype=tf.float32)
        self.escape_weight = tf.Variable(self.config_manager.get_trait_value('ESCAPE_WEIGHT'), dtype=tf.float32)
        self.chase_distance = tf.Variable(self.config_manager.get_trait_value('CHASE_DISTANCE'), dtype=tf.float32)
        self.chase_weight = tf.Variable(self.config_manager.get_trait_value('CHASE_WEIGHT'), dtype=tf.float32)

        # 種別情報の初期化
        self.species = tf.Variable(tf.zeros([self.max_agents_num], dtype=tf.int32))
        self.predator_species = tf.constant([self.config_manager.get_species_trait_value('PREDATOR_SPECIES', i) for i in range(1, 9)], dtype=tf.int32)
        self.prey_species = tf.constant([self.config_manager.get_species_trait_value('PREY_SPECIES', i) for i in range(1, 9)], dtype=tf.int32)
    
        # Profiling properties
        self.profiling_enabled = False
        self.profiling_results = {}

    #------------------for profiling---------------------
    
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
        
    # ---------------------------------------
    
    @tf.function
    def calculate_forces(self, positions, species):
        species_forces = self._species_forces(positions, species)
        environment_forces = self._environment_forces(positions)
        return species_forces + environment_forces

    @profile
    @tf.function
    def _environment_forces(self, positions):
        to_center = self.world_center - positions
        distances = tf.norm(to_center, axis=1, keepdims=True)
        normalized_to_center = to_center / (distances + 1e-5)
        
        # Center attraction (always applied)
        center_force = self.center_attraction_weight * normalized_to_center
        
        # Circular confinement (only applied outside the world radius)
        outside_circle = tf.cast(distances > self.world_radius, tf.float32)
        confinement_force = outside_circle * self.confinement_weight * (distances - self.world_radius) * normalized_to_center
        
        # Create perpendicular vector for rotation (counter-clockwise)
        rotation_force = tf.stack([-to_center[:, 1], to_center[:, 0]], axis=1)
        rotation_force = tf.nn.l2_normalize(rotation_force, axis=1)
        rotation_force *= self.rotation_strength
        
        # Combine all forces
        forces = center_force + confinement_force + rotation_force
        return forces

    @tf.function
    def _species_forces(self, positions, species):
        distances = self._calculate_distances(positions)
        separation = self._separation(positions, distances)
        cohesion = self._cohesion(positions, distances)
        predator_prey = self._predator_prey_forces(positions, distances, species)
        forces = (self.separation_weight * separation +
                  self.cohesion_weight * cohesion + predator_prey)
        
        return self._limit_magnitude(forces, self.max_force)
 
    @profile
    @tf.function
    def _predator_prey_forces(self, positions, distances, species):
        num_agents = tf.shape(positions)[0]
        forces = tf.zeros_like(positions)

        # 種ごとの捕食者と獲物のインデックスを事前計算
        predator_indices = tf.gather(self.predator_species, species - 1)
        prey_indices = tf.gather(self.prey_species, species - 1)

        # 各エージェントの種、捕食者、獲物の種を一度に取得
        species_mask = tf.equal(tf.expand_dims(species, 0), tf.range(1, 9, dtype=tf.int32)[:, tf.newaxis])
        predator_mask = tf.equal(tf.expand_dims(species, 1), predator_indices)
        prey_mask = tf.equal(tf.expand_dims(species, 1), prey_indices)

        # 逃避と追跡の距離マスクを作成
        escape_mask = tf.logical_and(predator_mask, distances < self.escape_distance)
        chase_mask = tf.logical_and(prey_mask, distances < self.chase_distance)

        # 位置の差分を一度に計算
        pos_diff = tf.expand_dims(positions, 1) - positions

        # 逃避力の計算
        escape_directions = tf.where(
            tf.expand_dims(escape_mask, -1),
            pos_diff,
            tf.zeros_like(pos_diff)
        )
        escape_force = tf.reduce_sum(escape_directions, axis=1)
        escape_force = tf.where(
            tf.reduce_any(escape_mask, axis=1, keepdims=True),
            tf.nn.l2_normalize(escape_force, axis=1) * self.escape_weight,
            tf.zeros_like(escape_force)
        )

        # 追跡力の計算
        chase_directions = tf.where(
            tf.expand_dims(chase_mask, -1),
            -pos_diff,
            tf.zeros_like(pos_diff)
        )
        chase_force = tf.reduce_sum(chase_directions, axis=1)
        chase_force = tf.where(
            tf.reduce_any(chase_mask, axis=1, keepdims=True),
            tf.nn.l2_normalize(chase_force, axis=1) * self.chase_weight,
            tf.zeros_like(chase_force)
        )

        # 合計力の計算
        total_force = escape_force + chase_force

        return total_force
   
    # @profile
    # @tf.function
    # def _predator_prey_forces(self, positions, distances, species):
    #     num_agents = tf.shape(positions)[0]
    #     num_species = 8  # 1から8までの種

    #     # 各種の捕食者と獲物のマスクを一度に作成
    #     species_mask = tf.equal(tf.expand_dims(species, 0), tf.range(1, num_species + 1, dtype=tf.int32)[:, tf.newaxis])
    #     predator_mask = tf.equal(tf.expand_dims(species, 1), tf.gather(self.predator_species, species - 1))
    #     prey_mask = tf.equal(tf.expand_dims(species, 1), tf.gather(self.prey_species, species - 1))

    #     # 逃避と追跡のマスクを作成
    #     escape_mask = tf.logical_and(predator_mask, distances < self.escape_distance)
    #     chase_mask = tf.logical_and(prey_mask, distances < self.chase_distance)

    #     # 最も近い捕食者と獲物を見つける
    #     escape_distances = tf.where(escape_mask, distances, tf.fill(tf.shape(distances), tf.float32.max))
    #     chase_distances = tf.where(chase_mask, distances, tf.fill(tf.shape(distances), tf.float32.max))
    #     nearest_predator = tf.argmin(escape_distances, axis=1)
    #     nearest_prey = tf.argmin(chase_distances, axis=1)

    #     # 逃避力と追跡力を計算
    #     escape_direction = positions - tf.gather(positions, nearest_predator)
    #     chase_direction = tf.gather(positions, nearest_prey) - positions

    #     escape_force = tf.where(
    #         tf.reduce_any(escape_mask, axis=1, keepdims=True),
    #         tf.nn.l2_normalize(escape_direction, axis=1) * self.escape_weight,
    #         tf.zeros_like(positions)
    #     )

    #     chase_force = tf.where(
    #         tf.reduce_any(chase_mask, axis=1, keepdims=True),
    #         tf.nn.l2_normalize(chase_direction, axis=1) * self.chase_weight,
    #         tf.zeros_like(positions)
    #     )

    #     # 種ごとの力を合計
    #     total_force = escape_force + chase_force

    #     return total_force
    
    @profile
    @tf.function
    def _separation(self, positions, distances):
        mask = tf.cast(tf.logical_and(distances < self.separation_distance, distances > 0), tf.float32)
        diff = positions[:, tf.newaxis, :] - positions
        steer = tf.reduce_sum(diff * mask[:, :, tf.newaxis], axis=1)
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        return tf.where(count > 0, steer / count, 0)

    @profile
    @tf.function
    def _cohesion(self, positions, distances):
        mask = tf.cast(tf.logical_and(distances < self.cohesion_distance, distances > 0), tf.float32)
        center_of_mass = tf.reduce_sum(positions * mask[:, :, tf.newaxis], axis=1)
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        center_of_mass = tf.where(count > 0, center_of_mass / count, positions)
        return center_of_mass - positions
    
 
    @profile
    @tf.function
    def _calculate_center_distances(self, positions):
        to_center = self.world_center - positions
        distances = tf.norm(to_center, axis=1, keepdims=True)
        return to_center, distances

    @profile
    @tf.function
    def _calculate_distances(self, positions):
        return tf.norm(positions[:, tf.newaxis, :] - positions, axis=2)

    
    @tf.function
    def _limit_magnitude(self, vectors, max_magnitude):
        magnitudes = tf.norm(vectors, axis=1, keepdims=True)
        scale = tf.minimum(max_magnitude / magnitudes, 1.0)
        return vectors * scale

    def update_species(self, species):
        self.species.assign(species)

    def update_parameters(self):
        while not self.ui_to_tensorflow_queue.empty():
            param_name, value = self.ui_to_tensorflow_queue.get()
            if hasattr(self, param_name):
                getattr(self, param_name).assign(value)
