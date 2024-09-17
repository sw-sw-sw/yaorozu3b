import tensorflow as tf
from config_manager import ConfigManager
import time
import numpy as np
from log import get_logger
from queue import Empty

logger = get_logger()

class TensorFlowSimulation:
    def __init__(self, queues, max_agents=None):
        logger.info("Initializing TensorFlowSimulation")
        # queue setting
        self.queues = queues
        self._ui_to_tensorflow_queue = queues['ui_to_tensorflow']
        self._box2d_to_tf = queues['box2d_to_tf']
        self._eco_to_tf_init = queues['eco_to_tf_init']
        self._eco_to_tf = queues['eco_to_tf']
        self._tf_to_box2d = queues['tf_to_box2d']
        self.config_manager = ConfigManager()
        
        # ConfigManagerから値を取得してプロパティとして設定
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')
        # for profiling test
        if max_agents is None:
            self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        else:
            self.max_agents_num = max_agents
            
        self.world_size = tf.constant([self.world_width, self.world_height], dtype=tf.float32)
        self.world_center = self.world_size / 2
        self.world_radius = tf.reduce_min(self.world_size) / 2 + 50

        # Profiling properties
        self.profiling_enabled = False
        self.profiling_results = {}
        
        # Initialize simulation parameters as tf.Variables
        self._init_simulation_parameters()

        # Initialize agent data as tf.Variables
        self.tf_positions = tf.Variable(tf.zeros((self.max_agents_num, 2), dtype=tf.float32))
        self.tf_species = tf.Variable(tf.zeros([self.max_agents_num], dtype=tf.int32))
        self.tf_current_agent_count = tf.Variable(0, dtype=tf.int32)
        self.tf_forces = tf.Variable(tf.zeros((self.max_agents_num, 2), dtype=tf.float32))

        # Initialize species information
        self._init_species_information()
        self.initialized = False
        logger.info("TensorFlowSimulation initialization completed")

    def _init_simulation_parameters(self):
        logger.debug("Initializing simulation parameters")
        param_names = [
            'MAX_FORCE', 'SEPARATION_DISTANCE', 'COHESION_DISTANCE', 'SEPARATION_WEIGHT',
            'COHESION_WEIGHT', 'CENTER_ATTRACTION_WEIGHT', 'ROTATION_STRENGTH',
            'CONFINEMENT_WEIGHT', 'ESCAPE_DISTANCE', 'ESCAPE_WEIGHT', 'CHASE_DISTANCE',
            'CHASE_WEIGHT','PREDATOR_PREY_WEIGHT'
        ]
        for param in param_names:
            setattr(self, param.lower(), tf.Variable(
                self.config_manager.get_trait_value(param), dtype=tf.float32
            ))
        logger.debug("Simulation parameters initialized")

    def _init_species_information(self):
        logger.debug("Initializing species information")
        self.predator_species = tf.constant([
            self.config_manager.get_species_trait_value('PREDATOR_SPECIES', i) for i in range(1, 9)
        ], dtype=tf.int32)
        self.prey_species = tf.constant([
            self.config_manager.get_species_trait_value('PREY_SPECIES', i) for i in range(1, 9)
        ], dtype=tf.int32)
        
        logger.debug("Species information initialized")
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
        
    # ---------------- property update -----------------------
    def initialize(self):
        logger.info("TensorFlowSimulation is initializing")

        while not self.initialized:
            try:
                data = self._eco_to_tf_init.get(timeout=0.1)
                self.tf_current_agent_count.assign(tf.convert_to_tensor(data['current_agent_count'], dtype=tf.int32))
                self.tf_positions.assign(tf.convert_to_tensor(data['positions'], dtype=tf.float32))
                self.tf_species.assign(tf.convert_to_tensor(data['species'], dtype=tf.int32))
                self.initialized = True
                logger.info(f"TensorFlowSimulation Initialized with {self.tf_current_agent_count.numpy()} agents")
            except Empty:
                logger.warning("Waiting for initialization data from Ecosystem")
                continue  # Queue is empty, continue waiting
        
        logger.info("TensorFlowSimulation initialized successfully")

    def update(self):
        self.update_property()
        forces = self.calculate_forces()
        self.send_forces_to_box2d(forces.numpy()[:])
        self.update_ui_parameters()
        
    def update_property(self):
        while True:
            try:
                data = self._box2d_to_tf.get_nowait()
                new_count = tf.convert_to_tensor(data['current_agent_count'], dtype=tf.int32)
                new_positions = tf.convert_to_tensor(data['positions'], dtype=tf.float32)
                new_species = tf.convert_to_tensor(data['species'], dtype=tf.int32)
                mask = tf.range(self.max_agents_num) < new_count
                mask = tf.reshape(mask, (-1, 1))
                self.tf_positions.assign(tf.where(mask, new_positions, tf.zeros_like(new_positions)))
                self.tf_species.assign(tf.where(mask[:, 0], new_species, tf.zeros_like(new_species)))
                self.tf_current_agent_count.assign(new_count)
            except Empty:
                break

    def send_forces_to_box2d(self, np_forces):
        data = {
            'forces': np_forces,
            'current_agent_count': int(self.tf_current_agent_count.numpy())
        }
        self._tf_to_box2d.put(data)
        logger.debug(f"Sent forces to Box2D for {data['current_agent_count']} agents")

    @tf.function
    def calculate_forces(self):
        species_forces = self._limit_magnitude(self._species_forces(self.tf_positions, self.tf_species))
        environment_forces = self._limit_magnitude(self._environment_forces(self.tf_positions))
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
        return self._limit_magnitude(forces)

    @tf.function
    def _species_forces(self, positions, species):
        distances = self._calculate_distances(positions)
        separation = self._separation(positions, distances)
        cohesion = self._cohesion(positions, distances)
        predator_prey = self._predator_prey_forces(positions, distances, species)
        forces = (self.separation_weight * separation +
                  self.cohesion_weight * cohesion + predator_prey * self.predator_prey_weight)
        
        return self._limit_magnitude(forces)

    @profile
    @tf.function
    def _predator_prey_forces(self, positions, distances, species):
        num_agents = tf.shape(positions)[0]
        
        # 捕食者と被捕食種のマスクを作成
        predator_mask = tf.equal(tf.expand_dims(species, 1), tf.gather(self.predator_species, species - 1))
        prey_mask = tf.equal(tf.expand_dims(species, 1), tf.gather(self.prey_species, species - 1))
        
        # 逃避と追跡のマスクを作成
        escape_mask = tf.logical_and(predator_mask, distances < self.escape_distance)
        chase_mask = tf.logical_and(prey_mask, distances < self.chase_distance)

        # 最も近い捕食者と獲物を見つける
        escape_distances = tf.where(escape_mask, distances, tf.fill(tf.shape(distances), tf.float32.max))
        chase_distances = tf.where(chase_mask, distances, tf.fill(tf.shape(distances), tf.float32.max))
        nearest_predator = tf.argmin(escape_distances, axis=1)
        nearest_prey = tf.argmin(chase_distances, axis=1)

        # 逃避力と追跡力を計算
        escape_direction = positions - tf.gather(positions, nearest_predator)
        chase_direction = tf.gather(positions, nearest_prey) - positions

        escape_force = tf.where(
            tf.reduce_any(escape_mask, axis=1, keepdims=True),
            tf.nn.l2_normalize(escape_direction, axis=1) * self.escape_weight,
            tf.zeros_like(positions)
        )
        chase_force = tf.where(
            tf.reduce_any(chase_mask, axis=1, keepdims=True),
            tf.nn.l2_normalize(chase_direction, axis=1) * self.chase_weight,
            tf.zeros_like(positions)
        )
        total_force = escape_force + chase_force


        return total_force


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
    
 

    @tf.function
    def _calculate_center_distances(self, positions):
        to_center = self.world_center - positions
        distances = tf.norm(to_center, axis=1, keepdims=True)
        return to_center, distances

    
    @tf.function
    def _calculate_distances(self, positions):
        return tf.norm(positions[:, tf.newaxis, :] - positions, axis=2)

    
    @tf.function
    def _limit_magnitude(self, vectors):
        magnitudes = tf.norm(vectors, axis=1, keepdims=True)
        scale = tf.minimum(self.max_force / magnitudes, 1.0)
        return vectors * scale

    def update_ui_parameters(self):
        while not self._ui_to_tensorflow_queue.empty():
            try:
                param_name, value = self._ui_to_tensorflow_queue.get_nowait()
                if hasattr(self, param_name.lower()):
                    getattr(self, param_name.lower()).assign(value)
                    logger.debug(f"Updated UI parameter: {param_name} = {value}")
            except Empty:
                break  # Queue is empty, exit the loop