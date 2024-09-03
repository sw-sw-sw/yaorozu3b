import tensorflow as tf
from config_manager import ConfigManager

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
        self.world_radius = tf.reduce_min(self.world_size) / 2 - 10
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
    
    @tf.function
    def calculate_forces(self, positions, species):
        species_forces = self._species_forces(positions, species)
        environment_forces = self._environment_forces(positions)
        return species_forces + environment_forces

    @tf.function
    def _environment_forces(self, positions):
        
        # center_attraction
        to_center, distances = self._calculate_center_distances(positions)
        center_attraction = self.center_attraction_weight * tf.nn.l2_normalize(to_center, axis=1)
        
        # circular_confinement
        outside_circle = tf.cast(distances > self.world_radius, tf.float32)
        circular_confinement = self.confinement_weight * outside_circle * (distances - self.world_radius) * to_center / distances
        
        # rotation_force
        inverse_distance = 1.0 / (distances + 1e-5)
        rotation_force = tf.stack([to_center[:, 1], -to_center[:, 0]], axis=1)
        rotation = self.rotation_strength * tf.nn.l2_normalize(rotation_force * inverse_distance, axis=1)
        
        return center_attraction + circular_confinement + rotation
    
    @tf.function
    def _species_forces(self, positions, species):
        distances = self._calculate_distances(positions)
        separation = self._separation(positions, distances)
        cohesion = self._cohesion(positions, distances)
        predator_prey = self._predator_prey_forces(positions, distances, species)
        forces = (self.separation_weight * separation +
                  self.cohesion_weight * cohesion + predator_prey)
        
        return self._limit_magnitude(forces, self.max_force)
    
    @tf.function
    def _predator_prey_forces(self, positions, distances, species):
        num_agents = tf.shape(positions)[0]
        num_species = 8  # 1から8までの種

        # 各種の捕食者と獲物のマスクを一度に作成
        species_mask = tf.equal(tf.expand_dims(species, 0), tf.range(1, num_species + 1, dtype=tf.int32)[:, tf.newaxis])
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

        # 種ごとの力を合計
        total_force = escape_force + chase_force

        return total_force
    
    @tf.function
    def _separation(self, positions, distances):
        mask = tf.cast(tf.logical_and(distances < self.separation_distance, distances > 0), tf.float32)
        diff = positions[:, tf.newaxis, :] - positions
        steer = tf.reduce_sum(diff * mask[:, :, tf.newaxis], axis=1)
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        return tf.where(count > 0, steer / count, 0)

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