import tensorflow as tf
from config import *
from dna_manager import DNAManager

class TensorFlowSimulation:
    def __init__(self, queues):
        self.queues = queues
        self.ui_to_tensorflow_queue = queues['ui_to_tensorflow']
        
        self.world_size = tf.constant([WORLD_WIDTH, WORLD_HEIGHT], dtype=tf.float32)
        self.world_center = self.world_size / 2
        self.world_radius = tf.reduce_min(self.world_size) / 2 - 10
        self.max_force = tf.Variable(MAX_FORCE, dtype=tf.float32)

        # Flocking parameters
        self.separation_distance = tf.Variable(SEPARATION_DISTANCE, dtype=tf.float32)
        self.cohesion_distance = tf.Variable(COHESION_DISTANCE, dtype=tf.float32)
        self.separation_weight = tf.Variable(SEPARATION_WEIGHT, dtype=tf.float32)
        self.cohesion_weight = tf.Variable(COHESION_WEIGHT, dtype=tf.float32)

        # Environmental forces parameters
        self.center_attraction_weight = tf.Variable(CENTER_ATTRACTION_WEIGHT, dtype=tf.float32)
        self.rotation_strength = tf.Variable(ROTATION_STRENGTH, dtype=tf.float32)
        self.confinement_weight = tf.Variable(CONFINEMENT_WEIGHT, dtype=tf.float32)

        self.dna_manager = DNAManager()
        self.escape_distance = tf.constant(self.dna_manager.get_trait_value('ESCAPE_DISTANCE'), dtype=tf.float32)
        self.escape_weight = tf.constant(self.dna_manager.get_trait_value('ESCAPE_WEIGHT'), dtype=tf.float32)
        self.chase_distance = tf.constant(self.dna_manager.get_trait_value('CHASE_DISTANCE'), dtype=tf.float32)
        self.chase_weight = tf.constant(self.dna_manager.get_trait_value('CHASE_WEIGHT'), dtype=tf.float32)

        # 種別情報の初期化
        self.species = tf.Variable(tf.zeros([MAX_AGENTS_NUM], dtype=tf.int32))

    @tf.function
    def calculate_forces(self, positions, species):
        species_forces = self._species_forces(positions, species)
        environment_forces = self._environment_forces(positions)
        return species_forces + environment_forces

    @tf.function
    def _environment_forces(self, positions):
        center_attraction = self._center_attraction(positions)
        circular_confinement = self._circular_confinement(positions)
        to_center, distances = self._calculate_center_distances(positions)
        rotation = self._rotation(to_center, distances)
        forces = (self.center_attraction_weight * center_attraction +
                  self.confinement_weight * circular_confinement +
                  self.rotation_strength * rotation)
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
    
    @tf.function
    def _predator_prey_forces(self, positions, distances, species):
        predator_mask = tf.equal(species[:, tf.newaxis], 
                                 tf.constant(self.dna_manager.get_trait_value('PREDATOR_SPECIES')))
        prey_mask = tf.equal(species[:, tf.newaxis], 
                             tf.constant(self.dna_manager.get_trait_value('PREY_SPECIES')))

        escape_mask = tf.cast(tf.logical_and(distances < self.escape_distance, predator_mask), tf.float32)
        escape_force = tf.reduce_sum(
            (positions[:, tf.newaxis, :] - positions) * escape_mask[:, :, tf.newaxis], 
            axis=1
        ) * self.escape_weight

        chase_mask = tf.cast(tf.logical_and(distances < self.chase_distance, prey_mask), tf.float32)
        chase_force = tf.reduce_sum(
            (positions - positions[:, tf.newaxis, :]) * chase_mask[:, :, tf.newaxis], 
            axis=1
        ) * self.chase_weight

        return escape_force + chase_force


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
    def _center_attraction(self, positions):
        to_center = self.world_center - positions
        return tf.nn.l2_normalize(to_center, axis=1)

    @tf.function
    def _circular_confinement(self, positions):
        to_center, distances = self._calculate_center_distances(positions)
        outside_circle = tf.cast(distances > self.world_radius, tf.float32)
        confinement_force = outside_circle * (distances - self.world_radius) * to_center / distances
        return confinement_force

    @tf.function
    def _rotation(self, to_center, distances):
        inverse_distance = 1.0 / (distances + 1e-5)
        rotation_force = tf.stack([to_center[:, 1], -to_center[:, 0]], axis=1)
        scaled_rotation_force = rotation_force * inverse_distance
        return tf.nn.l2_normalize(scaled_rotation_force, axis=1)

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
