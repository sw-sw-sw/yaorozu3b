import tensorflow as tf
from config import *
import numpy as np

class TensorFlowSimulation:
    def __init__(self, queues):
        self.world_size = tf.constant([WORLD_WIDTH, WORLD_HEIGHT], dtype=tf.float32)
        self.world_center = self.world_size / 2
        self.world_radius = tf.reduce_min(self.world_size) / 2 - 10
        self.positions = tf.Variable(tf.random.uniform([NUM_AGENTS, 2], 0, 1, dtype=tf.float32) * self.world_size)
        self.max_force = tf.constant(MAX_FORCE, dtype=tf.float32)

        # Flocking parameters
        self.separation_distance = tf.constant(SEPARATION_DISTANCE, dtype=tf.float32)
        self.cohesion_distance = tf.constant(COHESION_DISTANCE, dtype=tf.float32)
        self.separation_weight = tf.constant(SEPARATION_WEIGHT, dtype=tf.float32)
        self.cohesion_weight = tf.constant(COHESION_WEIGHT, dtype=tf.float32)

        # Environmental forces parameters
        self.center_attraction_weight = tf.constant(CENTER_ATTRACTION_WEIGHT, dtype=tf.float32)
        self.rotation_strength = tf.constant(ROTATION_STRENGTH, dtype=tf.float32)
        self.confinement_weight = tf.constant(CONFINEMENT_WEIGHT, dtype=tf.float32)

        # Predator-prey parameters
        self.predator_species = tf.constant([1, 2, 3, 4, 5, 6, 7, 8], dtype=tf.int32)
        self.prey_species = tf.constant([8, 1, 2, 3, 4, 5, 6, 7], dtype=tf.int32)
        self.chase_distance = tf.constant(20.0, dtype=tf.float32)
        self.escape_distance = tf.constant(10.0, dtype=tf.float32)
        self.chase_weight = tf.constant(25.0, dtype=tf.float32)
        self.escape_weight = tf.constant(20.0, dtype=tf.float32)

        self.species = tf.Variable(tf.random.uniform([NUM_AGENTS], minval=1, maxval=9, dtype=tf.int32))

    @tf.function
    def update_positions(self, new_positions):
        self.positions.assign(tf.constant(new_positions, dtype=tf.float32))

    def apply_force_to_shared_memory(self, _forces):
        forces = self.calculate_forces()
        np.frombuffer(_forces.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))[:] = forces.numpy()

    @tf.function
    def calculate_forces(self):
        flocking_forces = self._flocking_forces()
        environment_forces = self._environment_forces()
        predator_prey_forces = self._predator_prey_behavior()
        
        total_forces = flocking_forces + environment_forces + predator_prey_forces
        return self._limit_magnitude(total_forces, self.max_force)

    @tf.function
    def _flocking_forces(self):
        distances = self._calculate_distances()
        separation = self._separation(distances)
        cohesion = self._cohesion(distances)
        return self.separation_weight * separation + self.cohesion_weight * cohesion

    @tf.function
    def _environment_forces(self):
        center_attraction = self._center_attraction()
        circular_confinement = self._circular_confinement()
        to_center, distances = self._calculate_center_distances()
        rotation = self._rotation(to_center, distances)
        return (self.center_attraction_weight * center_attraction +
                self.confinement_weight * circular_confinement +
                self.rotation_strength * rotation)

    @tf.function
    def _predator_prey_behavior(self):
        distances = self._calculate_distances()
        
        chase_forces = tf.zeros_like(self.positions)
        escape_forces = tf.zeros_like(self.positions)
        
        for i in range(8):  # For each species
            predator_mask = tf.cast(tf.equal(self.species, self.predator_species[i]), tf.float32)
            prey_mask = tf.cast(tf.equal(self.species, self.prey_species[i]), tf.float32)
            
            # Chase behavior
            chase_mask = tf.logical_and(
                distances < self.chase_distance,
                tf.expand_dims(predator_mask, 1) * tf.expand_dims(prey_mask, 0)
            )
            chase_diff = self.positions[:, tf.newaxis, :] - self.positions
            chase_forces += tf.reduce_sum(
                chase_diff * tf.cast(chase_mask, tf.float32)[:, :, tf.newaxis],
                axis=1
            ) * tf.expand_dims(predator_mask, 1)
            
            # Escape behavior
            escape_mask = tf.logical_and(
                distances < self.escape_distance,
                tf.expand_dims(prey_mask, 1) * tf.expand_dims(predator_mask, 0)
            )
            escape_diff = self.positions - self.positions[:, tf.newaxis, :]
            escape_forces += tf.reduce_sum(
                escape_diff * tf.cast(escape_mask, tf.float32)[:, :, tf.newaxis],
                axis=1
            ) * tf.expand_dims(prey_mask, 1)
        
        return self.chase_weight * chase_forces + self.escape_weight * escape_forces

    @tf.function
    def _separation(self, distances):
        mask = tf.cast(tf.logical_and(distances < self.separation_distance, distances > 0), tf.float32)
        diff = self.positions[:, tf.newaxis, :] - self.positions
        steer = tf.reduce_sum(diff * mask[:, :, tf.newaxis], axis=1)
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        return tf.where(count > 0, steer / count, 0)

    @tf.function
    def _cohesion(self, distances):
        mask = tf.cast(tf.logical_and(distances < self.cohesion_distance, distances > 0), tf.float32)
        center_of_mass = tf.reduce_sum(self.positions * mask[:, :, tf.newaxis], axis=1)
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        center_of_mass = tf.where(count > 0, center_of_mass / count, self.positions)
        return center_of_mass - self.positions

    @tf.function
    def _center_attraction(self):
        to_center = self.world_center - self.positions
        return tf.nn.l2_normalize(to_center, axis=1)

    @tf.function
    def _circular_confinement(self):
        to_center, distances = self._calculate_center_distances()
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
    def _calculate_center_distances(self):
        to_center = self.world_center - self.positions
        distances = tf.norm(to_center, axis=1, keepdims=True)
        return to_center, distances

    @tf.function
    def _calculate_distances(self):
        return tf.norm(self.positions[:, tf.newaxis, :] - self.positions, axis=2)

    @tf.function
    def _limit_magnitude(self, vectors, max_magnitude):
        magnitudes = tf.norm(vectors, axis=1, keepdims=True)
        scale = tf.minimum(max_magnitude / magnitudes, 1.0)
        return vectors * scale