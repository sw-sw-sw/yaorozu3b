
# tensorflow_simulation.py
import tensorflow as tf
from config import *
import numpy as np

class TensorFlowSimulation:
    def __init__(self, num_agents, world_width, world_height, queues):
        self.world_size = tf.constant([world_width, world_height], dtype=tf.float32)
        self.num_agents = num_agents
        # Initialize agent data
        self.positions = tf.Variable(tf.random.uniform([num_agents, 2], 0, 1, dtype=tf.float32) * self.world_size)

        # シミュレーションパラメータ
        self.max_force = tf.constant(MAX_FORCE, dtype=tf.float32)
        self.separation_distance = tf.Variable(SEPARATION_DISTANCE, dtype=tf.float32)
        self.cohesion_distance = tf.constant(COHESION_DISTANCE, dtype=tf.float32)
        self.separation_weight = tf.constant(SEPARATION_WEIGHT, dtype=tf.float32)
        self.cohesion_weight = tf.constant(COHESION_WEIGHT, dtype=tf.float32)

    @tf.function
    def _update_positions(self, new_positions):
        self.positions.assign(new_positions)
    
    def update_positions(self, _positions):
        positions = np.frombuffer(_positions.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))
        self._update_positions(tf.constant(positions, dtype=tf.float32))
        
    def apply_force_to_shared_memory(self, _forces):
        new_forces = self.calculate_forces()
        np.frombuffer(_forces.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))[:] = np.array(new_forces)
        
    @tf.function
    def _precompute_distances(self):
        diff = self.positions[:, tf.newaxis, :] - self.positions
        return tf.norm(diff, axis=2)

    #-----------------------------calculate force----------------------------
    
    @tf.function
    def calculate_forces(self):
        distances = self._precompute_distances()
        separation = self._separation(distances)
        cohesion = self._cohesion(distances)
        forces = (
            self.separation_weight * separation +
            self.cohesion_weight * cohesion
        )
        return self._limit_magnitude(forces, self.max_force)

    @tf.function
    def _separation(self, distances):
        mask = tf.logical_and(distances < self.separation_distance, distances > 0)
        mask = tf.cast(mask, tf.float32)
        diff = self.positions[:, tf.newaxis, :] - self.positions
        steer = tf.reduce_sum(diff * mask[:, :, tf.newaxis], axis=1)
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        return tf.where(count > 0, steer / count, 0)


    @tf.function
    def _cohesion(self, distances):
        mask = tf.logical_and(distances < self.cohesion_distance, distances > 0)
        mask = tf.cast(mask, tf.float32)
        center_of_mass = tf.reduce_sum(self.positions * mask[:, :, tf.newaxis], axis=1)
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        center_of_mass = tf.where(count > 0, center_of_mass / count, self.positions)
        return center_of_mass - self.positions

    @tf.function
    def _limit_magnitude(self, vectors, max_magnitude):
        magnitudes = tf.norm(vectors, axis=1, keepdims=True)
        return tf.where(magnitudes > max_magnitude, vectors * max_magnitude / magnitudes, vectors)