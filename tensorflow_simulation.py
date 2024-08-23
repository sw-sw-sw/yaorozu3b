
# tensorflow_simulation.py
import tensorflow as tf
from config import *

class TensorFlowSimulation:
    def __init__(self, num_agents, world_width, world_height):
        self.world_size = tf.constant([world_width, world_height], dtype=tf.float32)
        self.num_agents = num_agents
        
        # Initialize agent data
        self.positions = tf.Variable(tf.random.uniform([num_agents, 2], 0, 1, dtype=tf.float32) * self.world_size)
        self.velocities = tf.Variable(tf.random.uniform([num_agents, 2], INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX, dtype=tf.float32))

        # シミュレーションパラメータ
        self.max_force = tf.constant(MAX_FORCE, dtype=tf.float32)
        self.separation_distance = tf.Variable(SEPARATION_DISTANCE, dtype=tf.float32)
        self.alignment_distance = tf.constant(ALIGNMENT_DISTANCE, dtype=tf.float32)
        self.cohesion_distance = tf.constant(COHESION_DISTANCE, dtype=tf.float32)
        self.separation_weight = tf.constant(SEPARATION_WEIGHT, dtype=tf.float32)
        self.alignment_weight = tf.constant(ALIGNMENT_WEIGHT, dtype=tf.float32)
        self.cohesion_weight = tf.constant(COHESION_WEIGHT, dtype=tf.float32)

    def update_positions(self, new_positions):
        self.positions.assign(new_positions)

    def get_initial_positions(self):
        return self.positions
    
    def get_initial_velocities(self):
        return self.velocities

    @tf.function
    def precompute_distances(self):
        diff = self.positions[:, tf.newaxis, :] - self.positions
        return tf.norm(diff, axis=2)
        # return tf.reduce_sum(tf.abs(diff), axis=2)

    @tf.function
    def calculate_forces(self):
        distances = self.precompute_distances()
        separation = self.separation(distances)
        alignment = self.alignment(distances)
        cohesion = self.cohesion(distances)
        
        forces = (
            self.separation_weight * separation +
            self.alignment_weight * alignment +
            self.cohesion_weight * cohesion
        )
        
        return self.limit_magnitude(forces, self.max_force)

    @tf.function
    def separation(self, distances):
        mask = tf.logical_and(distances < self.separation_distance, distances > 0)
        mask = tf.cast(mask, tf.float32)
        
        diff = self.positions[:, tf.newaxis, :] - self.positions
        steer = tf.reduce_sum(diff * mask[:, :, tf.newaxis], axis=1)
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        
        return tf.where(count > 0, steer / count, 0)

    @tf.function
    def alignment(self, distances):
        mask = tf.logical_and(distances < self.alignment_distance, distances > 0)
        mask = tf.cast(mask, tf.float32)
        
        diff = self.positions[:, tf.newaxis, :] - self.positions
        velocity_estimate = tf.reduce_sum(diff * mask[:, :, tf.newaxis], axis=1)
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        
        return tf.where(count > 0, velocity_estimate / count, 0)

    @tf.function
    def cohesion(self, distances):
        mask = tf.logical_and(distances < self.cohesion_distance, distances > 0)
        mask = tf.cast(mask, tf.float32)
        
        center_of_mass = tf.reduce_sum(self.positions * mask[:, :, tf.newaxis], axis=1)
        count = tf.reduce_sum(mask, axis=1, keepdims=True)
        
        center_of_mass = tf.where(count > 0, center_of_mass / count, self.positions)
        return center_of_mass - self.positions

    @tf.function
    def limit_magnitude(self, vectors, max_magnitude):
        magnitudes = tf.norm(vectors, axis=1, keepdims=True)
        return tf.where(magnitudes > max_magnitude, vectors * max_magnitude / magnitudes, vectors)