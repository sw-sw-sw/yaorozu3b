import tensorflow as tf
from config import *
import numpy as np
from dna_manager import DNAManager

class TensorFlowSimulation:
    def __init__(self, queues):
        self.dna_manager = DNAManager()
        # ... 既存の初期化コード ...

        # 新しいパラメータの初期化
        self.escape_distance = tf.constant(self.dna_manager.get_trait_value('ESCAPE_DISTANCE'), dtype=tf.float32)
        self.escape_weight = tf.constant(self.dna_manager.get_trait_value('ESCAPE_WEIGHT'), dtype=tf.float32)
        self.chase_distance = tf.constant(self.dna_manager.get_trait_value('CHASE_DISTANCE'), dtype=tf.float32)
        self.chase_weight = tf.constant(self.dna_manager.get_trait_value('CHASE_WEIGHT'), dtype=tf.float32)

        # 種別情報の初期化
        self.species = tf.Variable(tf.zeros([MAX_AGENTS_NUM], dtype=tf.int32))

    def update_positions(self, _positions):
        # ... 既存のコード ...

    def update_species(self, _species):
        species = np.frombuffer(_species.get_obj(), dtype=np.int32)
        self.species.assign(tf.constant(species, dtype=tf.int32))

    def apply_force_to_shared_memory(self, _forces):
        # ... 既存のコード ...

    @tf.function
    def _predator_prey_forces(self):
        distances = self._calculate_distances()
        
        # 捕食者と獲物の関係を決定
        predator_mask = tf.equal(self.species[:, tf.newaxis], 
                                 tf.constant(self.dna_manager.get_trait_value('PREDATOR_SPECIES')))
        prey_mask = tf.equal(self.species[:, tf.newaxis], 
                             tf.constant(self.dna_manager.get_trait_value('PREY_SPECIES')))

        # 逃避力の計算
        escape_mask = tf.cast(tf.logical_and(distances < self.escape_distance, predator_mask), tf.float32)
        escape_force = tf.reduce_sum(
            (self.positions[:, tf.newaxis, :] - self.positions) * escape_mask[:, :, tf.newaxis], 
            axis=1
        ) * self.escape_weight

        # 追跡力の計算
        chase_mask = tf.cast(tf.logical_and(distances < self.chase_distance, prey_mask), tf.float32)
        chase_force = tf.reduce_sum(
            (self.positions - self.positions[:, tf.newaxis, :]) * chase_mask[:, :, tf.newaxis], 
            axis=1
        ) * self.chase_weight

        return escape_force + chase_force

    @tf.function
    def calculate_forces(self):
        species_forces = self._species_forces()
        environment_forces = self._environment_forces()
        predator_prey_forces = self._predator_prey_forces()
        
        total_forces = species_forces + environment_forces + predator_prey_forces
        return self._limit_magnitude(total_forces, self.max_force)

    # ... その他の既存のメソッド ...