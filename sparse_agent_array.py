import tensorflow as tf
import numpy as np

def get_tf_positions(shared_memory):
    # with shared_memory['lock']:
    np_array = np.frombuffer(shared_memory['positions'].get_obj(), dtype=np.float32)
    positions = np_array.reshape((-1, 2))
    return tf.convert_to_tensor(positions, dtype=tf.float32)

def get_tf_species(shared_memory):
    # with shared_memory['lock']:
    species = np.frombuffer(shared_memory['agent_species'].get_obj(), dtype=np.int32)
    return tf.convert_to_tensor(species, dtype=tf.int32)

def set_forces(shared_memory, forces):
    # with shared_memory['lock']:
    np_array = np.frombuffer(shared_memory['forces'].get_obj(), dtype=np.float32)
    np_array[:] = forces.flatten()