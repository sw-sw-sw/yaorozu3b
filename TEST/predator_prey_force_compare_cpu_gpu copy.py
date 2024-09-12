import pygame
import numpy as np
import tensorflow as tf
from tensorflow_simulation import TensorFlowSimulation
from timer import Timer
# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 2000, 2000
AGENT_RADIUS = 5
SPECIES_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 128, 128), (255, 128, 0)
]
FORCE_SCALE = 5  # Reduced scale factor for force vector visualization
MAX_VECTOR_LENGTH = 1  # Maximum length of force vectors in pixels

#Set up timer
timer_gpu = Timer('GPU ')
timer_cpu = Timer('   CPU ')
# Set up the display

# Initialize TensorFlowSimulation
mock_queues = {'ui_to_tensorflow': None,'box2d_to_tf':None,'eco_to_tf':None}
tf_sim = TensorFlowSimulation(mock_queues)  # Pass None as we're not using queues

# Initialize agents
num_species = 8
agents_per_species = 50
total_agents = num_species * agents_per_species

positions = np.random.rand(total_agents, 2) * np.array([WIDTH, HEIGHT])
velocities = np.random.randn(total_agents, 2) * 2
species = np.repeat(np.arange(1, num_species + 1), agents_per_species)



# Main game loop
running = True

print("*" * 30)
print("total agent's num", agents_per_species * num_species)
print("*" * 30)
while running:

    tf_positions = tf.convert_to_tensor(positions, dtype=tf.float32)
    tf_species = tf.convert_to_tensor(species, dtype=tf.int32)
    
    timer_gpu.start()
    forces = tf_sim._predator_prey_forces(tf_positions, tf.norm(tf_positions[:, tf.newaxis, :] - tf_positions, axis=2), tf_species)
    timer_gpu.print_average_time(5)
    
    timer_cpu.start()
    forces = tf_sim._predator_prey_forces2(tf_positions, tf.norm(tf_positions[:, tf.newaxis, :] - tf_positions, axis=2), tf_species)
    timer_cpu.print_average_time(5)
    