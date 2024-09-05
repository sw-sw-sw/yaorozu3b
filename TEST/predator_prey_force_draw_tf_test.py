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
MAX_VECTOR_LENGTH = 3  # Maximum length of force vectors in pixels

#Set up timer
timer = Timer('tensorflow')

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Predator-Prey Forces Test")

# Initialize TensorFlowSimulation
mock_queues = {'ui_to_tensorflow': None}
tf_sim = TensorFlowSimulation(mock_queues)  # Pass None as we're not using queues

# Initialize agents
num_species = 8
agents_per_species = 375
total_agents = num_species * agents_per_species

positions = np.random.rand(total_agents, 2) * np.array([WIDTH, HEIGHT])
velocities = np.random.randn(total_agents, 2) * 2
species = np.repeat(np.arange(1, num_species + 1), agents_per_species)

# Function to limit vector length
def limit_vector_length(vector, max_length):
    length = np.linalg.norm(vector)
    if length > max_length:
        return vector * (max_length / length)
    return vector

# Function to draw a force vector
def draw_force_vector(surface, start_pos, force, color):
    scaled_force = limit_vector_length(force * FORCE_SCALE, MAX_VECTOR_LENGTH)
    end_pos = start_pos + scaled_force * 20
    pygame.draw.line(surface, color, start_pos.astype(int), end_pos.astype(int), 1)


# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    timer.start()
    tf_positions = tf.convert_to_tensor(positions, dtype=tf.float32)
    tf_species = tf.convert_to_tensor(species, dtype=tf.int32)
    forces = tf_sim._predator_prey_forces(tf_positions, tf.norm(tf_positions[:, tf.newaxis, :] - tf_positions, axis=2), tf_species)
    forces_np = forces.numpy()
    timer.print_average_time(1)
    
    # Update velocities and positions
    velocities += forces_np * 0.01  # Adjust the multiplier to control force strength
    velocities *= 0.99  # Add some damping
    positions += velocities * 0.1  # Adjust the multiplier to control speed

    # Wrap around screen edges
    positions %= np.array([WIDTH, HEIGHT])

    # Draw agents and force vectors
    screen.fill((0, 0, 0))
    for pos, force, spec in zip(positions, forces_np, species):
        color = SPECIES_COLORS[spec - 1]
        pygame.draw.circle(screen, color, pos.astype(int), AGENT_RADIUS, 1)
        draw_force_vector(screen, pos, force, color)
    pygame.display.flip()
    clock.tick(60)
pygame.quit()