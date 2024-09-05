import pygame
import numpy as np
from numba import jit
from timer import Timer
from predator_prey_force_numba_class import NumbaSimulation
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
FORCE_SCALE = 5
MAX_VECTOR_LENGTH = 1

timer = Timer('Numba')
nmb = NumbaSimulation()
# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Predator-Prey Forces Test (Numba Version)")

# Initialize agents
num_species = 8
agents_per_species = 50
total_agents = num_species * agents_per_species

positions = np.random.rand(total_agents, 2) * np.array([WIDTH, HEIGHT])
velocities = np.random.randn(total_agents, 2) * 2
species = np.repeat(np.arange(1, num_species + 1), agents_per_species)

# Predator-prey relationships (adjust as needed)
predator_species = np.array([2, 3, 4, 5, 6, 7, 8, 1])
prey_species = np.array([8, 1, 2, 3, 4, 5, 6, 7])

# Parameters
escape_distance = 100.0
chase_distance = 150.0
escape_weight = 1.0
chase_weight = 0.8

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
    forces = nmb.calculate_forces(positions, species, predator_species, prey_species, 
                                  escape_distance, chase_distance, escape_weight, chase_weight)
    timer.print_average_time(1)
    # Update velocities and positions
    velocities += forces * 0.1
    velocities *= 0.99
    positions += velocities * 0.1

    # Wrap around screen edges
    positions %= np.array([WIDTH, HEIGHT])

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw agents and force vectors
    for pos, force, spec in zip(positions, forces, species):
        color = SPECIES_COLORS[spec - 1]
        pygame.draw.circle(screen, color, pos.astype(int), AGENT_RADIUS, 1)
        draw_force_vector(screen, pos, force, color)

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(60)

pygame.quit()