import pygame
import random
from pygame import Vector2
from creature import Creature
import noise

# Initialize Pygame
pygame.init()

# Set up the display
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Creature Display Test")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Create a list to store creatures
creatures = []

# Create multiple creatures of different species
NUM_CREATURES = 50
for _ in range(NUM_CREATURES):
    species = random.randint(1, 8)  # Random species from 1 to 8
    position = Vector2(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT))
    creature = Creature(species, position)
    creatures.append(creature)

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(BLACK)

    # Update and draw creatures
    for creature in creatures:
        creature.update(creature._pos)  # Update with current position
        creature.draw(screen)


    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()