import pygame
import sys
import time
from death_effect import DeathEffect

pygame.init()

WIDTH, HEIGHT = 1200, 1200
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Death Effect Test")
BLACK = (0, 0, 0)
clock = pygame.time.Clock()

def main():
    effects = []
    last_effect_time = time.time()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_time = time.time()
        if current_time - last_effect_time >= 1:
            x = pygame.mouse.get_pos()[0]
            y = pygame.mouse.get_pos()[1]
            new_effect = DeathEffect(pygame.Vector2(x, y))
            effects.append(new_effect)
            last_effect_time = current_time

        screen.fill(BLACK)  # Gray background

        for effect in effects[:]:
            effect.update()
            effect.draw(screen)
            if effect.is_finished():
                effects.remove(effect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()