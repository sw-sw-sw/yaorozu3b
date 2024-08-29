import math
import random
import noise
import pygame
from pygame import Vector2
from dna_manager import DNAManager, DNASpecies

class Creature:
    def __init__(self, agent_species: int , position: Vector2):
        self.dna_manager = DNAManager()
        self.dna = DNASpecies(self.dna_manager, agent_species)
        self._pos = position
        self._initialize_traits()
        self._initialize_horns()
        self._initialize_shell()
        self._create_surface()


    def _initialize_traits(self):
        self._size = self.dna.get_trait_value("SIZE") 
        self._horn_num = int(self.dna.get_trait_value("HORN_NUM"))
        self._horn_length = self.dna.get_trait_value("HORN_LENGTH")
        self._horn_width = self.dna.get_trait_value("HORN_WIDTH")
        self._shell_size = self.dna.get_trait_value("SHELL_SIZE")
        self._shell_point_size = self.dna.get_trait_value("SHELL_POINT_SIZE")
        self._color = self._get_color_from_dna()
        self._rotate = 0
        self._rotate_v = self.dna.get_trait_value("SPEED") * (random.random() - 0.5)
        
        self._flash = False
        self._move_count = 0
        self._move_cycle = self._initialize_flash_interval()
        
        self._horn_pos_in = []
        self._horn_pos_out = []
        self._shell = []
        self._shell_point_num = 10

    def _initialize_horns(self):
        if self._horn_num > 0:
            for i in range(self._horn_num):
                angle = 2 * math.pi * i / self._horn_num
                x = math.sin(angle) * self._size / 2
                y = math.cos(angle) * self._size / 2
                self._horn_pos_in.append(Vector2(x, y))
                self._horn_pos_out.append(Vector2(x * self._horn_length, y * self._horn_length))

    def _initialize_shell(self):
        if self._shell_size > 0:
            for i in range(self._shell_point_num):
                angle = 2 * math.pi * i / self._shell_point_num
                x = math.sin(angle) * self._size / 2 * self._shell_size
                y = math.cos(angle) * self._size / 2 * self._shell_size
                self._shell.append(Vector2(x, y))

            
    def _initialize_flash_interval(self):
        return random.randint(100, 200)  # Random cycle for direction change
        
    
    def _create_surface(self):

        surface_size = self.get_radius() * 2
        self._surface = pygame.Surface((surface_size, surface_size), pygame.SRCALPHA)
        center = Vector2(surface_size / 2, surface_size / 2)
        # Draw core
        pygame.draw.circle(self._surface, self._color, center, self._size // 2, 1)
        
        # Draw horns
        if self._horn_num > 0:
            for i in range(self._horn_num):
                start = center + self._horn_pos_in[i]
                end = center + self._horn_pos_out[i]
                pygame.draw.line(self._surface, self._color, start, end, int(self._horn_width))

        # Draw shell
        if self._shell_size > 0:
            for point in self._shell:
                pos = center + point
                pygame.draw.circle(self._surface, self._color, pos, self._shell_point_size)

    def update(self, position: Vector2):
        self._pos = position
        self._rotate += self._rotate_v
        
        self._move_count += 1
        if self._move_count > self._move_cycle:
            self._move_count = 0
            self._flash = True

    def draw(self, screen: pygame.Surface):
        rotated_surface = pygame.transform.rotate(self._surface, math.degrees(self._rotate))
        rect = rotated_surface.get_rect(center=(self._pos.x, self._pos.y))
        screen.blit(rotated_surface, rect.topleft)
     
        if self._flash :
            flash_radius = min(self._size, 8)
            pygame.draw.circle(screen, (255, 255, 255), (int(self._pos.x), int(self._pos.y)), int(flash_radius *0.8))
            self._flash = False

    def get_radius(self):
        return max(self._size / 2, self._size * self._shell_size / 2 + self._shell_point_size + 2, self._size * self._horn_length / 2) + 1
    
    def _get_color_from_dna(self):
        color_value = int(self.dna.get_trait_value("COLOR"))
        return pygame.Color(color_value, color_value, color_value)