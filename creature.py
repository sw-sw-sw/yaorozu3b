import math
import random
import noise
import pygame
from pygame import Vector2
from config_manager import ConfigManager, DNASpecies

class Creature(pygame.sprite.Sprite):

    def __init__(self, species: int, position: Vector2):
        super().__init__() 
        self.config_manager = ConfigManager()
        self.species = species
        self.dna: DNASpecies = self.config_manager.get_dna_for_species(species)
        self.position = position
        self._initialize_traits()
        self._initialize_horns()
        self._initialize_shell()
        self.surface_size = self.get_radius() * 2
        self.center = Vector2(self.surface_size / 2, self.surface_size / 2)
        self.image = pygame.Surface((self.surface_size, self.surface_size), pygame.SRCALPHA)
        self._create_base_surface()
        self._create_surface()
        self.rect = self.image.get_rect(center=self.position)


    def _initialize_traits(self):
        self._size = self.dna.get_trait("SIZE")  * self.differ(0) 
        self._horn_num = int(self.dna.get_trait("HORN_NUM"))
        self._horn_length = self.dna.get_trait("HORN_LENGTH")
        self._horn_width = self.dna.get_trait("HORN_WIDTH")
        self._shell_size = self.dna.get_trait("SHELL_SIZE")
        self._shell_point_size = self.dna.get_trait("SHELL_POINT_SIZE")
        self._color = self._get_color_from_dna()
        self._rotate = 0
        self._rotate_v = self.dna.get_trait("SPEED")  * self.differ(1) * 0.1
        self._rotate_count = 0
        self._last_rotate = 0
        self._flash = False
        self._flash_count = 0
        self._flash_cycle = self._initialize_flash_interval()
        self._flash_radius = self.dna.get_trait("RADIUS") * 0.5
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
        return random.randint(20, 100)  # Random cycle for direction change
    
    def _create_surface(self):
        self.image = self.base_image.copy()
            # Draw flash circle if _flash is True
        if self._flash:
            pygame.draw.circle(self.image, (255, 255, 255), self.center, self._flash_radius)
        
    def _create_base_surface(self):
        self.base_image = pygame.Surface((self.surface_size, self.surface_size), pygame.SRCALPHA)

       # Draw core
        pygame.draw.circle(self.base_image, self._color, self.center, self._size / 2, 1)
        
        # Draw horns
        if self._horn_num > 0:
            for i in range(self._horn_num):
                start = self.center + self._horn_pos_in[i]
                end = self.center + self._horn_pos_out[i]
                pygame.draw.line(self.base_image, self._color, start, end, int(self._horn_width))

        # Draw shell
        if self._shell_size > 0:
            for point in self._shell:
                pos = self.center + point
                pygame.draw.circle(self.base_image, self._color, pos, self._shell_point_size)


    def update(self, new_position=None):
        if new_position is not None:
            self.position.update(new_position[0], new_position[1])
        
        self._rotate += self._rotate_v
        
        self._flash_count += 1
        if self._flash_count == self._flash_cycle:
            self._flash = True
            self._flash_count = 0
        else:
            self._flash = False

        self._create_surface()
        self.rect.center = self.position

        if abs(self._rotate - self._last_rotate) > 0.1:
                self.image = pygame.transform.rotate(self.image, self._rotate)
                self._last_rotate = abs(self._rotate)    

        self.rect = self.image.get_rect(center=self.position)

    def get_radius(self):
        return max(self._size / 2, 
                   self._size * self._shell_size / 2 + self._shell_point_size + 2, 
                   self._size * self._horn_length / 2) + 1
    
    def _get_color_from_dna(self):
        color_value = int(self.dna.get_trait("COLOR"))
        return pygame.Color(color_value, color_value, color_value)
    
    
    def differ(self, rate = 0.2):
        return 1 + random.random() * rate 