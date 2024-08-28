import pygame
import numpy as np
from config import *
from timer import Timer
from creature import Creature

class VisualSystem:
    def __init__(self, queues, running):
        pygame.init()
        self._rendering_queue = queues['rendering_queue']
        self.screen = pygame.display.set_mode((WORLD_WIDTH, WORLD_HEIGHT))
        self.running = running
        self.timer = Timer("Render ")
        self.creatures = {}
        self.rotation_angle = 0
        self.rotation_speed = 0.1  # 1フレームあたりの回転角度（度）
        self.center = np.array([WORLD_WIDTH // 2, WORLD_HEIGHT // 2])

    def create_creature(self, agent_id, agent_species, x, y):
        creature = Creature(agent_species, pygame.Vector2(x, y))
        self.creatures[agent_id] = creature
        return creature

    def draw(self, positions):
        self.screen.fill(BACKGROUND_COLOR)
        
        # 回転行列を作成
        angle_rad = np.radians(self.rotation_angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        for pos in positions:
            # 中心を原点とした座標に変換
            centered_pos = pos - self.center
            # 回転を適用
            rotated_pos = np.dot(rotation_matrix, centered_pos)
            # 画面座標に戻す
            screen_pos = rotated_pos + self.center
            
            pygame.draw.circle(self.screen, AGENT_COLOR, screen_pos.astype(int), AGENT_RADIUS, 1)

        pygame.display.flip()

        # 回転角度を更新
        self.rotation_angle += self.rotation_speed
        if self.rotation_angle >= 360:
            self.rotation_angle -= 360

    def render(self):
        if not self._rendering_queue.empty():
            positions = self._rendering_queue.get()
            self.draw(positions)

    def run(self):
        clock = pygame.time.Clock()
        while self.running.value:
            self.timer.start()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running.value = False
                    return
            self.render()
            self.timer.print_fps(1)
            clock.tick(RENDER_FPS)
        pygame.quit()