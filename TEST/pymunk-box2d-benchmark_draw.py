'''
テスト結果
Pymunk simulation time: 10.4654 seconds
Box2D simulation time: 6.8628 seconds
Box2D was faster in this test.
Pymunk is 0.66x faster than Box2D
Box2D is 1.52x faster than Pymunk

'''
import pygame
import pymunk
import pymunk.pygame_util
import Box2D
from Box2D.b2 import (world, polygonShape, staticBody, dynamicBody)
import time
import sys
import random

# PyGame initialization
pygame.init()
width, height = 1000, 1000
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Common physics parameters
GRAVITY = 1
BALL_RADIUS = 5
BALL_MASS = 1
BALL_FRICTION = 0.1
WALL_FRICTION = 0.1
BALL_ELASTICITY = 0.9
TIME_STEP = 1.0 / 60.0
VELOCITY_ITERATIONS = 8
POSITION_ITERATIONS = 3
FPS = 10000

def print_progress(engine, step, total_steps):
    percent = (step / total_steps) * 100
    sys.stdout.write(f"\r{engine} Progress: [{step}/{total_steps}] {percent:.1f}%")
    sys.stdout.flush()

def generate_random_positions(num_bodies):
    positions = []
    for _ in range(num_bodies):
        x = random.randint(BALL_RADIUS, width - BALL_RADIUS)
        y = random.randint(BALL_RADIUS, height // 2)  # 上半分に配置
        positions.append((x, y))
    return positions

def create_walls_pymunk(space):
    walls = [
        pymunk.Segment(space.static_body, (0, 0), (width, 0), 5),  # Top
        pymunk.Segment(space.static_body, (0, height), (width, height), 5),  # Bottom
        pymunk.Segment(space.static_body, (0, 0), (0, height), 5),  # Left
        pymunk.Segment(space.static_body, (width, 0), (width, height), 5)  # Right
    ]
    for wall in walls:
        wall.friction = WALL_FRICTION
        wall.color = pygame.Color("darkgreen")
    space.add(*walls)

def pymunk_simulation(num_bodies=100, steps=1000, positions=None):
    space = pymunk.Space()
    space.gravity = (0, GRAVITY * 100)  # Pymunk uses 100 pixels per meter

    create_walls_pymunk(space)

    # Bodies
    for pos in positions:
        moment = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS)
        body = pymunk.Body(BALL_MASS, moment)
        body.position = pos
        shape = pymunk.Circle(body, BALL_RADIUS)
        shape.friction = BALL_FRICTION
        shape.elasticity = BALL_ELASTICITY  # 弾性を設定
        shape.color = pygame.Color("blue")
        space.add(body, shape)


    # 壁の弾性も設定:
    for wall in space.static_body.shapes:
        wall.elasticity = BALL_ELASTICITY

    draw_options = pymunk.pygame_util.DrawOptions(screen)

    start_time = time.time()
    for step in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return time.time() - start_time

        screen.fill((255, 255, 255))
        space.step(TIME_STEP)
        space.debug_draw(draw_options)
        pygame.display.flip()

        if step % 10 == 0:
            print_progress("Pymunk", step, steps)
        clock.tick(FPS)

    end_time = time.time()
    print()
    return end_time - start_time

def create_walls_box2d(world):
    # Create boundary of the world
    world.CreateStaticBody(
        shapes=[
            Box2D.b2EdgeShape(vertices=[(0, 0), (width/100, 0)]),  # Top
            Box2D.b2EdgeShape(vertices=[(0, height/100), (width/100, height/100)]),  # Bottom
            Box2D.b2EdgeShape(vertices=[(0, 0), (0, height/100)]),  # Left
            Box2D.b2EdgeShape(vertices=[(width/100, 0), (width/100, height/100)])  # Right
        ]
    )

def box2d_simulation(num_bodies=100, steps=1000, positions=None):
    world = Box2D.b2World(gravity=(0, GRAVITY))

    create_walls_box2d(world)

    # Bodies
    for pos in positions:
        world.CreateDynamicBody(
            position=(pos[0] / 100, pos[1] / 100),  # Convert to meters
            fixtures=Box2D.b2FixtureDef(
                shape=Box2D.b2CircleShape(radius=BALL_RADIUS / 100),
                density=BALL_MASS / (3.14159 * (BALL_RADIUS / 100) ** 2),
                friction=BALL_FRICTION,
                restitution=BALL_ELASTICITY
            )
        )

    start_time = time.time()
    for step in range(steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return time.time() - start_time

        screen.fill((255, 255, 255))
        world.Step(TIME_STEP, VELOCITY_ITERATIONS, POSITION_ITERATIONS)

        for body in world.bodies:
            for fixture in body.fixtures:
                shape = fixture.shape
                if isinstance(shape, Box2D.b2CircleShape):
                    pos = body.transform * shape.pos * 100
                    pygame.draw.circle(screen, (0, 0, 255), [int(pos.x), int(pos.y)], int(shape.radius * 100))
                elif isinstance(shape, Box2D.b2EdgeShape):
                    vertices = [(body.transform * v) * 100 for v in shape.vertices]
                    pygame.draw.line(screen, (0, 100, 0), vertices[0], vertices[1], 5)

        for body in world.bodies:
            if body.type == Box2D.b2_staticBody:
                for fixture in body.fixtures:
                    fixture.restitution = BALL_ELASTICITY
                    
        pygame.display.flip()

        if step % 10 == 0:
            print_progress("Box2D", step, steps)
        clock.tick(FPS)

    end_time = time.time()
    print()
    return end_time - start_time

if __name__ == "__main__":
    num_bodies = 1000
    steps = 1000

    print(f"Running simulations with {num_bodies} bodies for {steps} steps...")

    # Generate random positions for bodies
    random_positions = generate_random_positions(num_bodies)

    print("\nRunning Pymunk simulation:")
    pymunk_time = pymunk_simulation(num_bodies, steps, random_positions)

    print("\nRunning Box2D simulation:")
    box2d_time = box2d_simulation(num_bodies, steps, random_positions)

    print(f"\nPymunk simulation time: {pymunk_time:.4f} seconds")
    print(f"Box2D simulation time: {box2d_time:.4f} seconds")

    if pymunk_time < box2d_time:
        print("Pymunk was faster in this test.")
    elif box2d_time < pymunk_time:
        print("Box2D was faster in this test.")
    else:
        print("Both engines performed equally in this test.")

    print(f"Pymunk is {box2d_time / pymunk_time:.2f}x faster than Box2D")
    print(f"Box2D is {pymunk_time / box2d_time:.2f}x faster than Pymunk")

    pygame.quit()