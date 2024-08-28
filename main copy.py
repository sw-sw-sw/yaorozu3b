# main.py
# new project

import tensorflow as tf
import time
import numpy as np
import pygame
from Box2D import b2World, b2Vec2
from config import *
import multiprocessing as mp


def tf_loop(shared_positions, shared_velocities, shared_forces, num_agents, world_width, world_height, running, tf_time, box2d_time):
    import tensorflow as tf
    from tensorflow_simulation import TensorFlowSimulation
    from timer import Timer
    tf_sim = TensorFlowSimulation(num_agents, world_width, world_height)
    timer = Timer("Tensor ")
    
    while running.value:
        timer.start()        
        positions = np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape((num_agents, 2))
        # velocities = np.frombuffer(shared_velocities.get_obj(), dtype=np.float32).reshape((num_agents, 2))

        tf_sim.update_positions(positions)
        # tf_sim.update_velocities(velocities)
        
        new_forces = tf_sim.calculate_forces().numpy()
        np.frombuffer(shared_forces.get_obj(), dtype=np.float32).reshape((num_agents, 2))[:] = new_forces

        # tf_time.value = timer.calculate()
        tf_time.value = timer.calculate_time()
        timer.sleep_time(box2d_time.value)
        timer.print_fps(1)




def box2d_loop(shared_positions, shared_velocities, shared_forces, num_agents, world_width, world_height, rendering_queue, running, tf_time, box2d_time):
    from box2d_simulation import Box2DSimulation
    from timer import Timer
    initial_positions = np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape((num_agents, 2))
    initial_velocities = np.random.uniform(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX, (num_agents, 2)).astype(np.float32)
    box2d_sim = Box2DSimulation(world_width, world_height)
    box2d_sim.create_bodies(initial_positions, initial_velocities)
    timer = Timer("Box2d  ")
    
    while running.value:
        timer.start() # timer
        forces = np.frombuffer(shared_forces.get_obj(), dtype=np.float32).reshape((num_agents, 2))
        box2d_sim.apply_forces(forces)
        box2d_sim.step()
        
        new_positions = np.array(box2d_sim.get_positions(), dtype=np.float32)
        new_velocities = np.array(box2d_sim.get_velocities(), dtype=np.float32)
        
        np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape((num_agents, 2))[:] = new_positions
        np.frombuffer(shared_velocities.get_obj(), dtype=np.float32).reshape((num_agents, 2))[:] = new_velocities
        
        if rendering_queue.empty():
            rendering_queue.put(new_positions)
            
        # box2d_time.value = timer.calculate()
        box2d_time.value = timer.calculate_time()
        timer.sleep_time(tf_time.value)
        timer.print_fps(1)


def render_loop(world_width, world_height, rendering_queue, running):
    from visual_system import PygameRenderer
    from timer import Timer
    pygame.init()
    renderer = PygameRenderer(world_width, world_height)
    clock = pygame.time.Clock()
    timer = Timer("Render ")

    while running.value:
        timer.start()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running.value = False
                return
        if not rendering_queue.empty():
            positions = rendering_queue.get()
            renderer.draw(positions)
            
        timer.print_fps(1)
        
        clock.tick(RENDER_FPS)
    pygame.quit()



def run_simulation(num_agents, world_width, world_height):
    shared_positions = mp.Array('f', num_agents * 2)
    shared_velocities = mp.Array('f', num_agents * 2)
    shared_forces = mp.Array('f', num_agents * 2)
    rendering_queue = mp.Queue(maxsize=1)
    running = mp.Value('b', True)
    tf_time = mp.Value('d', 0.0)
    box2d_time = mp.Value('d', 0.0)



    # 初期位置の設定
    initial_positions = np.random.uniform(0, 1, (num_agents, 2)).astype(np.float32)
    initial_positions[:, 0] *= world_width
    initial_positions[:, 1] *= world_height
    initial_velocities = np.random.uniform(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX, (num_agents, 2)).astype(np.float32)

    np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape((num_agents, 2))[:] = initial_positions
    np.frombuffer(shared_velocities.get_obj(), dtype=np.float32).reshape((num_agents, 2))[:] = initial_velocities

    processes = [
        mp.Process(target=tf_loop, args=(shared_positions, shared_velocities, shared_forces, num_agents, world_width, world_height, running, tf_time, box2d_time)),
        mp.Process(target=box2d_loop, args=(shared_positions, shared_velocities, shared_forces, num_agents, world_width, world_height, rendering_queue, running, tf_time, box2d_time)),
        mp.Process(target=render_loop, args=(world_width, world_height, rendering_queue, running))
    ]

    for p in processes:
        p.start()
        
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating processes")
        running.value = False
        for p in processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    run_simulation(NUM_AGENTS, WORLD_WIDTH, WORLD_HEIGHT)
