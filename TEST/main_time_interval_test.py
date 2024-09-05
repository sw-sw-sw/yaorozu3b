# main.py

import tensorflow as tf
import time
import numpy as np
import pygame
from Box2D import b2World, b2Vec2
from config import *
import multiprocessing as mp

def tf_loop(shared_positions, shared_forces, num_agents, world_width, world_height, running, tf_time, box2d_time, fps_update_interval):
    import tensorflow as tf
    from tensorflow_simulation import TensorFlowSimulation
    from timer import Timer
    tf_sim = TensorFlowSimulation(num_agents, world_width, world_height)
    timer = Timer("Tensorflow", fps_update_interval)
    
    while running.value:
        timer.start()        
        positions = np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape((num_agents, 2))
        tf_sim.update_positions(tf.constant(positions, dtype=tf.float32))
        new_forces = tf_sim.calculate_forces().numpy()
        np.frombuffer(shared_forces.get_obj(), dtype=np.float32).reshape((num_agents, 2))[:] = new_forces

        tf_time.value = timer.calculate()
        timer.adjust_time(box2d_time.value)

def box2d_loop(shared_positions, shared_forces, num_agents, world_width, world_height, rendering_queue, running, tf_time, box2d_time, fps_update_interval):
    from box2d_simulation import Box2DSimulation
    from timer import Timer
    initial_positions = np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape((num_agents, 2))
    initial_velocities = np.random.uniform(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX, (num_agents, 2)).astype(np.float32)
    box2d_sim = Box2DSimulation(world_width, world_height)
    box2d_sim.create_bodies(initial_positions, initial_velocities)
    timer = Timer("Box2d", fps_update_interval)
    
    while running.value:
        timer.start() # timer
        forces = np.frombuffer(shared_forces.get_obj(), dtype=np.float32).reshape((num_agents, 2))
        box2d_sim.apply_forces(forces)
        box2d_sim.step()
        new_positions = np.array(box2d_sim.get_positions(), dtype=np.float32)
        np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape((num_agents, 2))[:] = new_positions
        
        if rendering_queue.empty():
            rendering_queue.put(new_positions)
        #       timer
        box2d_time.value = timer.calculate() #
        timer.adjust_time(tf_time.value)



def render_loop(world_width, world_height, rendering_queue, running, fps_update_interval):
    from visual_system import PygameRenderer
    from timer import Timer
    pygame.init()
    renderer = PygameRenderer(world_width, world_height)
    clock = pygame.time.Clock()
    timer = Timer("Render", fps_update_interval)

    while running.value:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running.value = False
                return
        if not rendering_queue.empty():
            positions = rendering_queue.get()
            renderer.draw(positions)
            
        timer.calculate()
        clock.tick(RENDER_FPS)
    pygame.quit()



def run_simulation(num_agents, world_width, world_height, fps_update_interval, duration):
    shared_positions = mp.Array('f', num_agents * 2)
    shared_forces = mp.Array('f', num_agents * 2)
    rendering_queue = mp.Queue(maxsize=1)
    running = mp.Value('b', True)
    tf_time = mp.Value('d', 0.0)
    box2d_time = mp.Value('d', 0.0)

    # 初期位置の設定
    initial_positions = np.random.uniform(0, 1, (num_agents, 2)).astype(np.float32)
    initial_positions[:, 0] *= world_width
    initial_positions[:, 1] *= world_height
    np.frombuffer(shared_positions.get_obj(), dtype=np.float32).reshape((num_agents, 2))[:] = initial_positions

    processes = [
        mp.Process(target=tf_loop, args=(shared_positions, shared_forces, num_agents, world_width, world_height, running, tf_time, box2d_time, fps_update_interval)),
        mp.Process(target=box2d_loop, args=(shared_positions, shared_forces, num_agents, world_width, world_height, rendering_queue, running, tf_time, box2d_time, fps_update_interval)),
        mp.Process(target=render_loop, args=(world_width, world_height, rendering_queue, running, fps_update_interval))
    ]

    for p in processes:
        p.start()
        
    # 指定された時間だけシミュレーションを実行
    time.sleep(duration)
    # シミュレーションを停止
    running.value = False
    
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating processes")
        running.value = False
        for p in processes:
            p.terminate()
            p.join()

# if __name__ == "__main__":
#     run_simulation(NUM_AGENTS, WORLD_WIDTH, WORLD_HEIGHT)

# 実験用のメイン関数
if __name__ == "__main__":
    intervals = [0.005, 0.0001]
    test_duration = 10  # 各間隔でのテスト時間（秒）

    for interval in intervals:
        print(f"\nTesting with fps_update_interval = {interval}")
        run_simulation(NUM_AGENTS, WORLD_WIDTH, WORLD_HEIGHT, interval, test_duration)
        time.sleep(5)  # 各テスト間の待機時間