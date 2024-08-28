# main.py
# new project

import tensorflow as tf
import time
import numpy as np
import pygame
from Box2D import b2World, b2Vec2
from config import *
import multiprocessing as mp
from tensorflow_simulation import TensorFlowSimulation
from box2d_simulation import Box2DSimulation
from visual_system import VisualSystem
from timer import Timer
from ecosystem import Ecosystem

def eco_run(queue, share_memory, running):
    ecosystem = Ecosystem(NUM_AGENTS)



def tf_run(queues, shared_memory, running):
    tensorflow = TensorFlowSimulation(NUM_AGENTS, WORLD_WIDTH, WORLD_HEIGHT)
    timer = Timer("Tensor ")
    _positions = shared_memory['positions']
    _forces = shared_memory['forces']

    while running.value:
        timer.start()
        # forceを計算するルーチン
        
        positions = np.frombuffer(_positions.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))
        tensorflow.update_positions(tf.constant(positions, dtype=tf.float32))
        new_forces = tensorflow.calculate_forces().numpy()
        np.frombuffer(_forces.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))[:] = new_forces

        #fps計算　周期をあわせる
        shared_memory['tf_time'].value = timer.calculate_time()
        timer.sleep_time(shared_memory['box2d_time'].value)
        timer.print_fps(1)


def box2d_run(queues, shared_memory, running):
    timer = Timer("Box2d  ")
    box2d = Box2DSimulation(WORLD_WIDTH, WORLD_HEIGHT, queues)
    
    # 共有メモリの一時参照
    _positions = shared_memory['positions']
    _forces = shared_memory['forces']
    _box2d_time = shared_memory['box2d_time']
    _tf_time = shared_memory['tf_time']
    
    # エージェントのbox2dインスタンスのイニシャライズ
    initial_positions = np.frombuffer(_positions.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))
    initial_velocities = np.random.uniform(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX, (NUM_AGENTS, 2)).astype(np.float32)
    box2d.create_bodies(initial_positions, initial_velocities)


    while running.value:
        # 物理シミュレーションの更新
        timer.start() # timer
        # forces = np.frombuffer(_forces.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))
        # box2d.apply_forces(forces)
        box2d.apply_forces_to_box2d(_forces)
        box2d.step()
        box2d.apply_positions_to_shared_memory(_positions)
        # new_positions = box2d.get_positions()        
        # np.frombuffer(_positions.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))[:] = new_positions
        
        box2d.add_positions_to_render_queue() # visual systemにpositionを送る
            
        _box2d_time.value = timer.calculate_time()
        timer.sleep_time(_tf_time.value)
        timer.print_fps(1)
        
def visual_system_run(queues, shared_memory, running):
    visual_system = VisualSystem(WORLD_WIDTH, WORLD_HEIGHT, queues, running)
    visual_system.run()


def run_simulation():
    max_agents = NUM_AGENTS
    dimensions = 2
    
    # Shared memory arrays
    positions = mp.Array('f', max_agents * dimensions)
    velocities = mp.Array('f', max_agents * dimensions)
    forces = mp.Array('f', max_agents * dimensions)
    agent_ids = mp.Array('i', max_agents)
    agent_species = mp.Array('i', max_agents)
    active_mask = mp.Array('i', max_agents)
    tf_time = mp.Value('d', 0.0)
    box2d_time = mp.Value('d', 0.0)

    current_agent_count = mp.Value('i', 0)
    lock = mp.Lock()

    shared_memory = {
        'positions': positions,
        'velocities': velocities,
        'forces': forces,
        'agent_ids': agent_ids,
        'agent_species': agent_species,
        'active_mask': active_mask,
        'current_agent_count': current_agent_count,
        'lock': lock,
        'tf_time': tf_time,
        'box2d_time': box2d_time
    }

    running = mp.Value('b', True)

    queues = {
        'eco_to_box2d': mp.Queue(),
        'box2d_to_eco': mp.Queue(),
        'eco_to_visual': mp.Queue(),
        'visual_to_eco': mp.Queue(),
        'eco_to_tensorflow': mp.Queue(),
        'tensorflow_to_eco': mp.Queue(),
        'rendering_queue': mp.Queue(maxsize=1)
    }

    # NumPy arrays for convenient access
    np_positions = np.frombuffer(positions.get_obj(), dtype=np.float32).reshape((max_agents, dimensions))
    np_velocities = np.frombuffer(velocities.get_obj(), dtype=np.float32).reshape((max_agents, dimensions))
    np_forces = np.frombuffer(forces.get_obj(), dtype=np.float32).reshape((max_agents, dimensions))
    np_agent_ids = np.frombuffer(agent_ids.get_obj(), dtype=np.int32)
    np_agent_species = np.frombuffer(agent_species.get_obj(), dtype=np.int32)
    np_active_mask = np.frombuffer(active_mask.get_obj(), dtype=np.int32)
    
    # 初期位置の設定
    with lock:
        current_agent_count.value = NUM_AGENTS
        np_positions[:] = np.random.uniform(0, 1, (NUM_AGENTS, dimensions))
        np_positions[:, 0] *= WORLD_WIDTH
        np_positions[:, 1] *= WORLD_HEIGHT
        np_velocities[:] = np.random.uniform(INITIAL_VELOCITY_MIN, INITIAL_VELOCITY_MAX, (NUM_AGENTS, dimensions))
        np_forces[:] = np.zeros((NUM_AGENTS, dimensions))
        np_agent_ids[:] = np.arange(NUM_AGENTS)
        np_agent_species[:] = np.zeros(NUM_AGENTS, dtype=np.int32)
        np_active_mask[:] = np.ones(NUM_AGENTS, dtype=np.int32)
        

    processes = [
        mp.Process(target=tf_run, args=(queues, shared_memory, running), name="TensorFlow"),
        mp.Process(target=box2d_run, args=(queues, shared_memory, running), name="Box2D"),
        mp.Process(target=visual_system_run, args=(queues, shared_memory, running), name="Visual")
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
    run_simulation()