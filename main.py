
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


def eco_run(queues, shared_memory, running):
    ecosystem = Ecosystem()
    ecosystem.initialize_agents(shared_memory)
    ecosystem.run(shared_memory, queues, running)
    
    


def tf_run(queues, shared_memory, running):
    tensorflow = TensorFlowSimulation(queues)
    timer = Timer("Tensor ")
    
    _positions = shared_memory['positions']
    _forces = shared_memory['forces']
    _box2d_time = shared_memory['box2d_time']
    _tf_time = shared_memory['tf_time']
    
    while running.value:
        timer.start()
        tensorflow.update_positions(_positions)
        tensorflow.apply_force_to_shared_memory(_forces)

        #周期調整
        _tf_time.value = timer.calculate_time()
        timer.sleep_time(_box2d_time.value)
        timer.print_fps(1)


def box2d_run(queues, shared_memory, running):
    timer = Timer("Box2d  ")
    box2d = Box2DSimulation(queues)
    
    # 共有メモリの一時、参照
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
        box2d.apply_forces_to_box2d(_forces)
        box2d.step()
        box2d.apply_positions_to_shared_memory(_positions)
        box2d.add_positions_to_render_queue() # visual systemにpositionを送る
        #  周期調整
        _box2d_time.value = timer.calculate_time()
        timer.sleep_time(_tf_time.value)
        timer.print_fps(1)
        
def visual_system_run(queues, shared_memory, running):
    visual_system = VisualSystem(queues, running)
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
    current_agent_count = mp.Array('i', max_agents)
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
        'rendering_queue': mp.Queue(maxsize=1),
        'initialize_finished': mp.Queue()
    }

    processes = [
        mp.Process(target=eco_run, args=(queues, shared_memory, running), name='Ecosystem'),
        mp.Process(target=tf_run, args=(queues, shared_memory, running), name="TensorFlow"),
        mp.Process(target=box2d_run, args=(queues, shared_memory, running), name="Box2D"),
        mp.Process(target=visual_system_run, args=(queues, shared_memory, running), name="Visual")
    ]
    
    #ecosystemのプロセスの始動
    for process in processes:
        if process.name == 'Ecosystem':
            process.start()
            
    time.sleep(3) #初期化完了までの時間を待つ        

# ----------------- other processes start ----------------------


     #他のサブプロセスの始動
    for p in processes:
        if p.name != 'Ecosystem':
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