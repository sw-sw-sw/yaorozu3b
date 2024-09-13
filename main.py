import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import threading
from tensorflow_simulation import TensorFlowSimulation
from box2d_simulation import Box2DSimulation
from visual_system import VisualSystem
from ecosystem import Ecosystem
from config_manager import ConfigManager
from parameter_control_ui import *
from timer import Timer
from log import *


#------------------------------------- Main routine -----------------------------------------

def eco_run(queues, shared_memory, running, initialization_complete, eco_init_done):
    ecosystem = Ecosystem(queues)
    timer = Timer("Ecosystem")
    
    ecosystem.initialize()

    eco_init_done.set()  # Signal that Ecosystem initialization is complete
    initialization_complete['Ecosystem'].set()
    
    while running.value:
        timer.start()
        ecosystem.update()
        timer.print_fps(5)

def tf_run(queues, shared_memory, running, initialization_complete, eco_init_done):
    tensorflow = TensorFlowSimulation(queues)
    timer = Timer("TensorFlow")
    
    eco_init_done.wait()  # Wait for Ecosystem initialization to complete
    tensorflow.initialize()
    initialization_complete['TensorFlow'].set()
    
    while running.value:
        timer.start()
        tensorflow.update()
        timer.print_fps(5)
        time.sleep(0.005)

def box2d_run(queues, shared_memory, running, initialization_complete, eco_init_done):
    box2d = Box2DSimulation(queues)
    timer = Timer("Box2D")
    
    eco_init_done.wait()  # Wait for Ecosystem initialization to complete
    box2d.initialize()
    initialization_complete['Box2D'].set()
    
    while running.value:
        timer.start()
        box2d.update()
        timer.print_fps(5)
        time.sleep(0.005)

def visual_system_run(queues, shared_memory, running, initialization_complete, eco_init_done):
    timer = Timer("Render ")
    visual_system = VisualSystem(queues)
    eco_init_done.wait() 
    visual_system.initialize()
    initialization_complete['Visual'].set()
    
    while running.value:
        timer.start()
        visual_system.update()
        timer.print_fps(5)
        time.sleep(0.005)
        
    visual_system.cleanup()

def run_simulation():
    logger.info("Starting simulation")
    
    config_manager = ConfigManager()
    
    shared_memory = {
        'positions': mp.Array('f', config_manager.get_trait_value('MAX_AGENTS_NUM') * 2),
        'velocities': mp.Array('f', config_manager.get_trait_value('MAX_AGENTS_NUM') * 2),
        'forces': mp.Array('f', config_manager.get_trait_value('MAX_AGENTS_NUM') * 2),
        'agent_ids': mp.Array('i', config_manager.get_trait_value('MAX_AGENTS_NUM')),
        'species': mp.Array('i', config_manager.get_trait_value('MAX_AGENTS_NUM')),
        'current_agent_count': mp.Value('i', 0),
        'tf_time': mp.Value('d', 0.0),
        'box2d_time': mp.Value('d', 0.0),
        'lock': mp.Lock(),
    }

    controllable_params = [
        'SEPARATION_DISTANCE', 'SEPARATION_WEIGHT',
        'COHESION_DISTANCE', 'COHESION_WEIGHT',
        'MAX_FORCE', 'CENTER_ATTRACTION_WEIGHT',
        'CONFINEMENT_WEIGHT', 'ROTATION_STRENGTH',
        'ESCAPE_DISTANCE', 'ESCAPE_WEIGHT',
        'CHASE_DISTANCE', 'CHASE_WEIGHT'
    ]

    for param in controllable_params:
        shared_memory[param] = mp.Value('f', config_manager.get_trait_value(param))

    running = mp.Value('b', True)
    queues = {
        'eco_to_box2d': mp.Queue(),
        'eco_to_visual': mp.Queue(),
        'eco_to_tf_init': mp.Queue(),
        'eco_to_visual_init': mp.Queue(),
        'eco_to_box2d_init': mp.Queue(),
        'box2d_to_tf': mp.Queue(maxsize=1),
        'box2d_to_eco': mp.Queue(maxsize=1),
        'tf_to_box2d': mp.Queue(maxsize=1),
        'eco_to_visual_render': mp.Queue(maxsize=100),
        'ui_to_tensorflow': mp.Queue() 
    }

    initialization_complete = {
        'Ecosystem': mp.Event(),
        'TensorFlow': mp.Event(),
        'Box2D': mp.Event(),
        'Visual': mp.Event()
    }
    running = mp.Value('b', True)
    eco_init_done = mp.Event()  # New event to signal Ecosystem initialization completion

    processes = [
        mp.Process(target=eco_run, args=(queues, shared_memory, running, initialization_complete, eco_init_done), name='Ecosystem'),
        mp.Process(target=tf_run, args=(queues, shared_memory, running, initialization_complete, eco_init_done), name="TensorFlow"),
        mp.Process(target=box2d_run, args=(queues, shared_memory, running, initialization_complete, eco_init_done), name="Box2D"),
        mp.Process(target=visual_system_run, args=(queues, shared_memory, running, initialization_complete, eco_init_done), name="Visual"),
        mp.Process(target=run_parameter_control_ui, args=(shared_memory, queues, running), name="ParameterControlUI")
    ]

    for process in processes:
        logger.info(f"Starting {process.name} process")
        process.start()
    logger.info("Waiting for Ecosystem to initialize...")
    eco_init_done.wait()
    logger.info("Ecosystem initialization complete")
    # 全てのプロセスの初期化完了を待つ
    for name, event in initialization_complete.items():
        logger.info(f"Waiting for {name} to initialize...")
        event.wait()
        logger.info(f"{name} initialization complete")

    # 全てのプロセスが初期化完了したことを通知
    logger.info("All processes initialized and running")

    try:
        while all(p.is_alive() for p in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Caught KeyboardInterrupt, terminating processes")
    finally:
        running.value = False
        for p in processes:
            p.terminate()
            p.join()

    logger.info("Simulation ended")

if __name__ == "__main__":
    run_simulation()