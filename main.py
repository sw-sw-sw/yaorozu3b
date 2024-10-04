import pygame
import time
import numpy as np
import tensorflow as tf
import multiprocessing as mp
from tensorflow_simulation import TensorFlowSimulation
from box2d_simulation import Box2DSimulation
from visual_system import VisualSystem
from ecosystem import Ecosystem
from config_manager import ConfigManager
from parameter_control_ui import *
from timer import Timer
from log import get_logger, set_log_level
import logging

from TEST.performance_tracker import PerformanceTracker

logger = get_logger(__name__)

def eco_run(queues, shared_memory, running, initialization_complete, eco_init_done):
    ecosystem = Ecosystem(queues)
    timer = Timer("Ecosystem")
    
    try:
        ecosystem.initialize()
        eco_init_done.set()  # Signal that Ecosystem initialization is complete
        initialization_complete['Ecosystem'].set()
        logger.info("Ecosystem initialization complete")
    except Exception as e:
        logger.exception(f"Error during Ecosystem initialization: {e}")
        running.value = False
        return
    
    clock = pygame.time.Clock()
    target_fps = 60
    
    while running.value:
        try:
            timer.start()
            ecosystem.update()
            timer.print_fps(5)
            
            # Limit the frame rate to 60 FPS
            clock.tick(target_fps)
            
        except Exception as e:
            logger.exception(f"Error in Ecosystem update: {e}")
            running.value = False
            break
    
    logger.info("Ecosystem process ending")

def tf_run(queues, shared_memory, running, initialization_complete, eco_init_done):
    tensorflow = TensorFlowSimulation(queues)
    timer = Timer("TensorFlow")
    
    try:
        eco_init_done.wait()  # Wait for Ecosystem initialization to complete
        tensorflow.initialize()
        initialization_complete['TensorFlow'].set()
        logger.info("TensorFlow initialization complete")
    except Exception as e:
        logger.exception(f"Error during TensorFlow initialization: {e}")
        running.value = False
        return
    
    while running.value:
        try:
            timer.start()
            tensorflow.update()
            timer.print_fps(5)
        except Exception as e:
            logger.exception(f"Error in TensorFlow update: {e}")
            running.value = False
            break
    
    logger.info("TensorFlow process ending")

def box2d_run(queues, shared_memory, running, initialization_complete, eco_init_done):
    box2d = Box2DSimulation(queues)
    timer = Timer("Box2D")
    
    try:
        eco_init_done.wait()  # Wait for Ecosystem initialization to complete
        box2d.initialize()
        initialization_complete['Box2D'].set()
        logger.info("Box2D initialization complete")
    except Exception as e:
        logger.exception(f"Error during Box2D initialization: {e}")
        running.value = False
        return
    
    while running.value:
        try:
            timer.start()
            box2d.update()
            time.sleep(0.001)
            timer.print_fps(5)
        except Exception as e:
            logger.exception(f"Error in Box2D update: {e}")
            running.value = False
            break
    
    logger.info("Box2D process ending")
    
@PerformanceTracker.measure_time
def visual_system_run(queues, shared_memory, running, initialization_complete, eco_init_done):
    timer = Timer("Render ")
    visual_system = VisualSystem(queues)
    
    try:
        eco_init_done.wait() 
        visual_system.initialize()
        initialization_complete['Visual'].set()
        logger.info("Visual System initialization complete")
    except Exception as e:
        logger.exception(f"Error during Visual System initialization: {e}")
        running.value = False
        return
    
    while running.value:
        try:
            timer.start()
            visual_system.update()
            timer.print_fps(5)
            # time.sleep(0.001)
        except Exception as e:
            logger.exception(f"Error in Visual System update: {e}")
            running.value = False
            break
    
    visual_system.cleanup()
    logger.info("Visual System process ending")

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
        'eco_to_box2d_init': mp.Queue(),
        'eco_to_box2d': mp.Queue(),
        'eco_to_tf_init': mp.Queue(),
        'eco_to_visual_init': mp.Queue(),
        'eco_to_visual': mp.Queue(),
        # 'eco_to_visual_render': mp.Queue(maxsize=10),
        'eco_to_tf': mp.Queue(maxsize=1),
        'box2d_to_visual_render': mp.Queue(maxsize=1),
        'box2d_to_tf': mp.Queue(maxsize=1),
        'box2d_to_eco': mp.Queue(maxsize=1),
        'tf_to_box2d': mp.Queue(maxsize=1),
        'ui_to_tensorflow': mp.Queue(),
        'box2d_to_eco_collisions': mp.Queue(maxsize=1)
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
    set_log_level(logging.CRITICAL)  # ログレベルを設定（必要に応じて変更可能）
    run_simulation()