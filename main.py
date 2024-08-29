import time
import numpy as np
import multiprocessing as mp
import logging
import sys
import psutil
import threading
from config import *
from tensorflow_simulation import TensorFlowSimulation
from box2d_simulation import Box2DSimulation
from visual_system import VisualSystem
from timer import Timer
from ecosystem import Ecosystem

def setup_logging():
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])
    return logging.getLogger(__name__)

logger = setup_logging()

def monitor_resources():
    while True:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        logger.info(f"CPU usage: {cpu_percent}%, Memory usage: {memory_percent}%")
        time.sleep(5)

def eco_run(queues, shared_memory, running, initialization_complete):
    ecosystem = Ecosystem(queues)
    ecosystem.initialize_agents(shared_memory)
    initialization_complete.set()
    ecosystem.run(shared_memory, running)

def tf_run(queues, shared_memory, running, initialization_complete):
    tensorflow = TensorFlowSimulation(queues)
    timer = Timer("TensorFlow")
    initialization_complete.set()
    
    while running.value:
        timer.start()
        tensorflow.update_positions(shared_memory['positions'])
        tensorflow.update_species(shared_memory['agent_species'])
        tensorflow.apply_force_to_shared_memory(shared_memory['forces'])
        shared_memory['tf_time'].value = timer.calculate_time()
        timer.sleep_time(shared_memory['box2d_time'].value)
        timer.print_fps(1)

def box2d_run(queues, shared_memory, running, initialization_complete):
    box2d = Box2DSimulation(queues)
    timer = Timer("Box2D")
    box2d.create_bodies()
    initialization_complete.set()
    
    while running.value:
        timer.start()
        box2d.apply_forces_to_box2d(shared_memory['forces'])
        box2d.step()
        box2d.apply_positions_to_shared_memory(shared_memory['positions'])
        box2d.add_positions_to_render_queue()
        shared_memory['box2d_time'].value = timer.calculate_time()
        timer.sleep_time(shared_memory['tf_time'].value)
        timer.print_fps(1)

def visual_system_run(queues, shared_memory, running, initialization_complete):
    visual_system = VisualSystem(queues, running)
    initialization_complete.set()
    visual_system.run()

def run_simulation():
    logger.info("Starting simulation")
    
    shared_memory = {
        'positions': mp.Array('f', MAX_AGENTS_NUM * 2),
        'velocities': mp.Array('f', MAX_AGENTS_NUM * 2),
        'forces': mp.Array('f', MAX_AGENTS_NUM * 2),
        'agent_ids': mp.Array('i', MAX_AGENTS_NUM),
        'agent_species': mp.Array('i', MAX_AGENTS_NUM),
        'current_agent_count': mp.Value('i', 0),
        'tf_time': mp.Value('d', 0.0),
        'box2d_time': mp.Value('d', 0.0),
        'lock': mp.Lock()
    }

    running = mp.Value('b', True)
    queues = {
        'eco_to_box2d': mp.Queue(),
        'eco_to_visual': mp.Queue(),
        'eco_to_visual_creatures': mp.Queue(),
        'eco_to_box2d_creatures': mp.Queue(),
        'eco_to_tensorflow': mp.Queue(),
        'box2d_to_eco': mp.Queue(),
        'visual_to_eco': mp.Queue(),
        'tensorflow_to_eco': mp.Queue(),
        'rendering_queue': mp.Queue(maxsize=1),
    }

    initialization_complete = {
        'Ecosystem': mp.Event(),
        'TensorFlow': mp.Event(),
        'Box2D': mp.Event(),
        'Visual': mp.Event()
    }

    processes = [
        mp.Process(target=eco_run, args=(queues, shared_memory, running, initialization_complete['Ecosystem']), name='Ecosystem'),
        mp.Process(target=tf_run, args=(queues, shared_memory, running, initialization_complete['TensorFlow']), name="TensorFlow"),
        mp.Process(target=box2d_run, args=(queues, shared_memory, running, initialization_complete['Box2D']), name="Box2D"),
        mp.Process(target=visual_system_run, args=(queues, shared_memory, running, initialization_complete['Visual']), name="Visual")
    ]

    for process in processes:
        logger.info(f"Starting {process.name} process")
        process.start()

    for name, event in initialization_complete.items():
        logger.info(f"Waiting for {name} to initialize...")
        event.wait()
        logger.info(f"{name} initialization complete")

    logger.info("All processes initialized and running")

    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.daemon = True
    monitor_thread.start()
    logger.info("Resource monitoring thread started")

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