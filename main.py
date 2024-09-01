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
import multiprocessing as mp
from parameter_control_ui import ParameterControlUI  
import tkinter as tk 

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

def run_parameter_control_ui(shared_memory, queues, running):
    root = tk.Tk()
    ui_to_tensorflow_queue = queues['ui_to_tensorflow']

    def update_callback(param_name, value):
        shared_memory[param_name].value = value
        ui_to_tensorflow_queue.put((param_name, value))

    ui = ParameterControlUI(root, update_callback)

    # 初期値の設定
    initial_values = {
        'separation_distance': shared_memory['separation_distance'].value,
        'separation_weight': shared_memory['separation_weight'].value,
        'cohesion_distance': shared_memory['cohesion_distance'].value,
        'cohesion_weight': shared_memory['cohesion_weight'].value,
        'max_force': shared_memory['max_force'].value,
        'center_attraction_weight': shared_memory['center_attraction_weight'].value,
        'confinement_weight': shared_memory['confinement_weight'].value,
        'rotation_strength': shared_memory['rotation_strength'].value
    }
    ui.set_initial_values(initial_values)

    def on_closing():
        running.value = False
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    def check_running():
        if running.value:
            root.after(100, check_running)
        else:
            root.quit()

    root.after(100, check_running)
    root.mainloop()

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
        tensorflow.update_parameters(shared_memory)
        
        shared_memory['tf_time'].value = timer.calculate_time()
        timer.sleep_time(shared_memory['box2d_time'].value)
        timer.print_fps(5)
        

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
        timer.print_fps(5)

def visual_system_run(queues, shared_memory, running, initialization_complete):
    visual_system = VisualSystem(queues, running)
    initialization_complete.set()
    
    while running.value:
        if not visual_system.update():
            break

    visual_system.cleanup()
    
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
        'lock': mp.Lock(),
        # 新しいパラメータを追加
        'separation_distance': mp.Value('f', SEPARATION_DISTANCE),
        'separation_weight': mp.Value('f', SEPARATION_WEIGHT),
        'cohesion_distance': mp.Value('f', COHESION_DISTANCE),
        'cohesion_weight': mp.Value('f', COHESION_WEIGHT),
        'max_force': mp.Value('f', MAX_FORCE),
        'center_attraction_weight': mp.Value('f', CENTER_ATTRACTION_WEIGHT),
        'confinement_weight': mp.Value('f', CONFINEMENT_WEIGHT),
        'rotation_strength': mp.Value('f', ROTATION_STRENGTH)
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
        'ui_to_tensorflow': mp.Queue()  # 新しく追加
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
        mp.Process(target=visual_system_run, args=(queues, shared_memory, running, initialization_complete['Visual']), name="Visual"),
        mp.Process(target=run_parameter_control_ui, args=(shared_memory, queues, running), name="ParameterControlUI")  # 新しく追加
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