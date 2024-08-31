import threading
import numpy as np
import random
import time

class SharedMemoryManager:
    def __init__(self, max_agents, dimensions):
        self.max_agents = max_agents
        self.dimensions = dimensions
        self.positions = np.zeros((max_agents, dimensions), dtype=np.float32)
        self.agent_ids = np.zeros(max_agents, dtype=np.int32)
        self.active_mask = np.zeros(max_agents, dtype=np.int32)
        self.current_agent_count = 0
        self.next_agent_id = 0
        self.agent_changes = []
        self.lock = threading.Lock()

    def add_agent(self, position):
        with self.lock:
            if self.current_agent_count < self.max_agents:
                index = self.current_agent_count
                self.positions[index] = position
                self.agent_ids[index] = self.next_agent_id
                self.active_mask[index] = 1
                self.current_agent_count += 1
                new_agent_id = self.next_agent_id
                self.next_agent_id += 1
                self.agent_changes.append(('add', new_agent_id))
                return new_agent_id
            return -1

    def remove_agent(self, agent_id):
        with self.lock:
            index = np.where(self.agent_ids == agent_id)[0]
            if len(index) > 0:
                index = index[0]
                self.active_mask[index] = 0
                self.current_agent_count -= 1
                self.agent_changes.append(('remove', agent_id))
                return True
            return False

    def get_data_for_tensorflow(self):
        # ロックなしで読み取り、一時的な不一致を許容
        active_indices = np.where(self.active_mask == 1)[0]
        return self.positions[active_indices].copy()

    def get_data_for_box2d(self):
        # ロックなしで読み取り、一時的な不一致を許容
        active_indices = np.where(self.active_mask == 1)[0]
        return list(zip(self.agent_ids[active_indices], self.positions[active_indices]))

    def get_data_for_visual(self):
        return self.get_data_for_tensorflow()

    def update_positions(self, new_positions):
        # ロックなしで更新、一時的な不一致を許容
        active_indices = np.where(self.active_mask == 1)[0]
        if len(active_indices) != len(new_positions):
            print(f"Warning: Mismatch in number of active agents ({len(active_indices)}) and new positions ({len(new_positions)})")
            # 可能な範囲で更新を行う
            update_count = min(len(active_indices), len(new_positions))
            self.positions[active_indices[:update_count]] = new_positions[:update_count]
        else:
            self.positions[active_indices] = new_positions
        return True
    
class Ecosystem:
    def __init__(self, shared_memory):
        self.shared_memory = shared_memory

    def step(self):
        positions = self.shared_memory.get_data_for_tensorflow()
        if len(positions) > 0:
            new_positions = positions + np.random.uniform(-0.1, 0.1, positions.shape)
            self.shared_memory.update_positions(new_positions)

        if random.random() < 0.1:
            self.add_agent()
        if random.random() < 0.05:
            self.remove_agent()

    def add_agent(self):
        position = np.random.rand(2)
        agent_id = self.shared_memory.add_agent(position)
        if agent_id != -1:
            print(f"Added agent {agent_id}")

    def remove_agent(self):
        data = self.shared_memory.get_data_for_box2d()
        if data:
            agent_id = random.choice(data)[0]
            if self.shared_memory.remove_agent(agent_id):
                print(f"Removed agent {agent_id}")

class Box2DSimulation:
    def __init__(self, shared_memory):
        self.shared_memory = shared_memory

    def step(self):
        data = self.shared_memory.get_data_for_box2d()
        if data:
            new_positions = [pos + np.random.uniform(-0.05, 0.05, 2) for _, pos in data]
            self.shared_memory.update_positions(new_positions)

class TensorFlowSimulation:
    def __init__(self, shared_memory):
        self.shared_memory = shared_memory

    def calculate_forces(self):
        positions = self.shared_memory.get_data_for_tensorflow()
        if len(positions) > 0:
            return np.random.uniform(-1, 1, positions.shape)
        return np.array([])

class VisualSystem:
    def __init__(self, shared_memory):
        self.shared_memory = shared_memory

    def draw(self):
        positions = self.shared_memory.get_data_for_visual()
        print(f"Drawing {len(positions)} agents")

def ecosystem_loop(shared_memory, running):
    ecosystem = Ecosystem(shared_memory)
    while running.is_set():
        ecosystem.step()
        time.sleep(0.01)

def box2d_loop(shared_memory, running):
    box2d_sim = Box2DSimulation(shared_memory)
    while running.is_set():
        box2d_sim.step()
        time.sleep(0.01)

def tf_loop(shared_memory, running):
    tf_sim = TensorFlowSimulation(shared_memory)
    while running.is_set():
        forces = tf_sim.calculate_forces()
        if len(forces) > 0:
            shared_memory.update_positions(forces)
        time.sleep(0.01)

def visual_system_loop(shared_memory, running):
    visual_system = VisualSystem(shared_memory)
    while running.is_set():
        visual_system.draw()
        time.sleep(0.1)

def run_simulation():
    max_agents = 1000
    dimensions = 2
    shared_memory = SharedMemoryManager(max_agents, dimensions)
    running = threading.Event()
    running.set()

    threads = [
        threading.Thread(target=ecosystem_loop, args=(shared_memory, running)),
        threading.Thread(target=box2d_loop, args=(shared_memory, running)),
        threading.Thread(target=tf_loop, args=(shared_memory, running)),
        threading.Thread(target=visual_system_loop, args=(shared_memory, running))
    ]

    for thread in threads:
        thread.start()

    try:
        time.sleep(30)  # Run simulation for 30 seconds
    finally:
        running.clear()
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    run_simulation()