from log import get_logger
from collections import deque
import numpy as np

class FlameBuffer:
    def __init__(self, max_agents, target_size=10, max_size=20):
        self.logger = get_logger(self.__class__.__name__)
        self.buffer = deque(maxlen=max_size)
        self.target_size = target_size
        self.max_size = max_size
        self.current_positions = np.zeros((max_agents, 2), dtype=np.float32)
        self.next_positions = np.zeros((max_agents, 2), dtype=np.float32)
        self.agent_ids = np.full(max_agents, -1, dtype=np.int32)
        self.overflow_count = 0
        self.last_agent_count = 0

    def initialize(self, initial_positions, initial_agent_ids):
        self.current_positions[:len(initial_positions)] = initial_positions
        self.next_positions[:len(initial_positions)] = initial_positions
        self.agent_ids[:len(initial_agent_ids)] = initial_agent_ids
        self.buffer.clear()
        self.buffer.append(self.current_positions[:len(initial_positions)])
        self.last_agent_count = len(initial_positions)

    def update(self, new_positions, new_agent_ids, interpolation_steps, current_agent_count):
        if current_agent_count != self.last_agent_count:
            self.logger.warning(f"Agent count mismatch. Expected: {self.last_agent_count}, Got: {current_agent_count}. Resetting buffer.")
            self.current_positions[:current_agent_count] = new_positions
            self.next_positions[:current_agent_count] = new_positions
            self.agent_ids[:current_agent_count] = new_agent_ids
            # self.buffer.clear()
            self.buffer.append(self.current_positions[:current_agent_count])
            self.last_agent_count = current_agent_count
            self.get_stats()

        self.current_positions[:current_agent_count] = self.next_positions[:current_agent_count]
        self.next_positions[:current_agent_count] = new_positions
        self.agent_ids[:current_agent_count] = new_agent_ids

        interpolated_frames = self._interpolate_vectorized(interpolation_steps, current_agent_count)
        self.add_frames(interpolated_frames)
        self._adjust_buffer_size()
        
        self.last_agent_count = current_agent_count
        return self.get_stats()

    def update_with_physics_data(self, positions, agent_ids, current_agent_count, physics_update_interval, avg_frame_time):
        self.current_positions[:current_agent_count] = self.next_positions[:current_agent_count]
        self.next_positions[:current_agent_count] = positions
        self.agent_ids[:current_agent_count] = agent_ids

        base_interpolation_steps = max(1, int(physics_update_interval / avg_frame_time))
        correction_value = 2  # This can be adjusted based on your needs
        interpolation_steps = base_interpolation_steps + correction_value

        current_buffer_size = len(self.buffer)
        if current_buffer_size < self.target_size:
            interpolation_steps += 1
        elif current_buffer_size > self.target_size:
            interpolation_steps = max(1, interpolation_steps - 1)

        interpolated_frames = self._interpolate_vectorized(interpolation_steps, current_agent_count)
        self.add_frames(interpolated_frames)
        
        self.last_agent_count = current_agent_count
        
        self.logger.info(f"Buffer stats: {self.get_stats()}, Interpolation steps: {interpolation_steps}")
        
        return 
    
    def _interpolate_vectorized(self, steps, current_agent_count):
        current_array = self.current_positions[:current_agent_count]
        next_array = self.next_positions[:current_agent_count]
        t_values = np.linspace(0, 1, steps+1)[1:]
        interpolated_arrays = current_array[np.newaxis, :, :] + \
                              (next_array - current_array)[np.newaxis, :, :] * t_values[:, np.newaxis, np.newaxis]
        
        return [interpolated_arrays[i] for i in range(steps)]

    def add_frames(self, frames):
        for frame in frames:
            if len(self.buffer) >= self.max_size:
                self.overflow_count += 1
                self.buffer.popleft()
            self.buffer.append(frame)

    def get_next_position(self):
        return self.buffer.popleft() if self.buffer else self.current_positions[:self.last_agent_count]

    def get_stats(self):
        return {
            "current_size": len(self.buffer),
            "target_size": self.target_size,
            "max_size": self.max_size,
            "overflow_count": self.overflow_count
        }