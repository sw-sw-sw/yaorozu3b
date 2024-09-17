# numpy_position_buffer.pyx
# cython: language_level=3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

cimport cython
import numpy as np
cimport numpy as np
from collections import deque
from libc.math cimport floor

np.import_array()

ctypedef np.float32_t DTYPE_t

cdef class NumpyPositionBuffer:
    cdef:
        public object buffer
        public int target_size, min_size, max_size
        public dict current_positions, next_positions
        public int overflow_count
        public double physics_update_interval
        public int physics_update_count
        public double last_physics_update_count_time
        public object frame_times
    def __init__(self, int target_size=10, int min_size=5, int max_size=20):
        self.buffer = deque(maxlen=max_size)
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
        self.current_positions = {}
        self.next_positions = {}
        self.overflow_count = 0
        self.physics_update_interval = 0.1
        self.physics_update_count = 0
        self.last_physics_update_count_time = 0
        self.frame_times = deque(maxlen=60)

    def initialize(self, dict initial_positions):
        self.current_positions = initial_positions
        self.next_positions = initial_positions.copy()
        self.buffer.clear()
        self.buffer.append(self.current_positions)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef update(self, dict new_positions, double current_time, double target_fps):
        cdef:
            double avg_frame_time
            int base_interpolation_steps, correction_value, interpolation_steps, current_buffer_size
            list interpolated_frames
        self.current_positions = self.next_positions
        self.next_positions = new_positions
               
        self.physics_update_count += 1
        if current_time - self.last_physics_update_count_time >= 1.0:
            self.physics_update_interval = 1.0 / self.physics_update_count
            self.physics_update_count = 0
            self.last_physics_update_count_time = current_time

        avg_frame_time = sum(self.frame_times) / len(self.frame_times) if self.frame_times else (1.0 / target_fps)
        base_interpolation_steps = max(1, int(self.physics_update_interval / avg_frame_time))
        
        correction_value = 2
        interpolation_steps = base_interpolation_steps + correction_value

        current_buffer_size = len(self.buffer)
        if current_buffer_size < self.target_size:
            interpolation_steps += 1
        elif current_buffer_size > self.target_size:
            interpolation_steps = max(1, interpolation_steps - 1)

        interpolated_frames = self._interpolate_vectorized(interpolation_steps)
        self.add_frames(interpolated_frames)
        self._adjust_buffer_size()
        return interpolated_frames, self.get_stats(), interpolation_steps

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef list _interpolate_vectorized(self, int steps):
        cdef:
            list agent_ids
            np.ndarray[DTYPE_t, ndim=2] current_array, next_array
            np.ndarray[DTYPE_t, ndim=1] t_values
            np.ndarray[DTYPE_t, ndim=3] interpolated_arrays
            int i, j
            DTYPE_t[:, :] current_view, next_view
            DTYPE_t[:] t_view
            DTYPE_t[:, :, :] interpolated_view

        agent_ids = list(self.current_positions.keys())
        current_array = np.array([self.current_positions[aid] for aid in agent_ids], dtype=np.float32)
        next_array = np.array([self.next_positions[aid] for aid in agent_ids], dtype=np.float32)
        
        t_values = np.linspace(0, 1, steps+1, dtype=np.float32)[1:]
        
        current_view = current_array
        next_view = next_array
        t_view = t_values
        
        interpolated_arrays = np.empty((steps, len(agent_ids), 2), dtype=np.float32)
        interpolated_view = interpolated_arrays

        for i in range(steps):
            for j in range(len(agent_ids)):
                interpolated_view[i, j, 0] = current_view[j, 0] + (next_view[j, 0] - current_view[j, 0]) * t_view[i]
                interpolated_view[i, j, 1] = current_view[j, 1] + (next_view[j, 1] - current_view[j, 1]) * t_view[i]

        return [{aid: interpolated_arrays[i, j] for j, aid in enumerate(agent_ids)} 
                for i in range(steps)]

    cpdef add_frames(self, list frames):
        cdef dict frame
        for frame in frames:
            if len(self.buffer) >= self.max_size:
                self.overflow_count += 1
                self.buffer.popleft()
            self.buffer.append(frame)

    cpdef dict get_next_position(self):
        return self.buffer.popleft() if self.buffer else self.current_positions

    cdef void _adjust_buffer_size(self):
        cdef int current_size = len(self.buffer)
        cdef int new_size
        if current_size < self.target_size - 2:
            new_size = min(current_size + 2, self.max_size)
        elif current_size > self.target_size + 2:
            new_size = max(current_size - 2, self.min_size)
        else:
            return

        self.buffer = deque(list(self.buffer)[-new_size:], maxlen=new_size)

    cpdef dict get_stats(self):
        return {
            "current_size": len(self.buffer),
            "target_size": self.target_size,
            "max_size": self.max_size,
            "overflow_count": self.overflow_count
        }

    cpdef add_frame_time(self, double frame_time):
        self.frame_times.append(frame_time)