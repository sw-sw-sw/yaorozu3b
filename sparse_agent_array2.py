import numpy as np
import multiprocessing as mp
import tensorflow as tf
from typing import Dict, Any, List

agent_dtype = np.dtype([
    ('id', np.int32),
    ('species', np.int32),
    ('position', np.float32, (2,)),
    ('velocity', np.float32, (2,)),
    ('force', np.float32, (2,))
])

def setup_shared_memory(max_agents: int) -> Dict[str, Any]:
    return {
        'agents_data': mp.RawArray('B', max_agents * agent_dtype.itemsize),
        'count': mp.Value('i', 0),
        'tf_time': mp.Value('d', 0.0),
        'box2d_time': mp.Value('d', 0.0),
        'lock': mp.Lock()
    }

def get_numpy_array(shared_memory: Dict[str, Any]) -> np.ndarray:
    return np.frombuffer(shared_memory['agents_data'], dtype=agent_dtype)

def add_agents_batch(shared_memory: Dict[str, Any], agent_data_list: List[np.void]) -> None:
    with shared_memory['lock']:
        agents = get_numpy_array(shared_memory)
        current_count = shared_memory['count'].value
        new_count = min(current_count + len(agent_data_list), len(agents))
        agents[current_count:new_count] = agent_data_list[:new_count-current_count]
        shared_memory['count'].value = new_count

def remove_agents_batch(shared_memory: Dict[str, Any], agent_ids: List[int]) -> None:
    with shared_memory['lock']:
        agents = get_numpy_array(shared_memory)
        current_count = shared_memory['count'].value
        mask = np.isin(agents['id'][:current_count], agent_ids, invert=True)
        agents[:mask.sum()] = agents[:current_count][mask]
        shared_memory['count'].value = mask.sum()

def update_agents_batch(shared_memory: Dict[str, Any], agent_data_list: List[np.void]) -> None:
    with shared_memory['lock']:
        agents = get_numpy_array(shared_memory)
        current_count = shared_memory['count'].value
        id_to_index = {agent['id']: i for i, agent in enumerate(agents[:current_count])}
        for agent_data in agent_data_list:
            if agent_data['id'] in id_to_index:
                agents[id_to_index[agent_data['id']]] = agent_data

def get_all_agents(shared_memory: Dict[str, Any]) -> np.ndarray:
    with shared_memory['lock']:
        agents = get_numpy_array(shared_memory)
        return agents[:shared_memory['count'].value].copy()

#------------------ tf ---------------------

def get_tf_positions(shared_memory: Dict[str, Any]) -> tf.Tensor:
    with shared_memory['lock']:
        agents = get_numpy_array(shared_memory)
        return tf.convert_to_tensor(agents['position'][:shared_memory['count'].value], dtype=tf.float32)

def get_tf_species(shared_memory: Dict[str, Any]) -> tf.Tensor:
    with shared_memory['lock']:
        agents = get_numpy_array(shared_memory)
        return tf.convert_to_tensor(agents['species'][:shared_memory['count'].value], dtype=tf.int32)

def set_forces(shared_memory, forces):
    # with shared_memory['lock']:
    np_array = np.frombuffer(shared_memory['forces'].get_obj(), dtype=np.float32)
    np_array[:] = forces.flatten()
    
# ---------------------------------------

def update_forces(shared_memory: Dict[str, Any], forces: tf.Tensor) -> None:
    with shared_memory['lock']:
        agents = get_numpy_array(shared_memory)
        agents['force'][:shared_memory['count'].value] = forces.numpy()

def update_positions_and_velocities(shared_memory: Dict[str, Any], positions: np.ndarray, velocities: np.ndarray) -> None:
    with shared_memory['lock']:
        agents = get_numpy_array(shared_memory)
        agents['position'][:shared_memory['count'].value] = positions
        agents['velocity'][:shared_memory['count'].value] = velocities

def save_simulation_state(shared_memory: Dict[str, Any], filename: str) -> None:
    with shared_memory['lock']:
        agents = get_numpy_array(shared_memory)
        state = {
            'agents_data': agents[:shared_memory['count'].value],
            'count': shared_memory['count'].value,
            'tf_time': shared_memory['tf_time'].value,
            'box2d_time': shared_memory['box2d_time'].value
        }
        np.save(filename, state)

def load_simulation_state(shared_memory: Dict[str, Any], filename: str) -> None:
    state = np.load(filename, allow_pickle=True).item()
    with shared_memory['lock']:
        agents = get_numpy_array(shared_memory)
        agents[:len(state['agents_data'])] = state['agents_data']
        shared_memory['count'].value = state['count']
        shared_memory['tf_time'].value = state['tf_time']
        shared_memory['box2d_time'].value = state['box2d_time']

# Usage
shared_memory = setup_shared_memory(MAX_AGENTS_NUM)