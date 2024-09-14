import numpy as np
from queue import Empty

class AgentsData:
    def __init__(self, max_agents_num, queue_dict):
        self.max_agents_num = max_agents_num
        self.positions = np.zeros((max_agents_num, 2), dtype=np.float32)
        self.species = np.zeros(max_agents_num, dtype=np.int32)
        self.agent_ids = np.full(max_agents_num, -1, dtype=np.int32)
        self.current_agent_count = 0
        self.next_id = 0
        self.available_ids = []
        
        self._eco_to_box2d = queue_dict['eco_to_box2d']
        self._eco_to_visual = queue_dict['eco_to_visual']
        self._eco_to_box2d_init = queue_dict['eco_to_box2d_init']
        self._eco_to_visual_init = queue_dict['eco_to_visual_init']
        self._eco_to_tf_init = queue_dict['eco_to_tf_init']
        self._box2d_to_eco = queue_dict['box2d_to_eco']
        self._eco_to_visual_render = queue_dict['eco_to_visual_render']

    def _add_agent_internal(self, species, position):
        if self.current_agent_count < self.max_agents_num:
            if self.available_ids:
                agent_id = self.available_ids.pop()
            else:
                agent_id = self.next_id
                self.next_id += 1
            
            index = self.current_agent_count
            self.positions[index] = np.array(position, dtype=np.float32)
            self.species[index] = species
            self.agent_ids[index] = agent_id
            self.current_agent_count += 1

            return agent_id
        return None

    def add_agent(self, species, position):
        agent_id = self._add_agent_internal(species, position)
        if agent_id is not None:
            self._notify_agent_add(agent_id, species, position)
        else:
            print(f"Error: Cannot add agent. Maximum capacity of {self.max_agents_num} reached.")
        return agent_id

    def add_agent_no_notify(self, species, position):
        return self._add_agent_internal(species, position)

    def _notify_agent_add(self, agent_id, species, position):
        add_data = {
            'action': 'add',
            'agent_id': agent_id,
            'species': species,
            'position': position,
        }
        self._eco_to_box2d.put(add_data)
        self._eco_to_visual.put(add_data)

    def remove_agent(self, agent_id):
        index = np.where(self.agent_ids[:self.current_agent_count] == agent_id)[0]
        if len(index) > 0:
            index = index[0]
            last_index = self.current_agent_count - 1
            
            if index != last_index:
                self.positions[index] = self.positions[last_index]
                self.species[index] = self.species[last_index]
                self.agent_ids[index] = self.agent_ids[last_index]
            self.positions[last_index] = np.zeros(2, dtype=np.float32)
            self.species[last_index] = 0
            self.agent_ids[last_index] = -1
            self.current_agent_count -= 1
            self.available_ids.append(agent_id)
            
            self._notify_agent_removed(agent_id)
        else:
            print(f"Warning: Agent {agent_id} does not exist. No agent removed.")

    def _notify_agent_removed(self, agent_id):
        remove_data = {
            'action': 'remove',
            'agent_id': agent_id,
        }
        self._eco_to_visual.put(remove_data)
        self._eco_to_box2d.put(remove_data)

    def update_from_box2d(self):
        try:
            data = self._box2d_to_eco.get_nowait()
            self.positions[:self.current_agent_count] = data['positions']
            self.current_agent_count = data['current_agent_count']
            return True
        except Empty:
            return False

    def send_data_to_visual(self):
        visual_data = {
            'positions': self.positions[:self.current_agent_count],
            'agent_ids': self.agent_ids[:self.current_agent_count],
            'current_agent_count': self.current_agent_count
        }
        self._eco_to_visual_render.put(visual_data)
        return visual_data

    def send_data_to_tf_initialize(self):
        data = {
            'positions': self.positions,
            'species': self.species,
            'current_agent_count': self.current_agent_count
        }
        self._eco_to_tf_init.put(data)

    def send_data_to_box2d_initialize(self):
        data = {
            'positions': self.positions[:self.current_agent_count],
            'species': self.species[:self.current_agent_count],
            'agent_ids': self.agent_ids[:self.current_agent_count],
            'current_agent_count': self.current_agent_count
        }
        self._eco_to_box2d_init.put(data)

    def send_data_to_visual_initialize(self):
        data = {
            'positions': self.positions[:self.current_agent_count],
            'species': self.species[:self.current_agent_count],
            'agent_ids': self.agent_ids[:self.current_agent_count],
            'current_agent_count': self.current_agent_count
        }
        self._eco_to_visual_init.put(data)

    def available_agent_ids(self):
        return self.agent_ids[:self.current_agent_count]