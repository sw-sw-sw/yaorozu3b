import numpy as np
from queue import Empty
from log import get_logger
import time

logger = get_logger()

class AgentsData:
    def __init__(self, max_agents_num, queue_dict):
        logger.info("Initializing AgentsData")
        self.max_agents_num = max_agents_num
        self.positions = np.zeros((max_agents_num, 2), dtype=np.float32)
        self.velocities = np.zeros((max_agents_num, 2), dtype=np.float32)  # New velocity array
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
        self._eco_to_tf = queue_dict['eco_to_tf']
        self._box2d_to_eco = queue_dict['box2d_to_eco']
        self._eco_to_visual_render = queue_dict['eco_to_visual_render']
        logger.info(f"AgentsData initialized with max_agents_num: {max_agents_num}")
        
    def _add_agent_internal(self, species, position, velocity=(0, 0)):
        if self.current_agent_count < self.max_agents_num:
            if self.available_ids:
                agent_id = self.available_ids.pop()
            else:
                agent_id = self.next_id
                self.next_id += 1
            
            index = self.current_agent_count
            self.positions[index] = np.array(position, dtype=np.float32)
            self.velocities[index] = np.array(velocity, dtype=np.float32)  # Set initial velocity

            self.species[index] = species
            self.agent_ids[index] = agent_id
            self.current_agent_count += 1

            logger.debug(f"Agent added internally: id={agent_id}, species={species}, position={position}")
            return agent_id
        logger.warning("Failed to add agent: maximum capacity reached")
        return None

    def add_agent(self, species, position, velocity=(0, 0)):
        agent_id = self._add_agent_internal(species, position, velocity)
        if agent_id is not None:
            self._notify_agent_add(agent_id, species, position, velocity)
            logger.info(f"Agent added: id={agent_id}, species={species}, position={position}")
        else:
            logger.error(f"Error: Cannot add agent. Maximum capacity of {self.max_agents_num} reached.")
        return agent_id

    def add_agent_no_notify(self, species, position, velocity=(0,0)):
        agent_id = self._add_agent_internal(species, position)
        if agent_id is not None:
            logger.debug(f"Agent added without notification: id={agent_id}, species={species}, position={position}")
        return agent_id

    def _notify_agent_add(self, agent_id, species, position, velocity):
        add_data = {
            'action': 'add',
            'agent_id': agent_id,
            'species': species,
            'position': position,
            'velocity': velocity,
            'current_agent_count': self.current_agent_count

        }
        self._eco_to_box2d.put(add_data)
        self._eco_to_visual.put(add_data)
        logger.debug(f"Notified agent addition: id={agent_id}")


        
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
            logger.info(f"Agent removed: id={agent_id}")
        else:
            logger.warning(f"Warning: Agent {agent_id} does not exist. No agent removed.")

    def _notify_agent_removed(self, agent_id):
        remove_data = {
            'action': 'remove',
            'agent_id': agent_id,
        }
        self._eco_to_visual.put(remove_data)
        self._eco_to_box2d.put(remove_data)
        logger.debug(f"Notified agent removal: id={agent_id}")

    def update(self):
        try:
            if not self._box2d_to_eco.empty():
                box2d_data = self._box2d_to_eco.get_nowait()
                self.send_data_to_visual(box2d_data)
                
                positions = box2d_data['positions']
                if len(positions) == self.current_agent_count:
                    self.positions[:self.current_agent_count] = positions
                    logger.debug(f"Updated agent data. Current agent count: {self.current_agent_count}")
                else:
                    # logger.warning(f"Agent count mismatch. Box2D: {len(box2d_data['positions'])}, AgentsData: {self.current_agent_count}")
                    pass
        except Exception as e:
            logger.exception(f"Error in Ecosystem update: {e}")

    def send_data_to_visual(self, visual_data):
        self._eco_to_visual_render.put(visual_data)
        logger.debug("Sent data to visual system")

        
    def send_data_to_tf_initialize(self):
        data = {
            'positions': self.positions,
            'species': self.species,
            'current_agent_count': self.current_agent_count
        }
        self._eco_to_tf_init.put(data)
        logger.info(f"Sent initialization data to TensorFlow. Agent count: {self.current_agent_count}")

    def send_data_to_box2d_initialize(self):
        data = {
            'positions': self.positions[:self.current_agent_count],
            'velocities': self.velocities[:self.current_agent_count], 
            'species': self.species[:self.current_agent_count],
            'agent_ids': self.agent_ids[:self.current_agent_count],
            'current_agent_count': self.current_agent_count
        }
        self._eco_to_box2d_init.put(data)
        logger.info(f"Sent initialization data to Box2D. Agent count: {self.current_agent_count}")

    def send_data_to_visual_initialize(self):
        data = {
            'positions': self.positions[:self.current_agent_count],
            'species': self.species[:self.current_agent_count],
            'agent_ids': self.agent_ids[:self.current_agent_count],
            'current_agent_count': self.current_agent_count
        }
        self._eco_to_visual_init.put(data)
        logger.info(f"Sent initialization data to Visual System. Agent count: {self.current_agent_count}")

    def available_agent_ids(self):
        return self.agent_ids[:self.current_agent_count]