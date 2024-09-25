import numpy as np
from queue import Empty
from log import get_logger
import time
from config_manager import ConfigManager
from delayed_queue import QueueItem, DelayedQueue

logger = get_logger(__name__)

class AgentsData:
    def __init__(self, queue_dict):
        logger.info("Initializing AgentsData")
        self.config_manager = ConfigManager()
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')
        self.positions = np.zeros((self.max_agents_num, 2), dtype=np.float32)
        self.velocities = np.zeros((self.max_agents_num, 2), dtype=np.float32)  # New velocity array
        self.species = np.zeros(self.max_agents_num, dtype=np.int32)
        self.agent_ids = np.full(self.max_agents_num, -1, dtype=np.int32)
        self.current_agent_count = 0
        self.next_id = 0
        self.available_ids = []
        
        # New properties
        self.life_energy = np.zeros(self.max_agents_num, dtype=np.float32)
        self.loss_rate = np.zeros(self.max_agents_num, dtype=np.float32)
        self.life_gain = np.zeros(self.max_agents_num, dtype=np.float32)
        self.birth_threshold = np.zeros(self.max_agents_num, dtype=np.float32)
        self.predator_rate = np.zeros(self.max_agents_num, dtype=np.float32)
        self.reproduction_rate = np.zeros(self.max_agents_num, dtype=np.float32)
        
        self._eco_to_box2d = queue_dict['eco_to_box2d']
        self._eco_to_visual = queue_dict['eco_to_visual']
        self._eco_to_box2d_init = queue_dict['eco_to_box2d_init']
        self._eco_to_visual_init = queue_dict['eco_to_visual_init']
        self._eco_to_tf_init = queue_dict['eco_to_tf_init']
        self._eco_to_tf = queue_dict['eco_to_tf']
        self._box2d_to_eco = queue_dict['box2d_to_eco']
        
        self.delayed_queue = DelayedQueue()
        self._delay_time = 0.1

        logger.info(f"AgentsData initialized with max_agents_num: {self.max_agents_num}")

    # ----------------- main update ----------------------
    
    def initialize(self):
        logger.info("Initializing Ecosystem agents")
        octagon_radius = self.world_width / 4
        octagon_centers = []
        
        for i in range(8):
            angle = i * np.pi / 4
            x = self.world_width / 2 + octagon_radius * np.cos(angle)
            y = self.world_height / 2 + octagon_radius * np.sin(angle)
            octagon_centers.append((x, y))

        # 各種ごとにエージェントの位置とspeciesを初期化
        for species in range(1, 9):
            initial_agent_num = self.config_manager.get_species_trait_value('INITIAL_AGENT_NUM', species)
            center_x, center_y = octagon_centers[species - 1]
            circle_radius = self.world_width / 5

            # 正規分布を使用して円内にランダムな位置を生成
            r = np.random.randint(0, circle_radius / 2, initial_agent_num)
            theta = np.random.uniform(0, 2 * np.pi, initial_agent_num)
            x = center_x + r * np.cos(theta)
            y = center_y + r * np.sin(theta)
            for i in range(initial_agent_num):
                self.add_agent_no_notify(species, (float(x[i]), float(y[i])))
            logger.info(f"Initialized {initial_agent_num} agents for species {species}")
                
        self.send_data_to_box2d_initialize()
        self.send_data_to_tf_initialize()
        self.send_data_to_visual_initialize()
        logger.info("Ecosystem initialization completed")
        
    def update(self):
        try:
            # data from Box2DSimulation
            if not self._box2d_to_eco.empty():
                box2d_data = self._box2d_to_eco.get_nowait()                
                # update ecosystem's property
                positions = box2d_data['positions']
                if len(positions) == self.current_agent_count:
                    self.positions[:self.current_agent_count] = positions
                    logger.debug(f"Updated agent data. Current agent count: {self.current_agent_count}")
                else:
                    # logger.warning(f"Agent count mismatch. Box2D: {len(box2d_data['positions'])}, AgentsData: {self.current_agent_count}")
                    pass
        except Exception as e:
            logger.exception(f"Error in Ecosystem update: {e}")
        
        self.delayed_queue.update()

    # ------------------ add agent ---------------------
    def add_agent_delay(self, species, position, velocity, delay_time):
        self.delayed_queue.add((species, position, velocity), delay_time, self.add_agent_arg)
        
    def add_agent_arg(self, args):
        species, position, velocity = args
        self.add_agent(species, position, velocity)
    
    def add_agent(self, species, position, velocity=(0, 0)):
        agent_id = self._add_agent_internal(species, position, velocity)
        if agent_id is not None:
            self._initialize_agent_properties(agent_id, species)
            self._notify_agent_add(agent_id, species, position, velocity)
            logger.info(f"Agent added: id={agent_id}, species={species}, position={position}")
        else:
            logger.error(f"Error: Cannot add agent. Maximum capacity of {self.max_agents_num} reached.")
        return agent_id

    def _add_agent_internal(self, species, position, velocity=(0, 0)):
        if self.current_agent_count < self.max_agents_num:
            if self.available_ids:
                agent_id = self.available_ids.pop()
            else:
                agent_id = self.next_id
                self.next_id += 1
            
            index = self.current_agent_count
            self.positions[index] = np.array(position, dtype=np.float32)
            self.velocities[index] = np.array(velocity, dtype=np.float32)
            self.species[index] = species
            self.agent_ids[index] = agent_id
            self.current_agent_count += 1

            logger.debug(f"Agent added internally: id={agent_id}, species={species}, position={position}")
            return agent_id
        logger.warning("Failed to add agent: maximum capacity reached")
        return None

    def _initialize_agent_properties(self, agent_id, species):
        index = np.where(self.agent_ids == agent_id)[0][0]
        radius = self.config_manager.get_species_trait_value('RADIUS', species)
        self.life_energy[index] = radius * 1000
        self.loss_rate[index] = radius
        self.life_gain[index] = self.life_energy[index] / 500
        self.birth_threshold[index] = self.config_manager.get_species_trait_value('BIRTH_THRESHOLD', species)
        self.predator_rate[index] = self.config_manager.get_species_trait_value('PREDATOR_RATE', species)
        self.reproduction_rate[index] = self.config_manager.get_species_trait_value('REPRODUCTION_RATE', species) / radius

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
        self.send_data_to_visual_delay(add_data, self._delay_time)
        logger.debug(f"Notified agent addition: id={agent_id}")

    def remove_agent(self, agent_id):
        index = np.where(self.agent_ids[:self.current_agent_count] == agent_id)[0]
        if len(index) > 0:
            index = index[0]
            last_index = self.current_agent_count - 1
            
            if index != last_index:
                self._swap_agent_data(index, last_index)
            
            self._reset_agent_data(last_index)
            self.current_agent_count -= 1
            self.available_ids.append(agent_id)
            
            self._notify_agent_removed(agent_id)
            logger.info(f"Agent removed: id={agent_id}")
        else:
            logger.warning(f"Warning: Agent {agent_id} does not exist. No agent removed.")

    def _swap_agent_data(self, index1, index2):
        for arr in [self.positions, self.velocities, self.species, self.agent_ids, 
                    self.life_energy, self.loss_rate, self.life_gain, 
                    self.birth_threshold, self.predator_rate, self.reproduction_rate]:
            arr[index1], arr[index2] = arr[index2], arr[index1]

    def _reset_agent_data(self, index):
        self.positions[index] = np.zeros(2, dtype=np.float32)
        self.velocities[index] = np.zeros(2, dtype=np.float32)
        self.species[index] = 0
        self.agent_ids[index] = -1
        self.life_energy[index] = 0
        self.loss_rate[index] = 0
        self.life_gain[index] = 0
        self.birth_threshold[index] = 0
        self.predator_rate[index] = 0
        self.reproduction_rate[index] = 0

    def _notify_agent_removed(self, agent_id):
        remove_data = {
            'action': 'remove',
            'agent_id': agent_id,
            'current_agent_count': self.current_agent_count

        }
        self.send_data_to_visual_delay(remove_data, self._delay_time)
        # self._eco_to_visual.put(remove_data)
        self._eco_to_box2d.put(remove_data)
        logger.debug(f"Notified agent removal: id={agent_id}")

    # ----------------- Queues ----------------------

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

    def send_data_to_visual_delay(self, _data, _delay_time):
        self.delayed_queue.add(_data, _delay_time, self._eco_to_visual_queue)
        logger.info(f"Sent data to Visual System. Agent Add or Remove: {self.current_agent_count}")

    def _eco_to_visual_queue(self, data):
        self._eco_to_visual.put(data)

    # ------------------ sub methods ---------------------
    
    def available_agent_ids(self):
        return self.agent_ids[:self.current_agent_count]