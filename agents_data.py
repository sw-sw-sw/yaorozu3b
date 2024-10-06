import numpy as np
from queue import Empty
from log import get_logger
import time, random
import threading
from config_manager import ConfigManager

class AgentsData:
    def __init__(self, queue_dict):
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("Initializing AgentsData")
        self.config_manager = ConfigManager()
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.world_width = self.config_manager.get_trait_value('WORLD_WIDTH')
        self.world_height = self.config_manager.get_trait_value('WORLD_HEIGHT')

        # Define the structured array
        self.agents = np.zeros(self.max_agents_num, dtype=[
            ('id', np.int32),
            ('species', np.int32),
            ('position', np.float32, (2,)),
            ('velocity', np.float32, (2,)),
            ('life_energy', np.float32),
            ('loss_rate', np.float32),
            ('life_gain', np.float32),
            ('birth_threshold', np.float32),
            ('predator_rate', np.float32),
            ('reproduction_rate', np.float32)
        ])

        self.current_agent_count = 0
        self.next_id = 0
        self.available_ids = []

        # Queue setup
        self._eco_to_box2d = queue_dict['eco_to_box2d']
        self._eco_to_visual = queue_dict['eco_to_visual']
        self._eco_to_box2d_init = queue_dict['eco_to_box2d_init']
        self._eco_to_visual_init = queue_dict['eco_to_visual_init']
        self._eco_to_tf_init = queue_dict['eco_to_tf_init']
        self._eco_to_tf = queue_dict['eco_to_tf']
        self._box2d_to_eco = queue_dict['box2d_to_eco']

        self.logger.info(f"AgentsData initialized with max_agents_num: {self.max_agents_num}")

    
    def initialize(self):
        self.logger.warning("Initializing Ecosystem agents")
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
            self.logger.info(f"Initialized {initial_agent_num} agents for species {species}")
                
        self.send_data_to_box2d_initialize()
        self.send_data_to_tf_initialize()
        self.send_data_to_visual_initialize()
        self.logger.info("Ecosystem initialization completed")


    # ------------------ add agent ---------------------
    
    def add_agent(self, species, position, velocity=(0, 0)):
        agent_id = self._add_agent_internal(species, position, velocity)
        if agent_id is not None:
            self._notify_agent_add(agent_id, species, position, velocity)
            self.logger.warning(f"Agent added: id={agent_id}, species={species}, position={position}")
        else:
            self.logger.error(f"Error: Cannot add agent. Maximum capacity of {self.max_agents_num} reached.")
        return agent_id
    
    def add_agent_no_notify(self, species, position, velocity=(0,0)):
        agent_id = self._add_agent_internal(species, position, velocity)
        if agent_id is not None:
            self.logger.debug(f"Agent added without notification: id={agent_id}, species={species}, position={position}")
        return agent_id
    
    def _add_agent_internal(self, species, position, velocity=(0, 0)):
        if self.current_agent_count < self.max_agents_num:
            if self.available_ids:
                agent_id = self.available_ids.pop()
            else:
                agent_id = self.next_id
                self.next_id += 1
            
            index = self.current_agent_count
            self.agents[index]['id'] = agent_id
            self.agents[index]['species'] = species
            self.agents[index]['position'] = position
            self.agents[index]['velocity'] = velocity
            self._set_agent_properties(index, species)
            
            self.current_agent_count += 1

            self.logger.debug(f"Agent added internally: id={agent_id}, species={species}, position={position}")
            return agent_id
        self.logger.warning("Failed to add agent: maximum capacity reached")
        return None

    def _set_agent_properties(self, index, species):
        self.agents[index]['life_energy'] = self.config_manager.get_species_trait_value('LIFE_ENERGY', species)
        self.agents[index]['loss_rate'] = self.config_manager.get_species_trait_value('LIFE_ENERGY_LOSS_RATE', species)
        self.agents[index]['life_gain'] = self.config_manager.get_species_trait_value('LIFE_ENERGY', species)
        self.agents[index]['birth_threshold'] = self.config_manager.get_species_trait_value('BIRTH_THRESHOLD', species)
        self.agents[index]['predator_rate'] = self.config_manager.get_species_trait_value('PREDATOR_RATE', species)
        radius = self.config_manager.get_species_trait_value('RADIUS', species)
        self.agents[index]['reproduction_rate'] = self.config_manager.get_species_trait_value('REPRODUCTION_RATE', species) / radius
        
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
        self.send_data_to_visual(add_data)
        self.logger.debug(f"Notified agent addition: id={agent_id}")

    def remove_agent(self, agent_id):
        index = np.where(self.agents['id'][:self.current_agent_count] == agent_id)[0]
        if len(index) > 0:
            index = index[0]
            last_index = self.current_agent_count - 1
            
            if index != last_index:
                self.agents[index] = self.agents[last_index]
            
            self.agents[last_index] = 0  # Reset the last agent's data
            self.current_agent_count -= 1
            self.available_ids.append(agent_id)
            
            self._notify_agent_removed(agent_id)
            self.logger.info(f"Agent removed: id={agent_id}")
        else:
            self.logger.warning(f"Warning: Agent {agent_id} does not exist. No agent removed.")

    def _notify_agent_removed(self, agent_id):
        remove_data = {
            'action': 'remove',
            'agent_id': agent_id,
            'current_agent_count': self.current_agent_count

        }
        self.send_data_to_visual(remove_data)
        self._eco_to_box2d.put(remove_data)
        self.logger.debug(f"Notified agent removal: id={agent_id}")

    # ----------------- main update ----------------------

    def update(self):
        try:
            if not self._box2d_to_eco.empty():
                box2d_data = self._box2d_to_eco.get_nowait()
                positions = box2d_data['positions']
                if len(positions) == self.current_agent_count:
                    self.agents['position'][:self.current_agent_count] = positions
                else:
                    self.logger.warning(f"Agent count mismatch. Box2D(length): {len(positions)}, AgentsData(current agent count): {self.current_agent_count}")
        except Exception as e:
            self.logger.exception(f"Error in AgentsData update: {e}")
        
    def update_life_energy(self):
        active_agents = self.agents[:self.current_agent_count]
        energy_loss = active_agents['loss_rate']
        active_agents['life_energy'] -= energy_loss
        return np.sum(energy_loss)

    def check_deaths(self):
        active_agents = self.agents[:self.current_agent_count]
        dead_agents = active_agents[active_agents['life_energy'] <= 0]
        energy_to_env = 0
        
        for dead_agent in dead_agents:
            species = dead_agent['species']
            radius = self.config_manager.get_species_trait_value('RADIUS', species)
            energy_to_env += radius ** 2
            self.remove_agent(dead_agent['id'])
        
        return energy_to_env

    def check_reproductions(self):
        active_agents = self.agents[:self.current_agent_count]
        reproducing_agents = active_agents[active_agents['life_energy'] > active_agents['birth_threshold']]
        
        for agent in reproducing_agents:
            if random.random() < agent['reproduction_rate']:
                new_position = (
                    agent['position'][0] + random.uniform(-3, 3),
                    agent['position'][1] + random.uniform(-3, 3)
                )
                new_agent_id = self.add_agent(agent['species'], new_position)
                if new_agent_id is not None:
                    new_agent_index = np.where(self.agents['id'] == new_agent_id)[0][0]
                    agent['life_energy'] /= 2
                    self.agents[new_agent_index]['life_energy'] = agent['life_energy']
                self.logger.warning(f'Reproduction success!! species{agent["species"]} agent_id {agent["id"]} => {new_agent_id}.')
                self.logger.warning(f'species{agent["species"]} pos {agent["position"]} new_pos {new_position}.')
        
    # ----------------- Queues ----------------------

    def send_data_to_tf_initialize(self):
        data = {
            'positions': self.agents['position'],
            'species': self.agents['species'],
            'current_agent_count': self.current_agent_count
        }
        self._eco_to_tf_init.put(data)
        self.logger.info(f"Sent initialization data to TensorFlow. Agent count: {self.current_agent_count}")

    def send_data_to_box2d_initialize(self):
        data = {
            'positions': self.agents['position'][:self.current_agent_count],
            'velocities': self.agents['velocity'][:self.current_agent_count],
            'species': self.agents['species'][:self.current_agent_count],
            'agent_ids': self.agents['id'][:self.current_agent_count],
            'current_agent_count': self.current_agent_count
        }
        self._eco_to_box2d_init.put(data)
        self.logger.info(f"Sent initialization data to Box2D. Agent count: {self.current_agent_count}")

    def send_data_to_visual_initialize(self):
        data = {
            'positions': self.agents['position'][:self.current_agent_count],
            'species': self.agents['species'][:self.current_agent_count],
            'agent_ids': self.agents['id'][:self.current_agent_count],
            'current_agent_count': self.current_agent_count
        }
        self._eco_to_visual_init.put(data)
        self.logger.info(f"Sent initialization data to Visual System. Agent count: {self.current_agent_count}")

    def send_data_to_visual(self, data):
        self._eco_to_visual.put(data)
        self.logger.info(f"Sent data to Visual System. Agent Add or Remove: {self.current_agent_count}")

    def _eco_to_visual_queue(self, data):
        self._eco_to_visual.put(data)

    # ------------------ sub methods ---------------------
    
    def available_agent_ids(self):
        return self.agents['id'][:self.current_agent_count]

    def available_species8_positions(self):
        species8_agents = self.agents[:self.current_agent_count][self.agents['species'][:self.current_agent_count] == 8]
        if len(species8_agents) > 0:
            return random.choice(species8_agents['position'])
        else:
            self.logger.warning("No species 8 agents found.")
            return None
