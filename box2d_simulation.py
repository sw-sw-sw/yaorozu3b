from Box2D import b2World, b2Vec2, b2BodyDef, b2_dynamicBody, b2CircleShape
import numpy as np
import random
import time
from config_manager import ConfigManager
from log import get_logger
from queue import Empty

logger = get_logger()

class Box2DSimulation:
    def __init__(self, queues):
        self.queues = queues
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.bodies = {}
        
        # Queues
        self._eco_to_box2d_init = queues['eco_to_box2d_init']
        self._eco_to_box2d = queues['eco_to_box2d']
        self._tf_to_box2d = queues['tf_to_box2d']
        self._box2d_to_tf = queues['box2d_to_tf']
        self._box2d_to_eco = queues['box2d_to_eco'] 

        # ConfigManager setup
        self.config_manager = ConfigManager()
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.random_velocity_interval = self.config_manager.get_trait_value('RANDOM_VELOCITY_INTERVAL')
        self.random_speed = self.config_manager.get_trait_value('RANDOM_SPEED')
        self.dt = self.config_manager.get_trait_value('DT') 

        # Initialize arrays
        self.positions = np.zeros((self.max_agents_num, 2), dtype=np.float32)
        self.species = np.zeros(self.max_agents_num, dtype=np.int32)
        self.agent_ids = np.zeros(self.max_agents_num, dtype=np.int32)
        self.forces = np.zeros((self.max_agents_num, 2), dtype=np.float32)

        self.current_agent_count = 0
        self.last_random_velocity_time = time.time()
        logger.info("Box2DSimulation initialized")
  
    def initialize(self):
        logger.info("Box2DSimulation is initializing")
        init_data = self._eco_to_box2d_init.get()
        self.current_agent_count = init_data['current_agent_count']
        self.positions[:self.current_agent_count] = init_data['positions']
        self.species[:self.current_agent_count] = init_data['species']
        self.agent_ids[:self.current_agent_count] = init_data['agent_ids']
        
        for i in range(self.current_agent_count):
            self._create_body(self.agent_ids[i], self.species[i], self.positions[i])
        
        logger.info(f"Box2DSimulation initialized with {self.current_agent_count} agents")

    def _create_body(self, agent_id, species, position):
        linear_damping = self.config_manager.get_species_trait_value('DAMPING', species)
        density = self.config_manager.get_species_trait_value('DENSITY', species)
        restitution = self.config_manager.get_species_trait_value('RESTITUTION', species)
        friction = self.config_manager.get_species_trait_value('FRICTION', species)
        mass = self.config_manager.get_species_trait_value('MASS', species)
        radius = self.config_manager.get_species_trait_value('RADIUS', species)
        
        body_def = b2BodyDef(
            type=b2_dynamicBody,
            position=b2Vec2(float(position[0]), float(position[1])),
            linearVelocity=b2Vec2(0.0, 0.0),
            linearDamping=linear_damping
        )
        body = self.world.CreateBody(body_def)
        circle_shape = b2CircleShape(radius=radius)
        body.CreateFixture(shape=circle_shape, density=density, 
                           friction=friction, restitution=restitution)
        body.mass = mass * circle_shape.radius
        self.bodies[agent_id] = body
        logger.debug(f"Created body for agent {agent_id} of species {species}")
        
    def update(self):
        self.process_ecosystem_queue()
        self.update_forces()
        self.step()
        self.update_positions()
        self.send_data_to_tf()
        self.send_data_to_eco()
    
    def process_ecosystem_queue(self):
        while not self._eco_to_box2d.empty():
            try:
                update_data = self._eco_to_box2d.get_nowait()
                action = update_data.get('action')
                if action == 'add':
                    self._handle_agent_added(update_data)
                elif action == 'remove':
                    self._handle_agent_removed(update_data)
                else:
                    logger.warning(f"Box2DSimulation: Unknown action received: {action}")
            
            except Exception as e:
                logger.exception(f"Box2DSimulation: Error processing ecosystem queue: {e}")

    def _handle_agent_added(self, data):
        agent_id = data['agent_id']
        species = data['species']
        position = data['position']
        self._create_body(agent_id, species, position)
        self.current_agent_count += 1
        index = self.current_agent_count - 1
        self.positions[index] = position
        self.species[index] = species
        self.agent_ids[index] = agent_id
        logger.debug(f"Agent {agent_id} added to Box2D simulation. Total agents: {self.current_agent_count}")
        
    def _handle_agent_removed(self, data):
        agent_id = data['agent_id']
        if agent_id in self.bodies:
            self.world.DestroyBody(self.bodies[agent_id])
            del self.bodies[agent_id]
            index = np.where(self.agent_ids[:self.current_agent_count] == agent_id)[0][0]
            self.agent_ids[index:-1] = self.agent_ids[index+1:]
            self.positions[index:-1] = self.positions[index+1:]
            self.species[index:-1] = self.species[index+1:]
            self.current_agent_count -= 1
            logger.info(f"Box2DSimulation:Agent {agent_id} removed from Box2D. Total agents: {self.current_agent_count}")
        else:
            logger.warning(f"Box2DSimulation:Attempted to remove non-existent agent {agent_id} from Box2D")

    def update_forces(self):
        while not self._tf_to_box2d.empty():
            data = self._tf_to_box2d.get()
            current_agent_count = data['current_agent_count']
            forces = data['forces'][:current_agent_count]
            if current_agent_count != self.current_agent_count:
                logging.warning(f"Warning: Agent count mismatch in Box2D. Expected {self.current_agent_count}, got {current_agent_count}. Skipping update.")
                break
            
            for agent_id, force in zip(self.bodies.keys(), forces):
                body = self.bodies[agent_id]
                body.ApplyForceToCenter((float(force[0]), float(force[1])), wake=True)
                    # self.forces = forces
                
    def step(self):
        self.world.Step(self.dt, 6, 2)

    def update_positions(self):
        for i, agent_id in enumerate(self.agent_ids[:self.current_agent_count]):
            if agent_id in self.bodies:
                body = self.bodies[agent_id]
                self.positions[i] = body.position.x, body.position.y

    def send_data_to_tf(self):
        data = {
            'positions': self.positions.tolist(),  # numpy配列をリストに変換
            'species': self.species.tolist(),      # numpy配列をリストに変換
            'current_agent_count': int(self.current_agent_count)  # intに変換
        }
        self._box2d_to_tf.put(data)
    
    def send_data_to_visual_render(self):
        visual_data = {
            'positions': self.positions[:self.current_agent_count],
            'current_agent_count': self.current_agent_count,
            'agent_ids': self.agent_ids,
        }
        self._eco_to_visual_render.put(visual_data)
    
    def send_data_to_eco(self):
        eco_data = {
            'positions': self.positions[:self.current_agent_count],
            'current_agent_count': self.current_agent_count,
            'agent_ids': self.agent_ids,
        }
        self._box2d_to_eco.put(eco_data)

    def apply_random_velocity(self):
        current_time = time.time()
        if current_time - self.last_random_velocity_time >= self.random_velocity_interval:
            if self.bodies:
                random_body = random.choice(list(self.bodies.values()))
                random_velocity = b2Vec2(
                    random.uniform(-self.random_speed, self.random_speed),
                    random.uniform(-self.random_speed, self.random_speed)
                )
                random_body.linearVelocity = random_velocity
                self.last_random_velocity_time = current_time