from Box2D import b2World, b2Vec2, b2BodyDef, b2_dynamicBody, b2CircleShape, b2ContactListener
import numpy as np
import random
import time
from config_manager import ConfigManager
from log import get_logger, set_log_level
from queue import Empty



class CollisionListener(b2ContactListener):
    def __init__(self):
        super().__init__()
        self.collisions = set()

    def BeginContact(self, contact):
        body_a = contact.fixtureA.body
        body_b = contact.fixtureB.body
        collision = tuple(sorted((body_a.userData, body_b.userData)))
        self.collisions.add(collision)

    def clear(self):
        self.collisions.clear()

class Box2DSimulation:
    def __init__(self, queues):
        self.logger = get_logger(self.__class__.__name__)
        self.queues = queues
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.collision_listener = CollisionListener()
        self.world.contactListener = self.collision_listener
        self.bodies = {}
        
        # Queues
        self._eco_to_box2d_init = queues['eco_to_box2d_init']
        self._eco_to_box2d = queues['eco_to_box2d']
        self._tf_to_box2d = queues['tf_to_box2d']
        self._box2d_to_tf = queues['box2d_to_tf']
        self._box2d_to_eco = queues['box2d_to_eco']
        self._box2d_to_visual_render = queues['box2d_to_visual_render']

        self._box2d_to_eco_collisions = queues['box2d_to_eco_collisions']  # New queue for collision data

        # ConfigManager setup
        self.config_manager = ConfigManager()
        self.dt = self.config_manager.get_trait_value('DT')
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')

        # Initialize numpy arrays
        self.agent_ids = np.full(self.max_agents_num, -1, dtype=np.int32)
        self.species = np.zeros(self.max_agents_num, dtype=np.int32)
        self.positions = np.zeros((self.max_agents_num, 2), dtype=np.float32)
        self.current_agent_count = 0

        # reduce collision data
        self.frame_counter = 0
        self.collision_send_interval = 1
        self.collision_sample_size = 1
        
        self.logger.info("Box2DSimulation initialized")
  
    def initialize(self):
        self.logger.info("Box2DSimulation is initializing")
        init_data = self._eco_to_box2d_init.get()
        
        self.current_agent_count = init_data['current_agent_count']
        self.agent_ids[:self.current_agent_count] = init_data['agent_ids']
        self.species[:self.current_agent_count] = init_data['species']
        
        for i in range(self.current_agent_count):
            agent_id = self.agent_ids[i]
            species = self.species[i]
            position = init_data['positions'][i]
            velocity = init_data['velocities'][i]
            self._create_body(agent_id, species, position, velocity)
        
        self.logger.info(f"Box2DSimulation initialized with {self.current_agent_count} agents")

    def _create_body(self, agent_id, species, position, velocity=(0, 0)):
        linear_damping = self.config_manager.get_species_trait_value('DAMPING', species)
        density = self.config_manager.get_species_trait_value('DENSITY', species)
        restitution = self.config_manager.get_species_trait_value('RESTITUTION', species)
        friction = self.config_manager.get_species_trait_value('FRICTION', species)
        mass = self.config_manager.get_species_trait_value('MASS', species)
        radius = self.config_manager.get_species_trait_value('RADIUS', species)
        
        body_def = b2BodyDef(
            type=b2_dynamicBody,
            position=b2Vec2(float(position[0]), float(position[1])),
            linearVelocity=b2Vec2(float(velocity[0]), float(velocity[1])),
            linearDamping=linear_damping
        )
        body = self.world.CreateBody(body_def)
        body.userData = agent_id  # Set agent_id as userData for collision detection
        circle_shape = b2CircleShape(radius=radius)
        body.CreateFixture(shape=circle_shape, density=density, 
                           friction=friction, restitution=restitution)
        body.mass = mass * circle_shape.radius
        self.bodies[agent_id] = body
        self.logger.debug(f"Created body for agent {agent_id} of species {species}")
        
    def update(self):
        self.process_ecosystem_queue()
        self.update_forces()
        self.step()
        self.update_positions()
        self.send_data_to_tf()
        self.send_data_to_eco_visual()
        
        self.frame_counter += 1
        if self.frame_counter % self.collision_send_interval == 0:
            self.send_collision_data_to_eco()

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
                    self.logger.warning(f"Box2DSimulation: Unknown action received: {action}")
            
            except Exception as e:
                self.logger.exception(f"Box2DSimulation: Error processing ecosystem queue: {e}")

    def _handle_agent_added(self, data):
        agent_id = data['agent_id']
        species = data['species']
        position = data['position']
        velocity = data.get('velocity', (0, 0))
        self._create_body(agent_id, species, position, velocity)

        # Update numpy arrays
        index = self.current_agent_count
        self.agent_ids[index] = agent_id
        self.species[index] = species
        self.current_agent_count += 1

        self.logger.debug(f"Agent {agent_id} added to Box2D simulation.")

    def _handle_agent_removed(self, data):
        agent_id = data['agent_id']
        if agent_id in self.bodies:
            self.world.DestroyBody(self.bodies[agent_id])
            del self.bodies[agent_id]
            
            # Update numpy arrays
            index = np.where(self.agent_ids == agent_id)[0][0]
            self.agent_ids[index:-1] = self.agent_ids[index+1:]
            self.species[index:-1] = self.species[index+1:]
            self.agent_ids[self.current_agent_count-1] = -1
            self.species[self.current_agent_count-1] = 0
            self.current_agent_count -= 1
            
            self.logger.info(f"Box2DSimulation: Agent {agent_id} removed from Box2D.")
        else:
            self.logger.warning(f"Box2DSimulation: Attempted to remove non-existent agent {agent_id} from Box2D")

    def update_forces(self):
        try:
            while True:
                data = self._tf_to_box2d.get_nowait()
                forces = data['forces']
                for agent_id, force in zip(self.agent_ids[:self.current_agent_count], forces):
                    if agent_id in self.bodies:
                        body = self.bodies[agent_id]
                        body.ApplyForceToCenter((float(force[0]), float(force[1])), wake=True)
        except Empty:
            pass
                    
    def step(self):
        self.world.Step(self.dt, 36, 18 )

    def update_positions(self):
        for i, agent_id in enumerate(self.agent_ids[:self.current_agent_count]):
            if agent_id in self.bodies:
                body = self.bodies[agent_id]
                self.positions[i] = body.position.x, body.position.y
            else:
                self.logger.warning(f"Body for agent {agent_id} not found. Using last known position.")

    def send_data_to_tf(self):
        
        data = {
            'positions': self.positions,
            'species': self.species,
            'current_agent_count': self.current_agent_count
        }
        self._box2d_to_tf.put(data)

    def send_data_to_eco_visual(self):
        data = {
            'positions': self.positions[:self.current_agent_count],
            'agent_ids': self.agent_ids[:self.current_agent_count],
        }
        self._box2d_to_eco.put(data)
        self._box2d_to_visual_render.put(data)


    def send_collision_data_to_eco(self):
        all_collisions = self.collision_listener.collisions
        total_collisions = len(all_collisions)
        
        if total_collisions > self.collision_sample_size:
            sampled_collisions = random.sample(list(all_collisions), self.collision_sample_size)
        else:
            sampled_collisions = all_collisions
            
        collision_data = {
            'collisions': sampled_collisions,
        }

        self._box2d_to_eco_collisions.put(collision_data)
        self.collision_listener.clear()  # 衝突データをクリア
        
        self.logger.debug(f"Sent sampled collision data to Ecosystem: {len(sampled_collisions)} out of {total_collisions} collisions")
