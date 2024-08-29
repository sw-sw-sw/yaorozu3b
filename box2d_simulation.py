from Box2D import b2World, b2Vec2, b2BodyDef, b2_dynamicBody, b2CircleShape
from config import *
import numpy as np
import logging
import random
import time
from dna_manager import DNAManager

logger = logging.getLogger(__name__)

class Box2DSimulation:
    def __init__(self, queues):
        self.dna_manager = DNAManager()
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.bodies = []
        self._rendering_queue = queues['rendering_queue']
        self._eco_to_box2d_creatures = queues['eco_to_box2d_creatures']
        self.positions = np.zeros((MAX_AGENTS_NUM, 2), dtype=np.float32)

    #------------------ random move ---------------------
            
        self.last_random_velocity_time = time.time() # for radom velocity
        self.random_velocity_interval = RANDOM_VELOCITY_INTERVAL
        self.random_speed= RANDOM_SPEED
        
    #------------------- initialize --------------------    
    def create_bodies(self):
        for _ in range(MAX_AGENTS_NUM):
            creature_info = self._eco_to_box2d_creatures.get()
            self._create_body(creature_info)
        logger.info(f"Created {MAX_AGENTS_NUM} bodies in Box2D simulation")

    def _create_body(self, creature_info):
        species =creature_info['agent_species']
        linear_damping = self.dna_manager[species].get_trait_value('DAMPING')    
        density = self.dna_manager[species].get_trait_value('DENSITY') 
        restitution = self.dna_manager[species].get_trait_value('RESTITUTION') 
        friction = self.dna_manager[species].get_trait_value('FRICTION') 
        mass = self.dna_manager[species].get_trait_value('MASS') 
        
        body_def = b2BodyDef(
            type=b2_dynamicBody,
            position=b2Vec2(creature_info['x'], creature_info['y']),
            linearVelocity=b2Vec2(0, 0),
            linearDamping=linear_damping
        )
        body = self.world.CreateBody(body_def)
        circle_shape = b2CircleShape(radius=creature_info['radius'])
        body.CreateFixture(shape=circle_shape, density=density, 
                           friction=friction, restitution=restitution)
        body.mass = mass * creature_info['radius']
        self.bodies.append(body)
        
    # ------------------ update ---------------------
    def apply_forces_to_box2d(self, _forces):
        forces = np.frombuffer(_forces.get_obj(), dtype=np.float32).reshape((MAX_AGENTS_NUM, 2))
        for body, force in zip(self.bodies, forces):
            body.ApplyForceToCenter((float(force[0]), float(force[1])), wake=True)

    def apply_positions_to_shared_memory(self, _positions):
        positions = self.get_positions()
        np.frombuffer(_positions.get_obj(), dtype=np.float32).reshape((MAX_AGENTS_NUM, 2))[:] = positions

    def get_positions(self):
        return np.array([(body.position.x, body.position.y) for body in self.bodies], dtype=np.float32)

    def add_positions_to_render_queue(self):
        if self._rendering_queue.empty():
            self._rendering_queue.put(self.get_positions())
            
    # -----------------random move ----------------------
    
    def apply_random_velocity(self):
        current_time = time.time()
        if current_time - self.last_random_velocity_time >= self.random_velocity_interval:
            random_body = random.choice(self.bodies)
            random_velocity = b2Vec2(
                random.uniform(-self.random_speed, self.random_speed),
                random.uniform(-self.random_speed, self.random_speed)
            )
            random_body.linearVelocity = random_velocity
            self.last_random_velocity_time = current_time
            
    # -----------------to renderer ----------------------
    
    def step(self):
        self.world.Step(DT, 6, 2)
        self.apply_random_velocity() # random move
