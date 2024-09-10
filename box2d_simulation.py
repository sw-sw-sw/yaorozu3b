from Box2D import b2World, b2Vec2, b2BodyDef, b2_dynamicBody, b2CircleShape
import numpy as np
import logging
import random
import time
from config_manager import ConfigManager
from log import *
from queue import Empty


class Box2DSimulation:
    def __init__(self, queues):
        self.queues = queues  # この行を追加

        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.bodies = {}
        
        #queues
        self._box2d_to_visual_render = queues['box2d_to_visual_render']
        self._eco_to_box2d_creatures = queues['eco_to_box2d_creatures']
        self._tf_to_box2d  = queues['tf_to_box2d']
        self._box2d_to_tf = queues['box2d_to_tf']
        
        # ConfigManagerから値を取得してプロパティとして設定
        self.config_manager = ConfigManager()
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.random_velocity_interval = self.config_manager.get_trait_value('RANDOM_VELOCITY_INTERVAL')
        self.random_speed = self.config_manager.get_trait_value('RANDOM_SPEED')
        self.dt = self.config_manager.get_trait_value('DT') 

        self.positions = np.zeros((self.max_agents_num, 2), dtype=np.float32)
        self.agent_species = np.zeros(self.max_agents_num, dtype=np.int32)
        self.current_agent_count = 0


        self.last_random_velocity_time = time.time()
  
        
    def initialize(self):
        init_data = self._eco_to_box2d_creatures.get()
        self.positions = init_data['positions']
        self.species = init_data['species']
        self.agent_species = init_data['species']
        self.current_agent_count = init_data['current_agent_count']

        for agent_id in range(self.current_agent_count):
            self._create_body(agent_id)



    #------------------- create body  --------------------    
    
    def create_bodies(self):
        for _ in range(self.max_agents_num):
            creature_info = self._eco_to_box2d_creatures.get()
            self._create_body(creature_info)
        self.current_agent_count = self.max_agents_num
            
        logger.info(f"Created {self.max_agents_num} bodies in Box2D simulation")

    def _create_body(self, agent_id):
        species = self.species[agent_id]
        linear_damping = self.config_manager.get_species_trait_value('DAMPING', species)
        density = self.config_manager.get_species_trait_value('DENSITY', species)
        restitution = self.config_manager.get_species_trait_value('RESTITUTION', species)
        friction = self.config_manager.get_species_trait_value('FRICTION', species)
        mass = self.config_manager.get_species_trait_value('MASS', species)
        radius = self.config_manager.get_species_trait_value('RADIUS', species)
        x = float(self.positions[agent_id][0])
        y = float(self.positions[agent_id][1])        
        body_def = b2BodyDef(
            type=b2_dynamicBody,
            position=b2Vec2(x, y),
            linearVelocity=b2Vec2(0.0, 0.0),
            linearDamping=linear_damping
        )
        body = self.world.CreateBody(body_def)
        circle_shape = b2CircleShape(radius=radius)
        body.CreateFixture(shape=circle_shape, density=density, 
                           friction=friction, restitution=restitution)
        body.mass = mass * circle_shape.radius
        self.bodies[agent_id] = body
        
    # ------------------- remove body --------------------

    def remove_body(self):
        pass

    # ------------------Main routine ---------------------

    def run(self):
        self.update_forces()
        self.step()
        self.update_positions()
        self.send_data_to_tf()
        self.send_data_to_visual()
        
        
    def update_forces(self):
        while not self._tf_to_box2d.empty():
            forces = self._tf_to_box2d.get_nowait()
            for agent_id, force in zip(self.bodies.keys(), forces):
                body = self.bodies[agent_id]
                body.ApplyForceToCenter((float(force[0]), float(force[1])), wake=True)
    
    def step(self):
        self.world.Step(self.dt, 6, 2)
        # self.apply_random_velocity() # random move

    def update_positions(self):
        for agent_id, body in self.bodies.items():
            self.positions[agent_id] = body.position.x, body.position.y

    def send_data_to_tf(self):
        data = {
            'positions': self.positions,
            'agent_species': self.agent_species,
            'count': self.current_agent_count
        }
        self._box2d_to_tf.put(data)
    
    def send_data_to_visual(self):
        visual_data = {
            'positions': self.positions,
            'agent_ids': list(self.bodies.keys()),
            'current_agent_count': self.current_agent_count
        }
        self._box2d_to_visual_render.put(visual_data)
            
    # ----------------- sub method ----------------------
    
    # def get_positions(self):
    #     return np.array([(body.position.x, body.position.y) for body in self.bodies], dtype=np.float32)
    
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
            