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
        self._eco_to_box2d_init = queues['eco_to_box2d_init']
        self._eco_to_box2d = queues['eco_to_box2d']
        self._tf_to_box2d  = queues['tf_to_box2d']
        self._box2d_to_tf = queues['box2d_to_tf']
        self._box2d_to_eco = queues['box2d_to_eco'] 

        # ConfigManagerから値を取得してプロパティとして設定
        self.config_manager = ConfigManager()
        self.max_agents_num = self.config_manager.get_trait_value('MAX_AGENTS_NUM')
        self.random_velocity_interval = self.config_manager.get_trait_value('RANDOM_VELOCITY_INTERVAL')
        self.random_speed = self.config_manager.get_trait_value('RANDOM_SPEED')
        self.dt = self.config_manager.get_trait_value('DT') 

        self.positions = np.zeros((self.max_agents_num, 2), dtype=np.float32)
        self.species = np.zeros(self.max_agents_num, dtype=np.int32)
        self.agent_ids = np.zeros(self.max_agents_num, dtype=np.int32)

        self.current_agent_count = 0


        self.last_random_velocity_time = time.time()
  
        
    def initialize(self):
        init_data = self._eco_to_box2d_init.get()
        self.positions = np.zeros((self.max_agents_num, 2), dtype=np.float32)
        self.species = np.zeros(self.max_agents_num, dtype=np.int32)
        self.agent_ids = np.arange(self.current_agent_count)
        self.current_agent_count = init_data['current_agent_count']
        
        self.positions[:self.current_agent_count] = init_data['positions']
        self.species[:self.current_agent_count] = init_data['species']
        
        for agent_id in self.agent_ids:
            self._create_body(agent_id)

    #------------------- create body  --------------------    

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

    def remove_body(self, agent_id):
        if agent_id in self.bodies:
            self.world.DestroyBody(self.bodies[agent_id])
            del self.bodies[agent_id]
    # ------------------Main routine ---------------------

    def update(self):
        self.process_ecosystem_queue()
        self.update_forces()
        self.step()
        self.update_property()
        self.send_data_to_tf()
        self.send_data_to_eco()
        # self.apply_random_velocity()
    
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
                    # 'action'キーがない場合は、全体の更新データとして処理
                    self._handle_full_update(update_data)
            except Empty:
                break

    def _handle_agent_added(self, data):
        agent_id = data['agent_id']
        species = data['species']
        position = data['position']
        self.current_agent_count += 1
        self.positions[agent_id] = position
        self.species[agent_id] = species
        self.agent_ids.append(agent_id)
        
        self._create_body(agent_id)
        logger.info(f"Agent {agent_id} added to Box2D. Total agents: {self.current_agent_count}")

    def _handle_agent_removed(self, data):
        agent_id = data['agent_id']
        if agent_id in self.bodies:
            self.remove_body(agent_id)
            self.agent_ids.remove(agent_id)
            self.current_agent_count -= 1
            logger.info(f"Agent {agent_id} removed from Box2D. Total agents: {self.current_agent_count}")
        else:
            logger.warning(f"Attempted to remove non-existent agent {agent_id} from Box2D")

    def update_forces(self):
        while not self._tf_to_box2d.empty():
            forces = self._tf_to_box2d.get()
            for agent_id, force in zip(self.bodies.keys(), forces):
                body = self.bodies[agent_id]
                body.ApplyForceToCenter((float(force[0]), float(force[1])), wake=True)
                
    
    def step(self):
        self.world.Step(self.dt, 6, 2)

    def update_property(self):
        for agent_id, body in self.bodies.items():
            self.positions[agent_id] = body.position.x, body.position.y
            
    def send_data_to_tf(self):
        data = {
            'positions': self.positions,
            'species': self.species,
            'current_agent_count': self.current_agent_count
        }
        self._box2d_to_tf.put(data)
    
    def send_data_to_eco(self):
        eco_data = {
            'positions': self.positions,
            'current_agent_count': self.current_agent_count
        }
        self._box2d_to_eco.put(eco_data)
            
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
            