# box2d_simulation.py
from Box2D import b2World, b2Vec2, b2BodyDef, b2_dynamicBody, b2CircleShape
from config import *
import numpy as np

class Box2DSimulation:
    def __init__(self, world_width, world_height, queues):
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.bodies = []
        self._rendering_queue = queues['rendering_queue']
        self.positions = np.zeros(NUM_AGENTS, dtype=np.int32)
        

    def create_bodies(self, initial_positions, initial_velocities):
        for position, velocity in zip(initial_positions, initial_velocities):
            body_def = b2BodyDef(
                type=b2_dynamicBody,
                position=b2Vec2(float(position[0]), float(position[1])),  # ここを修正
                linearVelocity=b2Vec2(float(velocity[0]), float(velocity[1])),  # 速度を設定
                linearDamping=LINEAR_DAMPING
            )
            body = self.world.CreateBody(body_def)
            circle_shape = b2CircleShape(radius=AGENT_RADIUS)
            body.CreateFixture(shape=circle_shape, 
                               density=AGENT_DENSITY, 
                               friction=AGENT_FRICTION,
                               restitution=AGENT_RESTITUTION
                               )
            rnd = np.random.rand() * 1.4 + 1.5
            body.mass = AGENT_MASS * rnd
            self.bodies.append(body)

    def apply_forces(self, forces):
        for body, force in zip(self.bodies, forces):
            body.ApplyForceToCenter((float(force[0]), float(force[1])), wake=True) 
            
    def apply_forces_to_box2d(self, _forces ):
        forces = np.frombuffer(_forces.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))

        for body, force in zip(self.bodies, forces):
            body.ApplyForceToCenter((float(force[0]), float(force[1])), wake=True) 

    def apply_positions_to_shared_memory(self, _positions):
        np.frombuffer(_positions.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))[:] = self.get_positions()

        
        
        
    def step(self):
        self.world.Step(DT, 6, 2)

    def get_positions(self):
        self.positions = np.array([(body.position.x, body.position.y) for body in self.bodies], dtype=np.float32)
        return self.positions

    def add_positions_to_render_queue(self):
        if self._rendering_queue.empty():
            self._rendering_queue.put(self.positions)