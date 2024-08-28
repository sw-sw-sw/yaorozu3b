from Box2D import b2World, b2Vec2, b2BodyDef, b2_staticBody, b2_dynamicBody, b2CircleShape, b2FixtureDef
from config import *
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Box2DSimulation:
    def __init__(self, queues):
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self.bodies = []
        self._rendering_queue = queues['rendering_queue']
        self._eco_to_box2d_creatures = queues['eco_to_box2d_creatures']
        self.positions = np.zeros((NUM_AGENTS, 2), dtype=np.float32)
        self.world_radius = min(WORLD_WIDTH, WORLD_HEIGHT) / 2
        self.world_center = b2Vec2(WORLD_WIDTH / 2, WORLD_HEIGHT / 2)
        self.center_force_strength = CENTER_FORCE_STRENGTH
        self._create_boundary()

    def _create_boundary(self):
        boundary_def = b2BodyDef(
            position=self.world_center,
            type=b2_staticBody
        )
        boundary_body = self.world.CreateBody(boundary_def)
        boundary_shape = b2CircleShape(radius=self.world_radius)
        boundary_fixture = boundary_body.CreateFixture(
            shape=boundary_shape,
            density=0,
            friction=0.3,
            restitution=0.5
        )

    def create_bodies(self):
        for _ in range(NUM_AGENTS):
            creature_info = self._eco_to_box2d_creatures.get()
            self._create_body(creature_info)
        logger.info(f"Created {NUM_AGENTS} bodies in Box2D simulation")

    def _create_body(self, creature_info):
        body_def = b2BodyDef(
            type=b2_dynamicBody,
            position=b2Vec2(creature_info['x'], creature_info['y']),
            linearVelocity=b2Vec2(0, 0),
            linearDamping=LINEAR_DAMPING
        )
        body = self.world.CreateBody(body_def)
        circle_shape = b2CircleShape(radius=creature_info['radius'])
        fixture_def = b2FixtureDef(
            shape=circle_shape,
            density=AGENT_DENSITY,
            friction=AGENT_FRICTION,
            restitution=AGENT_RESTITUTION
        )
        body.CreateFixture(fixture_def)
        body.mass = AGENT_MASS
        self.bodies.append(body)

    def apply_forces_to_box2d(self, _forces):
        forces = np.frombuffer(_forces.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))
        for body, force in zip(self.bodies, forces):
            body.ApplyForceToCenter((float(force[0]), float(force[1])), wake=True)

    def apply_center_force(self):
        for body in self.bodies:
            to_center = self.world_center - body.position
            distance = to_center.length
            if distance > 0:
                force = to_center.normalize() * self.center_force_strength * body.mass
                body.ApplyForceToCenter(force, wake=True)

    def apply_positions_to_shared_memory(self, _positions):
        positions = self.get_positions()
        np.frombuffer(_positions.get_obj(), dtype=np.float32).reshape((NUM_AGENTS, 2))[:] = positions

    def step(self):
        self.apply_center_force()
        self.world.Step(DT, 6, 2)

    def get_positions(self):
        return np.array([(body.position.x, body.position.y) for body in self.bodies], dtype=np.float32)

    def add_positions_to_render_queue(self):
        if self._rendering_queue.empty():
            self._rendering_queue.put(self.get_positions())