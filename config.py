# Simulation parameters
WORLD_WIDTH = 1500
WORLD_HEIGHT = 1500
NUM_AGENTS = 3000

# Flocking parameters
MAX_FORCE = 30.0
SEPARATION_DISTANCE = 90.0
ALIGNMENT_DISTANCE = 30.0
COHESION_DISTANCE = 100.0
SEPARATION_WEIGHT = 40
ALIGNMENT_WEIGHT = 20
COHESION_WEIGHT = 70


# Box2D physics parameters
# Agent parameters
LINEAR_DAMPING = 1 #線形減衰
AGENT_RADIUS = 5 
AGENT_MASS = 1
AGENT_FRICTION = 0
AGENT_DENSITY = 1 #密度
AGENT_RESTITUTION = 0 # 弾力

# Simulation speed
RENDER_FPS = 240
DT = 1 / 30

# Colors
BACKGROUND_COLOR = (0, 0, 0)  # White
AGENT_COLOR = (255, 255, 255)  # Black

# Initial velocity range
INITIAL_VELOCITY_MIN = -500
INITIAL_VELOCITY_MAX = 500

