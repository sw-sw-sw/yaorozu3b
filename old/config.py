# ------------ INITIAL SETTING ---------------------------
# Simulation parameters
WORLD_WIDTH = 2000
WORLD_HEIGHT = 2000
MAX_AGENTS_NUM = 800

# Simulation speed
RENDER_FPS = 240
DT = 1 / 30

# Colors
BACKGROUND_COLOR = (0, 0, 0)  
AGENT_COLOR = (255, 255, 255)  

# ----------------- Environment parameters ----------------------
CENTER_ATTRACTION_WEIGHT = 60
CONFINEMENT_WEIGHT = 80.0
ROTATION_STRENGTH = 300

RANDOM_VELOCITY_INTERVAL = 0.1
RANDOM_SPEED = 200

# ------------------ Common species parameters ---------------------
# Flocking parameters
SEPARATION_DISTANCE = 20.0
SEPARATION_WEIGHT = 5
COHESION_DISTANCE = 100.0
COHESION_WEIGHT = 10

# Initial velocity range
INITIAL_VELOCITY_MIN = -200
INITIAL_VELOCITY_MAX = 200
MAX_FORCE = 250

