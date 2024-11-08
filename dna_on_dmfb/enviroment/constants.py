from enum import IntEnum
import numpy as np
import seaborn as sns

class ACTIONS(IntEnum):
    STILL = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

GRID_SIZE = 128
N_ACTIONS = 5
EXTENT_OF_ZONE = (11, 11, 118, 118)
OBSERVATION_SIZE = 15
MAX_STEPS = 10000

class COLORS:
    BLANK = (255, 255, 255)
    GREY = OBSTACLE = (190, 190, 190)
    BLACK = BOUNDARY = (0, 0, 0)
    def __init__(self, num_agents):
        self.agent_colors = sns.color_palette("hls", num_agents)
        import random
        random.shuffle(self.agent_colors)

class REWARDS:
    STEP = -.1
    COLLISION = -10
    SUCCESS = 100
    FAILURE = -100
    REACH = 10
    INVALID = -10
    TURNBACK = -.2

    def distance_reward(original_distance, new_distance):
        if new_distance < original_distance: # closer to target
            return 0
        elif new_distance == original_distance: # same distance
            return -.1
        else: # further from target
            return -.3

MOVEMENTS_2D = np.array(
    [
        [0, 0], # STILL
        [-1, 0], # UP
        [1, 0], # DOWN
        [0, -1], # LEFT
        [0, 1], # RIGHT
    ]
)
MOVEMENTS_4D = np.hstack(
    (MOVEMENTS_2D, MOVEMENTS_2D)
)  # movement for upper left and lower right

VALID_NAMES = (
    "dATP",
    "dTTP",
    "dCTP",
    "dGTP",
    "ddATP",
    "ddTTP",
    "ddCTP",
    "ddGTP",
    "agarose",
    "buffer",
    "eluent",
    "primer",
)

dNTPs = (
    "dATP",
    "dTTP",
    "dCTP",
    "dGTP",
)

ddNTPs = (
    "ddATP",
    "ddTTP",
    "ddCTP",
    "ddGTP",
)

elements = (
    "agarose",
    "buffer",
    "eluent",
    "primer",
)

situation = {
    "dNTPs": {
      "dATP": [],
      "dTTP": [],
      "dCTP": [],
      "dGTP": [],
      "fixed": [
        [16, 1, 17, 2],
        [48, 1, 49, 2],
        [80, 1, 81, 2],
        [112, 1, 113, 2]
      ]
    },
    "ddNTPs": {
      "ddATP": [],
      "ddTTP": [],
      "ddCTP": [],
      "ddGTP": [],
      "fixed": [
        [16, 127, 17, 128],
        [48, 127, 49, 128],
        [80, 127, 81, 128],
        [112, 127, 113, 128]
      ]
    },
    "elements": {
      "agarose": [1, 64, 2, 65],
      "buffer": [127, 64, 128, 65],
      "eluent": [127, 96, 128, 97],
      "primer": [127, 32, 128, 33]
    }
  }