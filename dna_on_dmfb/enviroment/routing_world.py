from collections import Counter
from copy import deepcopy
import gif
import gymnasium as gym
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (8, 8)
import math
import numpy as np
import networkx as nx
from pprint import pprint
import random
from typing import List, Tuple, Union
from dna_on_dmfb.enviroment.constants import (
    ACTIONS,
    COLORS,
    MAX_STEPS,
    MOVEMENTS_2D,
    MOVEMENTS_4D,
    N_ACTIONS,
    GRID_SIZE,
    OBSERVATION_SIZE,
    REWARDS,
    elements,
    dNTPs,
    ddNTPs,
    situation,
    VALID_NAMES,
)
from dna_on_dmfb.enviroment.units import Entrance, Zone, Intermediate
from dna_on_dmfb.enviroment.utils import (
    process_size,
    get_overlap_or_projection,
    try_with_different_params,
)
from dna_on_dmfb.enviroment.utils.graph import adjust_layout
from dna_on_dmfb.enviroment.utils.plot import plot
from dna_on_dmfb.enviroment.utils.tree import create_dna_mutation_tree


class RoutingDMFB(gym.Env):
    def __init__(
        self,
        size: Union[int, Tuple[int, int], List[int], np.ndarray] = None,
        obs_size: Union[int, Tuple[int, int], List[int], np.ndarray] = None,
        situation: dict = None,
        show_layout: bool = False,
        visualization: bool = False,
        cross: bool = True,
    ):
        super().__init__()
        self.actions = ACTIONS
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.size = process_size(size) if size is not None else process_size(GRID_SIZE)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(0, self.size[0], self.size[1]), dtype=np.uint8
        )
        self.obs_size = (
            process_size(obs_size)
            if obs_size is not None
            else process_size(OBSERVATION_SIZE)
        )
        # self.reset(show_layout=show_layout, visualization=visualization, cross=cross)

    def reset(
        self, show_layout: bool = False, visualization: bool = False, cross: bool = True
    ):
        def _center_to_region(center, w):
            if w % 2 == 0:
                return (
                    math.ceil(center[0] - w / 2),
                    math.ceil(center[1] - w / 2),
                    math.ceil(center[0] - w / 2) + w - 1,
                    math.ceil(center[1] - w / 2) + w - 1,
                )
            return (
                center[0] - w // 2,
                center[1] - w // 2,
                center[0] + w // 2,
                center[1] + w // 2,
            )

        def _get_zone_width(level, height):
            return 3 + 2 * (height - level)

        self.situation = deepcopy(situation)
        self.situation["products"] = create_dna_mutation_tree()
        pprint(self.situation["products"])
        self.show_layout = show_layout
        self.cross = cross
        self.visualization = visualization
        self.frames = [] if visualization else None
        # layout for entrances and zones
        params_list = [
            ((self.situation, (11, 11, 118, 118), i), {}) for i in range(23, 4, -2)
        ]
        try:
            self.graph, self.init_positions, self.zone_parent, height = (
                try_with_different_params(adjust_layout, params_list, 10)
            )
        except:
            raise ValueError("Failed to initialize grid")
        # zones as nodes
        self.zone_children = {}
        for child, parent in self.zone_parent.items():
            if parent not in self.zone_children:
                self.zone_children[parent] = []
            self.zone_children[parent].append(child)
        # initialize
        self.entrances = []
        self.entrances_idx = {}
        self.zones = []
        self.zones_idx = {}
        self.droplets = []
        self.droplets_idx = {}
        self.classes = []
        self.classes_idx = {}
        self.m_activation = np.zeros((self.height + 2, self.width + 2), dtype=int)
        self.m_class_obstacles = np.zeros(
            (len(self.classes), self.height + 2, self.width + 2), dtype=bool
        )
        self.m_class_trajectory = np.zeros(
            (len(self.classes), self.height + 2, self.width + 2), dtype=int
        )
        self.m_droplet_obstacles = np.zeros(
            (len(self.droplets), self.height + 2, self.width + 2), dtype=bool
        )
        self.m_droplet_trajectory = np.zeros(
            (len(self.droplets), self.height + 2, self.width + 2), dtype=int
        )
        # pre-defined classes
        for class_name in VALID_NAMES:
            self._add_class(class_name)
        # entrances
        for node, pos in self.init_positions.items():
            if node in elements or node in dNTPs or node in ddNTPs:
                self.entrances_idx[node] = len(self.entrances)
                self.entrances.append(Entrance(node, 2, 2, _center_to_region(pos, 2)))
            else:
                self.init_positions[node] = (
                    int(pos[0]),
                    int(pos[1]),
                )  # round to integer
        if show_layout:
            nx.draw(
                self.graph, pos=self.init_positions, with_labels=False, node_size=100
            )
            plt.show()
            plt.close()
        # BFS to initialize zones
        nodes = [self.situation["products"]]
        while len(nodes) > 0:
            node = nodes.pop(0)
            self.zones_idx[node["value"]] = len(self.zones)
            parent = self.zone_parent.get(node["value"], None)
            if parent is None:
                self.root_zone = Zone(
                    node["value"],
                    5,
                    5,
                    _center_to_region(
                        self.init_positions[node["value"]], _get_zone_width(1, height)
                    ),
                    node["value"],
                    1,
                )
                self.zones.append(self.root_zone)
            else:
                l = self.zones[self.zones_idx[parent]].level + 1
                self.zones.append(
                    Zone(
                        node["value"],
                        5,
                        5,
                        _center_to_region(
                            self.init_positions[node["value"]],
                            _get_zone_width(l, height),
                        ),
                        node["value"],
                        l,
                    )
                )
            for child in node["children"]:
                nodes.append(child)
        for parent, children in self.zone_children.items():
            for child in children:
                # add parent and child relationship
                self.zones[self.zones_idx[child]].add_parent(
                    self.zones[self.zones_idx[parent]]
                )
                self.zones[self.zones_idx[parent]].add_child(
                    self.zones[self.zones_idx[child]]
                )
        # droplets
        self.flux = {}
        self.distances = {}
        # define all possible droplets and their flux to zones
        for zone in self.zones:
            for entrance in self.entrances:
                if entrance.name.startswith("d"):
                    droplet = entrance.sent(deepcopy(zone.location))
                    droplet.name = entrance.name + "_" + zone.name
                    # self.distances[droplet.name] = distance(droplet.center, zone.center)
                    self.distances[droplet.name] = droplet.distance_to_goal
                    if entrance.name.startswith("dd"):
                        self.flux[droplet.name] = 1
                    else:
                        counter = Counter(zone.name[:-1])
                        self.flux[droplet.name] = counter[droplet.content]
                    self._add_droplet(droplet)
                elif entrance.name.__eq__("agarose"):
                    droplet = entrance.sent(deepcopy(zone.location))
                    droplet.name = entrance.name + "_" + zone.name
                    # self.distances[droplet.name] = distance(droplet.center, zone.center)
                    self.distances[droplet.name] = droplet.distance_to_goal
                    self.flux[droplet.name] = 1
                    self._add_droplet(droplet)
                elif entrance.name.__eq__("primer") and zone.level == 1:
                    droplet = entrance.sent(deepcopy(zone.location))
                    droplet.name = entrance.name + "_" + zone.name
                    # self.distances[droplet.name] = distance(droplet.center, zone.center)
                    self.distances[droplet.name] = droplet.distance_to_goal
                    self.flux[droplet.name] = 1
                    self._add_droplet(droplet)
                elif entrance.name.__eq__("eluent") and zone.level == 1:
                    droplet = entrance.sent(deepcopy(zone.location))
                    droplet.name = entrance.name + "_" + zone.name
                    self.flux[droplet.name] = len(zone.mutate_prefixs)
                    # self.distances[droplet.name] = distance(droplet.center, zone.center)
                    self.distances[droplet.name] = droplet.distance_to_goal
                    self._add_droplet(droplet)
            # for zone in self.zones:
            for intermediate, next_products in zone.mutate_prefixs.items():
                for next_product in next_products:
                    next_zone = self.zones[self.zones_idx[next_product]]
                    self._add_class(intermediate)
                    droplet = Intermediate(
                        intermediate + "_" + next_zone.name,
                        2,
                        2,
                        self._get_out_of_zone(zone, next_zone),
                        deepcopy(next_zone.location),
                    )
                    self.flux[droplet.name] = 1
                    # self.distances[droplet.name] = distance(droplet.center, next_zone.center)
                    self.distances[droplet.name] = droplet.distance_to_goal
                    self._add_droplet(droplet)
        self.droplets_to_move = set(self.droplets_idx.keys())
        self.colors = COLORS(len(self.classes))
        for droplet_name, droplet_idx in self.droplets_idx.items():
            self.droplets[droplet_idx].set_color(
                self.colors.agent_colors[
                    self.classes_idx[self.droplets[droplet_idx].class_name]
                ]
            )
        # 添加frontier及其color
        # TODO:
        self.goals = [zone.location.tolist() for zone in self.zones]
        self.frontiers = []
        for zone in self.zones:
            self.frontiers.append([])
        self.droplet_fixed_obstacles = np.zeros(
            (len(self.droplets), self.height + 2, self.width + 2), dtype=bool
        )
        for droplet in self.droplets:
            droplet_idx = self.droplets_idx[droplet.name]
            goal = droplet.goal.tolist()
            idx = self.goals.index(goal)
            if len(self.frontiers[idx]) < 5:
                self.frontiers[idx].append(droplet.color_rgba)
            m = np.zeros((self.height + 2, self.width + 2), dtype=bool)
            m[:, [0, -1]] = True
            m[[0, -1], :] = True
            for i, zone_location in enumerate(self.goals):
                if i == idx:
                    continue
                m[
                    zone_location[0] : zone_location[2] + 1,
                    zone_location[1] : zone_location[3] + 1,
                ] = True
            self.droplet_fixed_obstacles[droplet_idx] = m
        self.max_activation = 2 * (
            len(self.zones[0].name) * 4  # dNTP or ddNTP to zone
            + len(self.zones)
            - 1  # intermediate to zone
            + len(self.zones)  # agarose to zone
            + 2  # primer and eluent to zone
        )
        self.steps = 0
        self.droplets_history = {
            droplet_name: [] for droplet_name, droplet_idx in self.droplets_idx.items()
        }
        self.fluxes = []

    def step(self, droplet_name: str, action: int):
        terminated = False
        reward = REWARDS.STEP  # default reward as step reward
        info = {}
        original_distance = self.distances[droplet_name]
        # move the droplet and check if it is truncated
        truncated, new_distance, reach_goal, r = self._move_droplet(
            droplet_name, action
        )
        reward += r
        self.steps += 1
        # get the next observation
        next_observation = self._get_observation(droplet_name)
        reward += REWARDS.distance_reward(original_distance, new_distance)
        # droplet = self.droplets[self.droplets_idx[droplet_name]]
        if truncated:
            reward += REWARDS.INVALID
            info["reason"] = "invalid"
            return next_observation, reward, terminated, truncated, info
        if reach_goal:
            self.droplets_to_move.remove(droplet_name)
            reward += REWARDS.REACH
        done = self.finished
        if done:
            terminated = True
            info["reason"] = "success"
            # TODO: reward when all droplets reach their goals
            reward += REWARDS.SUCCESS
        if self.steps >= MAX_STEPS:
            terminated = True
            info["reason"] = "steps"
        return next_observation, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        plot(self.rgb_array, self.goals, self.frontiers)

    def show(self):
        self.render()
        plt.show()
        plt.close()

    @gif.frame
    def plot(self):
        self.render()

    def select(self):
        # TODO
        prob = np.random.uniform()
        if prob < 0.1:
            name = random.choice(list(self.droplets_to_move))
        elif prob < 0.5:
            # 选择距离最近的droplet
            name = min(self.droplets_to_move, key=lambda x: self.distances[x])
        else:
            # 选择flux最大的droplet
            name = max(self.droplets_to_move, key=lambda x: self.flux[x])
        obs = self._get_observation(name)
        return name, obs

    def _get_observation(self, droplet_name: str):
        obs = np.zeros(
            (0, self.obs_size[0], self.obs_size[1]), dtype=np.uint8
        )  # result
        droplet_idx = self.droplets_idx[droplet_name]  # droplet index
        droplet = self.droplets[droplet_idx]  # droplet
        class_name = droplet.class_name  # class
        class_idx = self.classes_idx[class_name]  # class index
        center = [int(i) for i in droplet.center]  # center
        x_upper_left = (
            center[0] - self.obs_size[0] // 2
        )  # upper left corner of observation in the grid
        y_upper_left = center[1] - self.obs_size[1] // 2
        x_lower_right = center[0] + self.obs_size[0] // 2
        y_lower_right = center[1] + self.obs_size[1] // 2
        # current location
        obs_location = np.zeros((self.obs_size[0], self.obs_size[1]), dtype=bool)
        obs_location[
            droplet.x_upper_left
            - x_upper_left : droplet.x_lower_right
            - x_upper_left
            + 1,
            droplet.y_upper_left
            - y_upper_left : droplet.y_lower_right
            - y_upper_left
            + 1,
        ] = True
        obs = np.concatenate((obs, np.expand_dims(obs_location, axis=0)), axis=0)
        # current goal
        goal = droplet.goal.tolist()
        obs_current_goal = np.zeros((self.obs_size[0], self.obs_size[1]), dtype=bool)
        to_goal = get_overlap_or_projection(
            [x_upper_left, y_upper_left, x_lower_right, y_lower_right], goal
        )
        if len(to_goal) == 2:
            obs_current_goal[to_goal[0] - x_upper_left, to_goal[1] - y_upper_left] = (
                True
            )
        elif len(to_goal) == 4:
            obs_current_goal[
                to_goal[0] - x_upper_left : to_goal[2] - x_upper_left + 1,
                to_goal[1] - y_upper_left : to_goal[3] - y_upper_left + 1,
            ] = True
        obs = np.concatenate((obs, np.expand_dims(obs_current_goal, axis=0)), axis=0)
        # same-class droplets and their goals
        obs_same_class_droplets = np.zeros(
            (self.obs_size[0], self.obs_size[1]), dtype=bool
        )
        obs_same_class_goals = np.zeros(
            (self.obs_size[0], self.obs_size[1]), dtype=bool
        )
        obs_same_class_trajectory = np.zeros(
            (self.obs_size[0], self.obs_size[1]), dtype=bool
        )
        # other droplets and their goals
        obs_other_droplets = np.zeros((self.obs_size[0], self.obs_size[1]), dtype=bool)
        obs_other_goals = np.zeros((self.obs_size[0], self.obs_size[1]), dtype=bool)
        obs_other_class_trajectory = np.zeros(
            (self.obs_size[0], self.obs_size[1]), dtype=bool
        )
        for n, idx in self.droplets_idx.items():
            if n.__eq__(droplet_name):  # skip itself
                continue
            d = self.droplets[idx]  # droplet
            c = d.class_name  # class
            g = d.goal.tolist()  # goal
            where_d = get_overlap_or_projection(
                [x_upper_left, y_upper_left, x_lower_right, y_lower_right],
                d.location.tolist(),
            )
            if len(where_d) == 2:  # out of the observation
                continue
            # the droplet is in the observation
            where_g = get_overlap_or_projection(
                [x_upper_left, y_upper_left, x_lower_right, y_lower_right], g
            )
            if len(where_g) == 2:  # out of the observation
                if c.__eq__(class_name):
                    obs_same_class_goals[
                        where_g[0] - x_upper_left, where_g[1] - y_upper_left
                    ] = True
                    obs_same_class_droplets[
                        where_d[0] - x_upper_left : where_d[2] - x_upper_left + 1,
                        where_d[1] - y_upper_left : where_d[3] - y_upper_left + 1,
                    ] = True
                else:
                    obs_other_goals[
                        where_g[0] - x_upper_left, where_g[1] - y_upper_left
                    ] = True
                    obs_other_droplets[
                        where_d[0] - x_upper_left : where_d[2] - x_upper_left + 1,
                        where_d[1] - y_upper_left : where_d[3] - y_upper_left + 1,
                    ] = True
            elif len(where_g) == 4:
                if c.__eq__(class_name):
                    obs_same_class_droplets[
                        where_d[0] - x_upper_left : where_d[2] - x_upper_left + 1,
                        where_d[1] - y_upper_left : where_d[3] - y_upper_left + 1,
                    ] = True
                    obs_same_class_goals[
                        where_g[0] - x_upper_left : where_g[2] - x_upper_left + 1,
                        where_g[1] - y_upper_left : where_g[3] - y_upper_left + 1,
                    ]
                else:
                    obs_other_droplets[
                        where_d[0] - x_upper_left : where_d[2] - x_upper_left + 1,
                        where_d[1] - y_upper_left : where_d[3] - y_upper_left + 1,
                    ] = True
                    obs_other_goals[
                        where_g[0] - x_upper_left : where_g[2] - x_upper_left + 1,
                        where_g[1] - y_upper_left : where_g[3] - y_upper_left + 1,
                    ] = True
        obs = np.concatenate(
            (obs, np.expand_dims(obs_same_class_droplets, axis=0)), axis=0
        )
        obs = np.concatenate(
            (obs, np.expand_dims(obs_same_class_goals, axis=0)), axis=0
        )
        obs_same_class_trajectory[
            max(0, x_upper_left)
            - x_upper_left : min(self.height, x_lower_right)
            - x_upper_left
            + 1,
            max(0, y_upper_left)
            - y_upper_left : min(self.width, y_lower_right)
            - y_upper_left
            + 1,
        ] = (
            self.m_class_trajectory[
                class_idx,
                max(0, x_upper_left) : min(self.height, x_lower_right) + 1,
                max(0, y_upper_left) : min(self.width, y_lower_right) + 1,
            ]
            > 0
        )
        obs = np.concatenate(
            (obs, np.expand_dims(obs_same_class_trajectory, axis=0)), axis=0
        )
        obs = np.concatenate((obs, np.expand_dims(obs_other_droplets, axis=0)), axis=0)
        obs = np.concatenate((obs, np.expand_dims(obs_other_goals, axis=0)), axis=0)
        obs_other_class_trajectory[
            max(0, x_upper_left)
            - x_upper_left : min(self.height, x_lower_right)
            - x_upper_left
            + 1,
            max(0, y_upper_left)
            - y_upper_left : min(self.width, y_lower_right)
            - y_upper_left
            + 1,
        ] = np.any(
            self.m_class_trajectory[
                [i for i in range(len(self.classes)) if i != class_idx],
                max(0, x_upper_left) : min(self.height, x_lower_right) + 1,
                max(0, y_upper_left) : min(self.width, y_lower_right) + 1,
            ]
            > 0,
            axis=0,
        )
        obs = np.concatenate(
            (obs, np.expand_dims(obs_other_class_trajectory, axis=0)), axis=0
        )

        # obstacles
        obs_fixed_obstacles = np.zeros((self.obs_size[0], self.obs_size[1]), dtype=bool)
        obs_fixed_obstacles[
            max(0, x_upper_left)
            - x_upper_left : min(self.height, x_lower_right)
            - x_upper_left
            + 1,
            max(0, y_upper_left)
            - y_upper_left : min(self.width, y_lower_right)
            - y_upper_left
            + 1,
        ] = self.droplet_fixed_obstacles[
            droplet_idx,
            max(0, x_upper_left) : min(self.height, x_lower_right) + 1,
            max(0, y_upper_left) : min(self.width, y_lower_right) + 1,
        ]
        if x_upper_left < 0:  # out of the boundary
            obs_fixed_obstacles[: abs(x_upper_left), :] = True
        if x_lower_right > self.height:  # out of the boundary
            obs_fixed_obstacles[self.height - abs(x_lower_right) :, :] = True
        if y_upper_left < 0:  # out of the boundary
            obs_fixed_obstacles[:, : abs(y_upper_left)] = True
        if y_lower_right > self.width:  # out of the boundary
            obs_fixed_obstacles[:, self.width - abs(y_lower_right) :] = True
        obs = np.concatenate((obs, np.expand_dims(obs_fixed_obstacles, axis=0)), axis=0)
        # activation
        obs_activation = np.zeros((self.obs_size[0], self.obs_size[1]), dtype=float)
        obs_activation[
            max(0, x_upper_left)
            - x_upper_left : min(self.height, x_lower_right)
            - x_upper_left
            + 1,
            max(0, y_upper_left)
            - y_upper_left : min(self.width, y_lower_right)
            - y_upper_left
            + 1,
        ] = (
            self.m_activation[
                max(0, x_upper_left) : min(self.height, x_lower_right) + 1,
                max(0, y_upper_left) : min(self.width, y_lower_right) + 1,
            ]
            / self.max_activation
        )
        obs = np.concatenate((obs, np.expand_dims(obs_activation, axis=0)), axis=0)
        return obs

    def _move_droplet(self, droplet_name: str, action: int):
        r = 0
        droplet_idx = self.droplets_idx[droplet_name]
        movement = MOVEMENTS_4D[action]
        turn_back = self.droplets[droplet_idx].move(movement)
        r += REWARDS.TURNBACK if turn_back else 0
        droplet = self.droplets[droplet_idx]
        new_distance = droplet.distance_to_goal
        if action == 0:
            return False, new_distance, droplet.reach_goal, r
        obstacle = self.droplet_fixed_obstacles[droplet_idx]
        if np.any(
            obstacle[
                droplet.x_upper_left : droplet.x_lower_right + 1,
                droplet.y_upper_left : droplet.y_lower_right + 1,
            ]
        ):
            return True, new_distance, False, r
        self.distances[droplet_name] = droplet.distance_to_goal
        if action == 1:  # UP
            next_area = (
                droplet.x_upper_left,
                droplet.y_upper_left,
                droplet.x_upper_left,
                droplet.y_lower_right,
            )
        elif action == 2:  # DOWN
            next_area = (
                droplet.x_lower_right,
                droplet.y_upper_left,
                droplet.x_lower_right,
                droplet.y_lower_right,
            )
        elif action == 3:  # LEFT
            next_area = (
                droplet.x_upper_left,
                droplet.y_upper_left,
                droplet.x_lower_right,
                droplet.y_upper_left,
            )
        elif action == 4:  # RIGHT
            next_area = (
                droplet.x_upper_left,
                droplet.y_lower_right,
                droplet.x_lower_right,
                droplet.y_lower_right,
            )
        # check if get to the trajectory of other droplets
        class_idx = self.classes_idx[droplet.class_name]
        other_classes = [i for i in range(len(self.classes)) if i != class_idx]

        if np.any(
            self.m_class_trajectory[
                other_classes,
                droplet.x_upper_left : droplet.x_lower_right + 1,
                droplet.y_upper_left : droplet.y_lower_right + 1,
            ]
        ):
            f = sum(
                [
                    self.flux[droplet_name]
                    for droplet_name, idx in self.droplets_idx.items()
                    if not droplet_name.split("_")[0].__eq__(self.classes[class_idx])
                    and np.any(
                        self.m_droplet_trajectory[
                            idx,
                            next_area[0] : next_area[2] + 1,
                            next_area[1] : next_area[3] + 1,
                        ]
                    )
                ]
            )  # flux of other droplets in the next area with different class
            r -= np.tanh(f / self.max_activation + 0.2)

        self.m_activation[
            droplet.x_upper_left : droplet.x_lower_right + 1,
            droplet.y_upper_left : droplet.y_lower_right + 1,
        ] += self.flux[droplet_name]
        self.m_droplet_trajectory[
            droplet_idx,
            droplet.last_area[0] : droplet.last_area[2] + 1,
            droplet.last_area[1] : droplet.last_area[3] + 1,
        ] += 1
        self.m_class_trajectory[
            class_idx,
            droplet.last_area[0] : droplet.last_area[2] + 1,
            droplet.last_area[1] : droplet.last_area[3] + 1,
        ] += 1
        if action > 0:
            self.droplets_history[droplet_name].append((droplet.last_area, action))
        return False, new_distance, droplet.reach_goal, r

    def _add_droplet(self, droplet):
        assert droplet.class_name in self.classes_idx
        self.droplets_idx[droplet.name] = len(self.droplets)
        self.droplets.append(droplet)
        m_trajectory = np.zeros((self.height + 2, self.width + 2), dtype=int)
        self.m_droplet_trajectory = np.concatenate(
            (self.m_droplet_trajectory, np.expand_dims(m_trajectory, axis=0)), axis=0
        )
        m_obstacles = np.zeros((self.height + 2, self.width + 2), dtype=bool)
        m_obstacles[:, [0, -1]] = True
        m_obstacles[[0, -1], :] = True
        for zone in self.zones:
            if not zone.is_goal_of(droplet.goal):
                m_obstacles[
                    zone.x_upper_left : zone.x_lower_right + 1,
                    zone.y_upper_left : zone.y_lower_right + 1,
                ] = True
        if not self.cross:
            m_obstacles[np.any(self.m_droplet_trajectory, axis=0)] = True
        self.m_activation[
            droplet.x_upper_left : droplet.x_lower_right + 1,
            droplet.y_upper_left : droplet.y_lower_right + 1,
        ] += self.flux[droplet.name]
        # self.droplets[-1].set_color(COLORS.agent_colors[self.classes_idx[droplet.class_name]])
        self.m_droplet_obstacles = np.concatenate(
            (self.m_droplet_obstacles, np.expand_dims(m_obstacles, axis=0)), axis=0
        )
        if not self.cross:
            self.m_droplet_obstacles[
                :-1,
                droplet.x_upper_left : droplet.x_lower_right + 1,
                droplet.y_upper_left : droplet.y_lower_right + 1,
            ] = True
            self.m_class_obstacles[
                [
                    class_name
                    for class_name, class_idx in self.classes_idx.items()
                    if class_name.__eq__(droplet.class_name)
                ],
                droplet.x_upper_left : droplet.x_lower_right + 1,
                droplet.y_upper_left : droplet.y_lower_right + 1,
            ] = True

    def valid_actions(self, droplet_name: str):
        droplet_idx = self.droplets_idx[droplet_name]
        droplet = self.droplets[droplet_idx]
        obstacles = self.droplet_fixed_obstacles[droplet_idx]
        res = [0] + [
            i
            for i in range(1, N_ACTIONS)
            if not np.any(
                obstacles[
                    droplet.x_upper_left
                    + MOVEMENTS_2D[i][0] : droplet.x_lower_right
                    + MOVEMENTS_2D[i][0]
                    + 1,
                    droplet.y_upper_left
                    + MOVEMENTS_2D[i][1] : droplet.y_lower_right
                    + MOVEMENTS_2D[i][1]
                    + 1,
                ]
            )
        ]
        return res

    def _add_class(self, class_name):
        self.classes_idx[class_name] = len(self.classes)
        self.classes.append(class_name)
        m_obstacles = np.zeros((self.height + 2, self.width + 2), dtype=bool)
        m_obstacles[:, [0, -1]] = True
        m_obstacles[[0, -1], :] = True
        if not self.cross:
            m_obstacles[np.any(self.m_class_trajectory, axis=0)] = True
        self.m_class_obstacles = np.concatenate(
            (self.m_class_obstacles, np.expand_dims(m_obstacles, axis=0)), axis=0
        )
        m_trajectory = np.zeros((self.height + 2, self.width + 2), dtype=int)
        self.m_class_trajectory = np.concatenate(
            (self.m_class_trajectory, np.expand_dims(m_trajectory, axis=0)), axis=0
        )

    def _get_out_of_zone(self, zone: Zone, new_zone: Zone):
        # 左下角
        # return (zone.location[2]+1, zone.location[1], zone.location[2]+2, zone.location[1]+1)
        c = zone.center
        new_c = new_zone.center
        v = (new_c[0] - c[0], new_c[1] - c[1])
        # TODO:选择边界上一个没有被占用且离new_c较近的位置作为新的位置
        min_distance = float("inf")
        new_x, new_y = zone.location[2] + 1, zone.location[3] - 1
        if v[0] < 0:  # left
            for x_upper_left in range(zone.location[0], zone.location[2] - 1):
                if not np.any(
                    self.m_droplet_obstacles[
                        :,
                        x_upper_left : x_upper_left + 2,
                        zone.location[1] - 2 : zone.location[1],
                    ]
                ):
                    d = distance((x_upper_left, zone.location[1] - 2), new_c)
                    if d < min_distance:
                        min_distance = d
                        new_x = x_upper_left
                        new_y = zone.location[1] - 2
        elif v[0] > 0:  # right
            for x_upper_left in range(zone.location[0], zone.location[2] - 1):
                if not np.any(
                    self.m_droplet_obstacles[
                        :,
                        x_upper_left : x_upper_left + 2,
                        zone.location[3] + 1 : zone.location[3] + 3,
                    ]
                ):
                    d = distance((x_upper_left, zone.location[3] + 1), new_c)
                    if d < min_distance:
                        min_distance = d
                        new_x = x_upper_left
                        new_y = zone.location[3] + 1
        elif v[1] < 0:  # up
            for y_upper_left in range(zone.location[1], zone.location[3] - 1):
                if not np.any(
                    self.m_droplet_obstacles[
                        :,
                        zone.location[0] - 2 : zone.location[0],
                        y_upper_left : y_upper_left + 2,
                    ]
                ):
                    d = distance((zone.location[0] - 2, y_upper_left), new_c)
                    if d < min_distance:
                        min_distance = d
                        new_x = zone.location[0] - 2
                        new_y = y_upper_left
        elif v[1] > 0:  # down
            for y_upper_left in range(zone.location[1], zone.location[3] - 1):
                if not np.any(
                    self.m_droplet_obstacles[
                        :,
                        zone.location[2] + 1 : zone.location[2] + 3,
                        y_upper_left : y_upper_left + 2,
                    ]
                ):
                    d = distance((zone.location[2] + 1, y_upper_left), new_c)
                    if d < min_distance:
                        min_distance = d
                        new_x = zone.location[2] + 1
                        new_y = y_upper_left
        return (new_x, new_y, new_x + 1, new_y + 1)

    def _get_zone_name_from_droplet_name(self, droplet_name):
        return droplet_name.split("_")[1]

    @property
    def height(self):
        return self.size[0]

    @property
    def width(self):
        return self.size[1]

    @property
    def finished(self):
        return len(self.droplets_to_move) == 0

    @property
    def rgb_array(self):
        img = np.zeros((self.height + 2, self.width + 2, 3), dtype=np.uint8)
        img.fill(255)
        img[0, :, :] = img[-1, :, :] = img[:, 0, :] = img[:, -1, :] = COLORS.BOUNDARY

        for droplet_name, idx in self.droplets_idx.items():
            img[self.m_droplet_trajectory[idx, :, :] > 0] = COLORS.GREY

        for name, idx in self.droplets_idx.items():
            droplet = self.droplets[idx]
            # img[self.m_droplet_obstacles[idx]] = COLORS.GREY
            img[
                droplet.location[0] : droplet.location[2] + 1,
                droplet.location[1] : droplet.location[3] + 1,
                :,
            ] = droplet.color_integer

        for e_name, idx in self.entrances_idx.items():
            entrance = self.entrances[idx]
            location = entrance.location
            color = self.colors.agent_colors[self.classes_idx[e_name]]
            color = [int(c * 255) for c in color]
            if location[0] == 1:
                img[0, location[1] : location[3] + 1, :] = color
            elif location[0] == 127:
                img[-1, location[1] : location[3] + 1, :] = color
            elif location[1] == 1:
                img[location[0] : location[2] + 1, 0, :] = color
            elif location[1] == 127:
                img[location[0] : location[2] + 1, -1, :] = color

        return img


def distance(p1, p2):
    # 曼哈顿距离
    return sum(abs(np.array(p1) - np.array(p2)))
