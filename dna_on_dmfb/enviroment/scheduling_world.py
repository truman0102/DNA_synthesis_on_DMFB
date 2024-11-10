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

    def reset(
        self, products:dict=None, show_layout: bool = False, 
        visualization: bool = False, zone_locations: dict = None
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
        self.situation["products"] = create_dna_mutation_tree(
            sequence_length=random.randint(10, 20),
            max_tree_height=random.randint(2, 4),
            max_total_nodes=random.randint(5, 10),
            max_children_per_node=random.randint(2, 3),
        ) if products is None else products
        if zone_locations is not None:
            self.situation["zone_locations"] = zone_locations
        pprint(self.situation["products"])
        self.show_layout = show_layout
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
            (len(self.classes), self.height + 2, self.width + 2), dtype=bool
        )
        self.m_droplet_obstacles = np.zeros(
            (len(self.droplets), self.height + 2, self.width + 2), dtype=bool
        )
        self.m_droplet_trajectory = np.zeros(
            (len(self.droplets), self.height + 2, self.width + 2), dtype=bool
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
                    self.distances[droplet.name] = droplet.distance_to_goal
                    self.flux[droplet.name] = 1
                    self._add_droplet(droplet)
                elif entrance.name.__eq__("primer") and zone.level == 1:
                    droplet = entrance.sent(deepcopy(zone.location))
                    droplet.name = entrance.name + "_" + zone.name
                    self.distances[droplet.name] = droplet.distance_to_goal
                    self.flux[droplet.name] = 1
                    self._add_droplet(droplet)
                elif entrance.name.__eq__("eluent") and zone.level == 1:
                    droplet = entrance.sent(deepcopy(zone.location))
                    droplet.name = entrance.name + "_" + zone.name
                    self.flux[droplet.name] = len(zone.mutate_prefixs)
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
                    self.distances[droplet.name] = droplet.distance_to_goal
                    self._add_droplet(droplet)
        self.colors = COLORS(len(self.classes))
        for droplet_name, droplet_idx in self.droplets_idx.items():
            self.droplets[droplet_idx].set_color(
                self.colors.agent_colors[
                    self.classes_idx[self.droplets[droplet_idx].class_name]
                ]
            )
        self.goals = [zone.location.tolist() for zone in self.zones]
        self.frontiers = []
        for zone in self.zones:
            self.frontiers.append([])

        self.droplets_to_move = []
        self.droplets_waiting = []

    def _add_class(self, class_name):
        self.classes_idx[class_name] = len(self.classes)
        self.classes.append(class_name)
        m_obstacles = np.zeros((self.height + 2, self.width + 2), dtype=bool)
        m_obstacles[:, [0, -1]] = True
        m_obstacles[[0, -1], :] = True
        self.m_class_obstacles = np.concatenate(
            (self.m_class_obstacles, np.expand_dims(m_obstacles, axis=0)), axis=0
        )
        m_trajectory = np.zeros((self.height + 2, self.width + 2), dtype=bool)
        self.m_class_trajectory = np.concatenate(
            (self.m_class_trajectory, np.expand_dims(m_trajectory, axis=0)), axis=0
        )


    def _add_droplet(self, droplet):
        assert droplet.class_name in self.classes_idx
        self.droplets_idx[droplet.name] = len(self.droplets)
        self.droplets.append(droplet)
        m_trajectory = np.zeros((self.height + 2, self.width + 2), dtype=bool)
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
        self.m_droplet_obstacles = np.concatenate(
            (self.m_droplet_obstacles, np.expand_dims(m_obstacles, axis=0)), axis=0
        )

    def _get_out_of_zone(self, zone: Zone, new_zone: Zone):
        c = zone.center
        new_c = new_zone.center
        v = (new_c[0] - c[0], new_c[1] - c[1])
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
    
    def get_routing(self, routing_dict):
        assert set(routing_dict.keys()) == set(self.droplets_idx.keys())
        self.droplets_ways = routing_dict
    
    def _get_droplet_obstacles(self, droplet_name):
        """
        返回两个矩阵，第一个矩阵表示droplet_name的实体障碍物，第二个矩阵表示droplet_name的障碍物中的其他droplet的轨迹。
        第二个矩阵的障碍是可以擦除的。
        """
        droplet_idx = self.droplets_idx[droplet_name]
        class_name = droplet_name.split("_")[0]
        class_idx = self.classes_idx[class_name]
        other_classes = [i for i in range(len(self.classes)) if i != class_idx]
        m = self.m_droplet_obstacles[droplet_idx]
        for idx in self.droplets_idx_to_move:
            p = self.droplets[idx].extended_boundaries
            m[p[0] : p[2] + 1, p[1] : p[3] + 1] = True
        m_contaminated = self.m_class_trajectory[other_classes].any(axis=0)
        # m = np.logical_or(m, m_contaminated)
        return m, m_contaminated

    @property
    def droplets_idx_to_move(self):
        return [self.droplets_idx[droplet_name] for droplet_name in self.droplets_to_move]
    
    @property
    def droplets_idx_to_wait(self):
        return [self.droplets_idx[droplet_name] for droplet_name in self.droplets_waiting]

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